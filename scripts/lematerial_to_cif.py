# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "numpy", "pyarrow", "tqdm"]
# ///
"""Convert LeMat-Bulk (Hugging Face) to a single concatenated CIF file.

The output is the input of the preprocessing CLI:

    uv run scripts/lematerial_to_cif.py lemat.cif --limit 10000
    uv run preprocessing lemat.cif lemat.hdf5 --max-atoms 64

Total energy comes from LeMat-Bulk ("energy", eV) and the energy above the hull
from LeMat-Bulk-DFT-Hull-All ("dft_hull", eV/atom), joined on "immutable_id".
With --unique, only the structure with the lowest energy per atom is kept among
the structures sharing the same "entalpic_fingerprint".
"""

import argparse
import glob
import os
import re
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm
from huggingface_hub import snapshot_download

from _cif_writer import write_block

REPO = "LeMaterial/LeMat-Bulk"
REVISION = "0dc17eea904b860ad7288141e9870f67f8e6bb2c"
HULL_REPO = "LeMaterial/LeMat-Bulk-DFT-Hull-All"
HULL_REVISION = "fc063a965498df6481911bdfdb3cad5619016d81"

COLUMNS = [
    "immutable_id",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "energy",
]


def parquet_files(repo: str, pattern: str, revision: str | None = None) -> list[str]:
    """Download (or reuse from the cache) the parquet files of a dataset."""
    root = snapshot_download(
        repo, repo_type="dataset", allow_patterns=pattern, revision=revision
    )
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if len(files) == 0:
        raise RuntimeError(f'no file matching "{pattern}" in {repo}')
    return files


class HullLookup:
    """Energy above the hull indexed by immutable_id.

    When an id is repeated, its first value is used.

    Args:
        ids: Identifier of each row.
        values: Energy above the hull of each row, in eV/atom.
    """

    def __init__(self, ids: np.ndarray, values: np.ndarray):
        ids = np.asarray(ids).astype("S")
        values = np.asarray(values, dtype=np.float64)

        order = np.argsort(ids, kind="stable")
        self.ids = ids[order]
        self.values = values[order]

        repeated = self.ids[1:] == self.ids[:-1]
        self.num_repeated = int(repeated.sum())
        self.max_disagreement = 0.0
        if self.num_repeated > 0:
            self.max_disagreement = float(
                np.abs(self.values[1:] - self.values[:-1])[repeated].max()
            )

    @classmethod
    def from_files(cls, files: list[str], column: str) -> "HullLookup":
        """Load the lookup from parquet files with an "immutable_id" column."""
        table = pa.concat_tables(
            [pq.read_table(f, columns=["immutable_id", column]) for f in files]
        )
        return cls(
            table["immutable_id"].to_numpy(zero_copy_only=False),
            table[column].to_numpy(zero_copy_only=False),
        )

    def __call__(self, ids: list[str]) -> np.ndarray:
        if len(ids) == 0:
            return np.zeros(0, dtype=np.float64)

        # not cast to the dtype of self.ids: longer ids would be truncated
        query = np.array(ids).astype("S")
        pos = np.searchsorted(self.ids, query, side="left").clip(max=len(self.ids) - 1)
        return np.where(self.ids[pos] == query, self.values[pos], np.nan)


def unique_mask(fingerprints: np.ndarray, energy_per_atom: np.ndarray) -> np.ndarray:
    """Select the structure with the lowest energy of each fingerprint.

    When energies are equal the first structure is kept. Structures without
    fingerprint (None) are all kept.

    Returns:
        Mask of the structures to keep.
    """
    fingerprints = np.asarray(fingerprints, dtype=object)
    missing = np.array([f is None for f in fingerprints])
    keep = missing.copy()

    index = np.flatnonzero(~missing)
    _, group = np.unique(fingerprints[index].astype(str), return_inverse=True)

    # sorted by fingerprint, then energy, then position: first of each group
    order = np.lexsort((index, np.asarray(energy_per_atom)[index], group))
    first = np.ones(len(order), dtype=bool)
    first[1:] = group[order][1:] != group[order][:-1]
    keep[index[order[first]]] = True

    return keep


def load_unique_mask(files: list[str]) -> np.ndarray:
    """Compute unique_mask over all the rows of the dataset."""
    # the number of atoms is the length of "species_at_sites": the column
    # "nsites" is not consistent with it for every structure
    table = pa.concat_tables(
        [
            pq.read_table(
                f, columns=["entalpic_fingerprint", "energy", "species_at_sites"]
            )
            for f in files
        ]
    )
    energy = table["energy"].to_numpy(zero_copy_only=False).astype(np.float64)
    num_atoms = pc.list_value_length(table["species_at_sites"]).to_numpy()

    return unique_mask(
        table["entalpic_fingerprint"].to_numpy(zero_copy_only=False),
        energy / num_atoms,
    )


def iter_rows(
    files: list[str], keep: np.ndarray | None = None, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Iterate over the structures of the dataset.

    Args:
        files: Parquet files of the dataset, in order.
        keep: Optional mask over all the rows, structures not kept are skipped.
        batch_size: Number of rows read at once.
    """
    offset = 0

    for file_name in files:
        parquet = pq.ParquetFile(file_name)

        for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
            num_rows = batch.num_rows

            if keep is not None:
                batch = batch.filter(pa.array(keep[offset : offset + num_rows]))
            offset += num_rows

            yield from batch.to_pylist()


def fractional(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Get the lattice (rows are vectors) and the fractional coordinates."""
    lattice = np.array(row["lattice_vectors"], dtype=np.float64)
    cart = np.array(row["cartesian_site_positions"], dtype=np.float64)
    return lattice, cart @ np.linalg.inv(lattice)


def block_name(row: dict[str, Any], index: int) -> str:
    """Name of the CIF data block of a structure."""
    if row["immutable_id"] is None:
        return f"row-{index}"
    return re.sub(r"\s+", "_", row["immutable_id"])


def convert(
    files: list[str],
    destination: str | os.PathLike,
    hull: HullLookup | None = None,
    keep: np.ndarray | None = None,
    limit: int | None = None,
    desc: str = "converting",
) -> dict[str, int]:
    """Write the structures of the dataset to a single CIF file.

    Args:
        files: Parquet files of the dataset, in order.
        destination: Output file (cif format).
        hull: Optional energy above the hull of the structures.
        keep: Optional mask over all the rows, structures not kept are skipped.
        limit: Maximum number of structures written.
        desc: Description of the progress bar.

    Returns:
        Number of structures written and written without energy above the hull.
    """
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if keep is not None:
        total = int(keep.sum())
    if limit is not None:
        total = min(total, limit)

    counts = {"written": 0, "without_energy_above_hull": 0}
    rows: list[dict[str, Any]] = []

    def flush(fp) -> None:
        ids = [row["immutable_id"] or "" for row in rows]
        e_above_hull = hull(ids) if hull is not None else np.full(len(rows), np.nan)
        counts["without_energy_above_hull"] += int(np.isnan(e_above_hull).sum())

        for row, energy_above_hull in zip(rows, e_above_hull):
            lattice, frac = fractional(row)
            write_block(
                fp,
                block_name(row, counts["written"]),
                lattice,
                frac,
                row["species_at_sites"],
                total_energy=row["energy"],
                energy_above_hull=float(energy_above_hull),
            )
            counts["written"] += 1

        rows.clear()

    with open(destination, "w", encoding="utf-8", buffering=1 << 20) as fp:
        progress = tqdm.tqdm(
            iter_rows(files, keep), total=total, desc=desc, unit="struct"
        )
        for row in progress:
            if counts["written"] + len(rows) >= total:
                break

            rows.append(row)
            if len(rows) == 1024:
                flush(fp)

        flush(fp)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--subset", default="compatible_pbe")
    parser.add_argument(
        "--hull-repo",
        default=HULL_REPO,
        help='dataset with the energy above the hull, "" to disable',
    )
    parser.add_argument("--hull-revision", default=HULL_REVISION)
    parser.add_argument("--hull-column", default="dft_hull")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="keep the lowest energy structure of each entalpic fingerprint",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="maximum number of structures"
    )
    args = parser.parse_args()

    files = parquet_files(args.repo, f"{args.subset}/*.parquet", args.revision)

    hull = None
    if args.hull_repo:
        hull = HullLookup.from_files(
            parquet_files(args.hull_repo, "data/*.parquet", args.hull_revision),
            args.hull_column,
        )

    keep = load_unique_mask(files) if args.unique else None
    counts = convert(files, args.destination, hull, keep, args.limit)

    print(f'{counts["written"]} structures written to {args.destination}')
    if hull is not None:
        print(
            f'{counts["without_energy_above_hull"]} structures without energy '
            "above the hull"
        )


if __name__ == "__main__":
    main()
