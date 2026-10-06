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
"""

import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq
import tqdm
from huggingface_hub import snapshot_download

from _cif_writer import write_block

COLUMNS = [
    "immutable_id",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "energy",
]


def parquet_files(repo: str, pattern: str) -> list[str]:
    """Download (or reuse from the cache) the parquet files of a dataset."""
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=pattern)
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if len(files) == 0:
        raise RuntimeError(f'no file matching "{pattern}" in {repo}')
    return files


class HullLookup:
    """Energy above the hull indexed by immutable_id."""

    def __init__(self, repo: str, column: str):
        ids, values = [], []
        for file_name in parquet_files(repo, "data/*.parquet"):
            table = pq.read_table(file_name, columns=["immutable_id", column])
            ids.append(table["immutable_id"].to_numpy().astype("S"))
            values.append(table[column].to_numpy())

        ids = np.concatenate(ids)
        values = np.concatenate(values)

        order = np.argsort(ids)
        self.ids = ids[order]
        self.values = values[order]

    def __call__(self, ids: list[str]) -> np.ndarray:
        query = np.array(ids).astype(self.ids.dtype)
        pos = np.searchsorted(self.ids, query).clip(max=len(self.ids) - 1)
        return np.where(self.ids[pos] == query, self.values[pos], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument("--repo", default="LeMaterial/LeMat-Bulk")
    parser.add_argument("--subset", default="compatible_pbe")
    parser.add_argument(
        "--hull-repo",
        default="LeMaterial/LeMat-Bulk-DFT-Hull-All",
        help='dataset with the energy above the hull, "" to disable',
    )
    parser.add_argument("--hull-column", default="dft_hull")
    parser.add_argument(
        "--limit", type=int, default=None, help="maximum number of structures"
    )
    args = parser.parse_args()

    files = parquet_files(args.repo, f"{args.subset}/*.parquet")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if args.limit is not None:
        total = min(total, args.limit)

    hull = HullLookup(args.hull_repo, args.hull_column) if args.hull_repo else None

    progress = tqdm.tqdm(total=total, desc=f"{args.repo}/{args.subset}", unit="struct")
    count = 0
    missing_hull = 0

    with open(args.destination, "w", encoding="utf-8", buffering=1 << 20) as fp:
        for file_name in files:
            parquet = pq.ParquetFile(file_name)

            for batch in parquet.iter_batches(batch_size=1024, columns=COLUMNS):
                rows = batch.to_pydict()
                ids = rows["immutable_id"][: total - count]

                if hull is not None:
                    e_above_hull = hull(ids)
                    missing_hull += int(np.isnan(e_above_hull).sum())
                else:
                    e_above_hull = np.full(len(ids), np.nan)

                for i, name in enumerate(ids):
                    lattice = np.array(rows["lattice_vectors"][i], dtype=np.float64)
                    cart = np.array(
                        rows["cartesian_site_positions"][i], dtype=np.float64
                    )

                    write_block(
                        fp,
                        name,
                        lattice,
                        cart @ np.linalg.inv(lattice),
                        rows["species_at_sites"][i],
                        total_energy=rows["energy"][i],
                        energy_above_hull=float(e_above_hull[i]),
                    )

                count += len(ids)
                progress.update(len(ids))

                if count >= total:
                    break

            if count >= total:
                break

    progress.close()
    print(f"{count} structures written to {args.destination}")
    if hull is not None:
        print(f"{missing_hull} structures without energy above the hull")


if __name__ == "__main__":
    main()
