"""Build the LeMat-Bulk dataset (PBE) in HDF5 format.

The dataset is read from Hugging Face, converted to CIF, preprocessed without
any filter and every structure is checked against its source. Run inside the
project environment:

    uv run python scripts/build_lematerial.py datasets            # all.hdf5
    uv run python scripts/build_lematerial.py datasets --unique   # unique.hdf5
    uv run python scripts/build_lematerial.py datasets --hull     # hull.hdf5

With --unique, only the structure with the lowest energy per atom is kept
among the structures sharing the same entalpic fingerprint.

The energy above the hull comes from LeMat-Bulk-DFT-Hull-All. The few
structures without one get the energy above the convex hull computed from the
total energies of the dataset (see torchms.data.hull).

With --hull, the composition and the energy of the structures on the convex
hull are saved, to compute the energy above the hull of other structures. This
hull is built from the structures of the Materials Project and Alexandria:
the energies of OQMD are not comparable with them for some elements, see
build_hull. The energy above the hull stored in all.hdf5 and unique.hdf5 is
the published one, which mixes the three sources.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm

from lematerial_to_cif import (
    HULL_REPO,
    HULL_REVISION,
    REPO,
    REVISION,
    HullLookup,
    convert,
    fractional,
    iter_rows,
    load_unique_mask,
    parquet_files,
)
from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.hull import (
    energy_above_reference,
    fill_missing,
    load_hull,
    save_hull,
)
from torchms.data.preprocessing import Preprocessing, preprocessing

SUBSET = "compatible_pbe"


def check(condition: bool, message: str) -> None:
    """Stop the build when a verification fails."""
    if not condition:
        raise RuntimeError(f"verification failed: {message}")


def verify_chunk(
    rows: list[dict[str, Any]],
    structures: Structures,
    hull: HullLookup,
    computed: np.ndarray,
    start: int,
    numbers: dict[str, int],
) -> None:
    """Compare consecutive structures of the dataset with their source rows.

    computed is the energy above the hull computed from the dataset, expected
    for the structures without one in the source.
    """
    where = f"structures {start}-{start + len(rows) - 1}"

    num_atoms = torch.tensor([len(row["species_at_sites"]) for row in rows])
    check(torch.equal(structures.num_atoms, num_atoms), f"{where}: number of atoms")

    for row in rows:
        for symbol in row["species_at_sites"]:
            if symbol not in numbers:
                numbers[symbol] = symbol_to_number(symbol)

    z = torch.tensor([numbers[s] for row in rows for s in row["species_at_sites"]])
    check(torch.equal(structures.z, z), f"{where}: elements")

    cells = [fractional(row) for row in rows]
    frac = np.concatenate([frac for _, frac in cells]) % 1.0
    diff = np.abs(structures.x.double().numpy() - frac)
    diff = np.minimum(diff, 1.0 - diff)
    check(diff.max() < 1e-5, f"{where}: coordinates differ by {diff.max():.2e}")

    # the lattice is rebuilt in the standard orientation: compare the metric
    lattice = np.stack([lattice for lattice, _ in cells])
    gram = lattice @ lattice.transpose(0, 2, 1)
    stored = structures.lattice.double().numpy()
    error = np.abs(stored @ stored.transpose(0, 2, 1) - gram).max(axis=(1, 2))
    check(bool((error < 1e-4 * gram.max(axis=(1, 2))).all()), f"{where}: lattice")

    energy = torch.tensor([row["energy"] for row in rows], dtype=torch.float64)
    check(
        torch.allclose(structures.total_energy.double(), energy, rtol=1e-6, atol=1e-5),
        f"{where}: total_energy",
    )

    e_above_hull = hull([row["immutable_id"] or "" for row in rows])
    e_above_hull = np.where(np.isnan(e_above_hull), computed, e_above_hull)
    check(
        torch.allclose(
            structures.energy_above_hull.double(),
            torch.from_numpy(e_above_hull),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ),
        f"{where}: energy_above_hull",
    )

    check(bool(structures.band_gap.isnan().all()), f"{where}: band_gap")
    check(
        bool(structures.formation_energy_per_atom.isnan().all()),
        f"{where}: formation_energy_per_atom",
    )


def verify(
    files: list[str],
    keep: np.ndarray | None,
    hull: HullLookup,
    computed: np.ndarray,
    hdf5_file: Path,
    chunk_size: int = 4096,
) -> tuple[list[str | None], dict[str, int]]:
    """Compare every structure of the dataset with LeMat-Bulk.

    computed is the energy above the hull computed from the dataset, expected
    for the structures without one in LeMat-Bulk-DFT-Hull-All.

    Returns:
        The immutable_id of each structure and statistics of the dataset.
    """
    dataset = HDF5Dataset(hdf5_file)
    stats = {"structures": 0, "atoms": 0, "max_atoms": 0, "stable": 0, "metastable": 0}
    ids: list[str | None] = []
    rows: list[dict[str, Any]] = []
    numbers: dict[str, int] = {}

    def flush() -> None:
        start = stats["structures"]
        check(start + len(rows) <= len(dataset), "dataset is shorter than its source")
        structures = dataset[start : start + len(rows)]
        verify_chunk(
            rows, structures, hull, computed[start : start + len(rows)], start, numbers
        )

        ids.extend(row["immutable_id"] for row in rows)
        stats["structures"] += len(rows)
        stats["atoms"] += int(structures.num_atoms.sum())
        stats["max_atoms"] = max(stats["max_atoms"], int(structures.num_atoms.max()))
        stats["stable"] += int((structures.energy_above_hull <= 0.0).sum())
        stats["metastable"] += int((structures.energy_above_hull <= 0.1).sum())
        rows.clear()

    for row in tqdm.tqdm(iter_rows(files, keep), desc="verifying", unit="struct"):
        rows.append(row)
        if len(rows) == chunk_size:
            flush()

    if len(rows) > 0:
        flush()

    check(
        stats["structures"] == len(dataset),
        f'{len(dataset)} structures in the dataset, {stats["structures"]} in its source',
    )

    return ids, stats


# Sources of LeMat-Bulk whose structures define the convex hull, from the prefix
# of their immutable_id: the Materials Project and Alexandria. OQMD is left out,
# see build_hull.
HULL_SOURCES = ("mp-", "agm")


def build_hull(files: list[str], hull_file: Path, num_workers: int) -> dict[str, Any]:
    """Save the convex hull of LeMat-Bulk, without the structures from OQMD.

    LeMat-Bulk gathers structures from the Materials Project, Alexandria and
    OQMD. The total energies of OQMD are not on the scale of the two others
    for some elements: compared with them, a structure from OQMD is shifted
    by about +9.4 eV per atom of Gd, +0.38 Cu, +0.20 Ni, +0.19 Pu, +0.16 Pa,
    +0.14 Fe, +0.07 Np, +0.05 Re and -0.19 Bi (measured with the energies
    predicted by orb-v3, see build_energy_correction.py; the shifts of Gd, Cu
    and Ni are also the differences between the elemental structures of the
    sources). A hull mixing the three sources is wrong for these elements:
    with a negative shift (Bi), structures from OQMD are on the hull only
    because of the shift, and hide the stable structures of the other sources.

    The hull is therefore built from the structures of the Materials Project
    and Alexandria only. Structures from OQMD and the structure without
    identifier are compared to it afterwards, but never define it.

    The hull is saved to hull_file and the immutable_id of its structures to
    "hull_index.parquet". Every structure of the two sources is then compared
    to the saved hull, which must give back its energy above the hull.

    Returns:
        Statistics of the hull.
    """
    table = pa.concat_tables(
        [
            pq.read_table(f, columns=["immutable_id", "species_at_sites", "energy"])
            for f in files
        ]
    )
    species = table["species_at_sites"].combine_chunks()
    symbols = species.flatten().dictionary_encode()
    numbers = np.array([symbol_to_number(s) for s in symbols.dictionary.to_pylist()])

    z = numbers[symbols.indices.to_numpy()]
    num_atoms = np.diff(species.offsets.to_numpy())
    energy = table["energy"].to_numpy(zero_copy_only=False).astype(np.float64)

    ids = table["immutable_id"].to_pandas().fillna("")
    kept = ids.str.startswith(HULL_SOURCES).to_numpy()
    kept_atoms = np.repeat(kept, num_atoms)
    reference = (z[kept_atoms], num_atoms[kept], energy[kept])

    above = save_hull(hull_file, *reference, num_workers, progress=True)
    on_hull = above == 0.0

    pq.write_table(
        table.select(["immutable_id"]).filter(pa.array(kept)).filter(pa.array(on_hull)),
        hull_file.with_name("hull_index.parquet"),
        compression="zstd",
    )

    hull = load_hull(hull_file, num_workers)
    again = energy_above_reference(hull, *reference, num_workers, progress=True)
    difference = float(np.abs(again - above).max())
    check(difference < 1e-9, f"saved hull differs from the dataset by {difference:.2e}")
    check(bool(((again == 0.0) == on_hull).all()), "saved hull has other structures")

    # structures which do not define the hull, compared to it
    left_atoms = np.repeat(~kept, num_atoms)
    left = energy_above_reference(
        hull, z[left_atoms], num_atoms[~kept], energy[~kept], num_workers, progress=True
    )

    return {
        "source_structures": len(num_atoms),
        "structures_defining_the_hull": int(kept.sum()),
        "structures_on_hull": int(on_hull.sum()),
        "chemical_systems_with_stable_structures": len(hull.stable),
        "elements": int((~np.isnan(hull.reference)).sum()),
        "max_difference_from_saved_hull": difference,
        "structures_left_out": int((~kept).sum()),
        "left_out_below_the_hull": int((left < 0).sum()),
        "left_out_more_than_0.1_below_the_hull": int((left < -0.1).sum()),
        "left_out_without_energy_above_hull": int(np.isnan(left).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="keep the lowest energy structure of each entalpic fingerprint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the existing HDF5 file: only fill, verify and write the report",
    )
    parser.add_argument(
        "--hull",
        action="store_true",
        help="only save the convex hull of the dataset (hull.hdf5)",
    )
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    name = "unique" if args.unique else "all"
    root = Path(args.destination)

    if args.hull:
        hull_file = root / "lematerial" / "hull.hdf5"
        hull_file.parent.mkdir(parents=True, exist_ok=True)

        files = parquet_files(REPO, f"{SUBSET}/*.parquet", REVISION)
        report = build_hull(files, hull_file, args.num_workers)

        with open(hull_file.with_name("hull_report.json"), "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        print(json.dumps(report, indent=2))
        print(f"{hull_file.stat().st_size / 1e6:.2f} MB  {hull_file}")
        return
    cif_file = root / "cif" / "lematerial" / f"{name}.cif"
    hdf5_file = root / "lematerial" / f"{name}.hdf5"

    for file_name in (cif_file, hdf5_file):
        file_name.parent.mkdir(parents=True, exist_ok=True)

    files = parquet_files(REPO, f"{SUBSET}/*.parquet", REVISION)
    hull = HullLookup.from_files(
        parquet_files(HULL_REPO, "data/*.parquet", HULL_REVISION), "dft_hull"
    )

    num_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep = load_unique_mask(files) if args.unique else None

    report = {
        "source_structures": num_rows,
        "hull_repeated_ids": hull.num_repeated,
        "hull_repeated_ids_max_disagreement": hull.max_disagreement,
    }
    if keep is not None:
        report["structures_dropped_as_duplicates"] = int(num_rows - keep.sum())

    if not args.resume:
        counts = convert(files, cif_file, hull, keep, desc=f"converting {name}")
        preprocessing(
            Preprocessing(source=cif_file, destination=hdf5_file, batch_size=256)
        )
        report.update(counts)

    computed, fill_stats = fill_missing(hdf5_file, args.num_workers, progress=True)
    check(fill_stats["still_missing"] == 0, "structures without energy above the hull")
    check(
        fill_stats["max_difference"] < 1e-3,
        "computed energy above the hull differs from LeMat-Bulk-DFT-Hull-All by "
        f'{fill_stats["max_difference"]:.2e} eV/atom',
    )
    report.update({f"hull_{key}": value for key, value in fill_stats.items()})

    ids, stats = verify(files, keep, hull, computed, hdf5_file)

    expected = num_rows if keep is None else int(keep.sum())
    check(stats["structures"] == expected, "structures lost in preprocessing")

    report.update(stats)

    pq.write_table(
        pa.table({"immutable_id": pa.array(ids, type=pa.string())}),
        hdf5_file.with_name(f"{name}_index.parquet"),
        compression="zstd",
    )
    with open(hdf5_file.with_name(f"{name}_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report, indent=2))
    print(f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}")
    print("every structure verified against its source")


if __name__ == "__main__":
    main()
