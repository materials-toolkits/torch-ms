"""Build the Materials Project dataset in HDF5 format.

The summary collection is downloaded, converted to CIF, preprocessed without
any filter (deprecated materials are not part of the dataset) and every
structure is checked against its source. Run inside the project environment:

    uv run python scripts/build_materials_project.py datasets
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

from materials_project_to_cif import (
    URL,
    DisorderedStructure,
    convert,
    download,
    iter_rows,
    site_symbols,
)
from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.preprocessing import Preprocessing, preprocessing

LABELS = {
    "formation_energy_per_atom": "formation_energy_per_atom",
    "energy_above_hull": "energy_above_hull",
    "band_gap": "band_gap",
}


def check(condition: bool, message: str) -> None:
    """Stop the build when a verification fails."""
    if not condition:
        raise RuntimeError(f"verification failed: {message}")


def column(rows: list[dict[str, Any]], name: str) -> torch.Tensor:
    """Get a column of the source as a tensor, NaN when the value is missing."""
    return torch.tensor(
        [float("nan") if row[name] is None else row[name] for row in rows],
        dtype=torch.float64,
    )


def verify_chunk(rows: list[dict[str, Any]], structures: Structures, start: int):
    """Compare consecutive structures of the dataset with their source rows."""
    where = f"structures {start}-{start + len(rows) - 1}"

    symbols = [site_symbols(row) for row in rows]
    num_atoms = torch.tensor([len(s) for s in symbols])
    check(torch.equal(structures.num_atoms, num_atoms), f"{where}: number of atoms")

    z = torch.tensor([symbol_to_number(s) for struct in symbols for s in struct])
    check(torch.equal(structures.z, z), f"{where}: elements")

    frac = np.concatenate(
        [[site["abc"] for site in row["structure"]["sites"]] for row in rows]
    )
    diff = np.abs(structures.x.double().numpy() - frac % 1.0)
    diff = np.minimum(diff, 1.0 - diff)
    check(diff.max() < 1e-5, f"{where}: coordinates differ by {diff.max():.2e}")

    # the lattice is rebuilt in the standard orientation: compare the metric
    lattice = np.array([row["structure"]["lattice"]["matrix"] for row in rows])
    gram = lattice @ lattice.transpose(0, 2, 1)
    stored = structures.lattice.double().numpy()
    error = np.abs(stored @ stored.transpose(0, 2, 1) - gram).max(axis=(1, 2))
    check(
        bool((error < 1e-4 * gram.max(axis=(1, 2))).all()),
        f"{where}: lattice",
    )

    for label, name in LABELS.items():
        check(
            torch.allclose(
                getattr(structures, label).double(),
                column(rows, name),
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            f"{where}: {label}",
        )

    check(
        torch.allclose(
            structures.total_energy.double() / num_atoms,
            column(rows, "energy_per_atom"),
            rtol=1e-6,
            atol=1e-5,
            equal_nan=True,
        ),
        f"{where}: total_energy",
    )


def verify(source: Path, hdf5_file: Path, chunk_size: int = 2048) -> dict[str, int]:
    """Compare every structure of the dataset with the collection.

    Returns:
        Number of structures, atoms and structures without labels.
    """
    dataset = HDF5Dataset(hdf5_file)
    stats = {"structures": 0, "atoms": 0, "max_atoms": 0, "unlabelled": 0}
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        start = stats["structures"]
        check(start + len(rows) <= len(dataset), "dataset is shorter than its source")
        structures = dataset[start : start + len(rows)]
        verify_chunk(rows, structures, start)

        stats["structures"] += len(rows)
        stats["atoms"] += int(structures.num_atoms.sum())
        stats["max_atoms"] = max(stats["max_atoms"], int(structures.num_atoms.max()))
        stats["unlabelled"] += int(structures.total_energy.isnan().sum())
        rows.clear()

    for row in tqdm.tqdm(iter_rows(source), desc="verifying", unit="struct"):
        if row["deprecated"]:
            continue

        try:
            site_symbols(row)
        except DisorderedStructure:
            continue

        rows.append(row)
        if len(rows) == chunk_size:
            flush()

    if len(rows) > 0:
        flush()

    check(
        stats["structures"] == len(dataset),
        f'{len(dataset)} structures in the dataset, {stats["structures"]} in its source',
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    root = Path(args.destination)
    source = root / "raw" / "materials_project" / "summary.parquet"
    cif_file = root / "cif" / "materials_project" / "all.cif"
    hdf5_file = root / "materials_project" / "all.hdf5"

    for file_name in (source, cif_file, hdf5_file):
        file_name.parent.mkdir(parents=True, exist_ok=True)

    download(args.url, source)
    counts = convert(source, cif_file)
    preprocessing(Preprocessing(source=cif_file, destination=hdf5_file, batch_size=256))
    stats = verify(source, hdf5_file)

    check(stats["structures"] == counts["written"], "structures lost in preprocessing")

    print(
        f'{stats["structures"]} structures ({stats["atoms"]} atoms, up to '
        f'{stats["max_atoms"]} per structure), '
        f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}"
    )
    print(
        f'skipped: {counts["deprecated"]} deprecated, {counts["disordered"]} disordered; '
        f'kept without labels: {stats["unlabelled"]}'
    )
    print("every structure verified against its source")


if __name__ == "__main__":
    main()
