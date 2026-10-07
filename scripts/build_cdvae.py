"""Build the CDVAE datasets (Perov-5, Carbon-24, MP-20) in HDF5 format.

Each split is downloaded, converted to CIF, preprocessed without any filter and
checked against its source. Run inside the project environment:

    uv run python scripts/build_cdvae.py datasets
"""

import argparse
import csv
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from csv_to_cif import convert
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.preprocessing import Preprocessing, preprocessing

URL = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/{name}/{split}.csv"
SPLITS = ("train", "val", "test")
LABELS = (
    "total_energy",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
)


@dataclass
class Source:
    """Columns of a dataset mapped to the labels of the structures."""

    name: str
    num_structures: int
    energy_per_atom_col: str | None = None
    formation_energy_col: str | None = None
    hull_col: str | None = None
    band_gap_col: str | None = None

    def columns(self) -> dict[str, str | None]:
        """Source column of each label, None if the label is not available."""
        return {
            "total_energy": self.energy_per_atom_col,
            "formation_energy_per_atom": self.formation_energy_col,
            "energy_above_hull": self.hull_col,
            "band_gap": self.band_gap_col,
        }


SOURCES = (
    Source("perov_5", 18928, formation_energy_col="heat_ref", band_gap_col="ind_gap"),
    Source("carbon_24", 10153, energy_per_atom_col="energy_per_atom"),
    Source(
        "mp_20",
        45229,  # the CDVAE paper reports 45,231, the released files hold 45,229
        formation_energy_col="formation_energy_per_atom",
        hull_col="e_above_hull",
        band_gap_col="band_gap",
    ),
)


def check(condition: bool, message: str) -> None:
    """Stop the build when a verification fails."""
    if not condition:
        raise RuntimeError(f"verification failed: {message}")


def verify(source: Source, csv_file: Path, hdf5_file: Path, num_samples: int) -> int:
    """Compare a preprocessed split with its source, row by row.

    Returns:
        Number of structures of the split.
    """
    with open(csv_file, "r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    dataset = HDF5Dataset(hdf5_file)
    where = f"{source.name}/{csv_file.stem}"

    check(len(dataset) == len(rows), f"{where}: {len(dataset)} != {len(rows)} rows")

    structures = dataset[:]

    for label, column in source.columns().items():
        values = getattr(structures, label).double()

        if column is None:
            check(bool(values.isnan().all()), f"{where}: {label} must be unknown")
            continue

        expected = torch.tensor([float(row[column]) for row in rows]).double()
        if label == "total_energy":
            values = values / structures.num_atoms

        check(
            torch.allclose(values, expected, rtol=1e-6, atol=5e-5),
            f"{where}: {label} differs from column {column}",
        )

    for i in np.linspace(0, len(rows) - 1, min(num_samples, len(rows)), dtype=int):
        reference = Structure.from_str(rows[i]["cif"], fmt="cif")
        struct = dataset[int(i)]

        # pymatgen may reorder atoms: match each atom with the closest
        # reference atom of the same element
        z_ref = np.array(reference.atomic_numbers)
        x_ref = reference.frac_coords % 1.0
        z = struct.z.numpy()

        check(sorted(z.tolist()) == sorted(z_ref.tolist()), f"{where}[{i}]: elements")

        dist = np.abs(struct.x.double().numpy()[:, None] - x_ref[None])
        dist = np.minimum(dist, 1.0 - dist).max(axis=-1)
        dist[z[:, None] != z_ref[None]] = np.inf
        check(
            len(set(dist.argmin(axis=1).tolist())) == len(z_ref),
            f"{where}[{i}]: atoms are not matched one to one",
        )
        check(dist.min(axis=1).max() < 1e-4, f"{where}[{i}]: coordinates")

        lattice = struct.lattice[0].double().numpy()
        gram_ref = reference.lattice.matrix @ reference.lattice.matrix.T
        check(
            np.abs(lattice @ lattice.T - gram_ref).max() < 1e-3 * gram_ref.max(),
            f"{where}[{i}]: lattice",
        )

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="structures per split compared with pymatgen",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", module="pymatgen")
    csv.field_size_limit(sys.maxsize)

    root = Path(args.destination)
    summary = []

    for source in SOURCES:
        total = 0

        for split in SPLITS:
            csv_file = root / "raw" / source.name / f"{split}.csv"
            cif_file = root / "cif" / source.name / f"{split}.cif"
            hdf5_file = root / source.name / f"{split}.hdf5"

            for file_name in (csv_file, cif_file, hdf5_file):
                file_name.parent.mkdir(parents=True, exist_ok=True)

            if not csv_file.exists():
                url = URL.format(name=source.name, split=split)
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, csv_file)

            convert(
                csv_file,
                cif_file,
                energy_per_atom_col=source.energy_per_atom_col,
                formation_energy_col=source.formation_energy_col,
                hull_col=source.hull_col,
                band_gap_col=source.band_gap_col,
            )
            preprocessing(
                Preprocessing(source=cif_file, destination=hdf5_file, batch_size=256)
            )

            count = verify(source, csv_file, hdf5_file, args.num_samples)
            total += count

            num_atoms = HDF5Dataset(hdf5_file)[:].num_atoms
            summary.append(
                f"{source.name:10s} {split:5s} {count:6d} structures, "
                f"{int(num_atoms.min())}-{int(num_atoms.max())} atoms, "
                f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}"
            )

        check(
            total == source.num_structures,
            f"{source.name}: {total} != {source.num_structures} structures",
        )

    print("\n".join(summary))
    print("all splits verified against their source")


if __name__ == "__main__":
    main()
