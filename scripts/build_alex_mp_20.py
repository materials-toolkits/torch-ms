"""Build Alex-MP-20, the training set of MatterGen, in HDF5 format.

The structures of Alexandria and of MP-20 that MatterGen was trained on, as its
authors release them (data-release/alex-mp of the repository of MatterGen,
CC BY 4.0 for Alexandria): at most 20 atoms, at most 0.1 eV/atom above the
hull, relaxed with PBE. Each split is converted to CIF, preprocessed without
any filter, given the other columns of the source as named properties, and
checked against its source. Run inside the project environment:

    uv run python scripts/build_alex_mp_20.py datasets
"""

import argparse
import csv
import hashlib
import sys
import urllib.request
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import torch

from build_cdvae import Source, check, verify
from csv_to_cif import convert, count_atoms
from torchms.data.constants import get_spacegroup
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.preprocessing import Preprocessing, preprocessing

# the archive of MatterGen at the commit of its port in torchms, a file of git LFS
URL = (
    "https://media.githubusercontent.com/media/microsoft/mattergen/"
    "92423660a8bd70e83679086e88f88596d484dc16/data-release/alex-mp/alex_mp_20.zip"
)
SHA256 = "6688e652e45fac4ac8b016dbc4a3280afde9ba687bdacfd8c676775ed19b6228"
SPLITS = ("train", "val")

SOURCE = Source(
    "alex_mp_20", 675204, hull_col="energy_above_hull", band_gap_col="dft_band_gap"
)
# named properties, by source column: the space group is stored by its number
PROPERTIES = (
    "space_group",
    "dft_bulk_modulus",
    "ml_bulk_modulus",
    "dft_mag_density",
    "hhi_score",
)
CHUNK = 50_000


def download(destination: Path) -> None:
    """Download the archive, skipped if it exists, and check its SHA-256."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, destination)

    digest = hashlib.sha256()
    with open(destination, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    check(digest.hexdigest() == SHA256, f"{destination} is not the file of MatterGen")


def property_values(rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    """The named properties of the rows of a split, NaN when the cell is empty."""

    def value(name: str, cell: str) -> float:
        if cell == "":
            return float("nan")
        if name == "space_group":
            return float(get_spacegroup(cell).number)
        return float(cell)

    return {
        name: torch.tensor(
            [value(name, row[name]) for row in rows], dtype=torch.float64
        )
        for name in PROPERTIES
    }


def attach_properties(
    labelled: Path, destination: Path, properties: dict[str, torch.Tensor]
) -> None:
    """Write the structures of a file again, with their named properties."""
    dataset = HDF5Dataset(labelled)

    def chunks():
        for start in range(0, len(dataset), CHUNK):
            structures = dataset[start : start + CHUNK]
            stop = start + len(structures.num_atoms)
            yield replace(
                structures,
                properties={
                    name: values[start:stop].float()
                    for name, values in properties.items()
                },
            )

    HDF5Dataset.from_structures(destination, chunks())


def verify_properties(
    hdf5_file: Path, rows: list[dict[str, str]], expected: dict[str, torch.Tensor]
) -> int:
    """The named properties and the numbers of atoms are the ones of the source.

    Returns:
        Number of rows whose column num_sites is not the number of atoms of
        their CIF: it is larger for some structures of the Materials Project.
    """
    structures = HDF5Dataset(hdf5_file)[:]
    where = str(hdf5_file)

    in_cif = torch.tensor([count_atoms(row["cif"]) for row in rows])
    check(torch.equal(structures.num_atoms, in_cif), f"{where}: atoms of the CIF")
    num_sites = torch.tensor([int(row["num_sites"]) for row in rows])

    for name, values in expected.items():
        check(
            torch.allclose(
                structures.properties[name].double(),
                values,
                rtol=1e-6,
                atol=5e-5,
                equal_nan=True,
            ),
            f"{where}: {name} differs from its column",
        )

    return int((num_sites != in_cif).sum())


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
    raw = root / "raw" / SOURCE.name
    archive = raw / "alex_mp_20.zip"
    download(archive)

    total, summary = 0, []
    for split in SPLITS:
        csv_file = raw / "alex_mp_20" / f"{split}.csv"
        cif_file = root / "cif" / SOURCE.name / f"{split}.cif"
        labelled = root / "cif" / SOURCE.name / f"{split}_labels.hdf5"
        hdf5_file = root / SOURCE.name / f"{split}.hdf5"

        if not csv_file.exists():
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extract(f"alex_mp_20/{split}.csv", raw)
        for file_name in (cif_file, hdf5_file):
            file_name.parent.mkdir(parents=True, exist_ok=True)

        convert(
            csv_file,
            cif_file,
            hull_col=SOURCE.hull_col,
            band_gap_col=SOURCE.band_gap_col,
        )
        preprocessing(
            Preprocessing(source=cif_file, destination=labelled, batch_size=256)
        )

        with open(csv_file, "r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        properties = property_values(rows)
        attach_properties(labelled, hdf5_file, properties)

        count = verify(SOURCE, csv_file, hdf5_file, args.num_samples)
        stale = verify_properties(hdf5_file, rows, properties)
        total += count

        num_atoms = HDF5Dataset(hdf5_file)[:].num_atoms
        summary.append(
            f"{SOURCE.name} {split:5s} {count:7d} structures, "
            f"{int(num_atoms.min())}-{int(num_atoms.max())} atoms, "
            f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}, "
            f"{stale} rows whose num_sites is not the CIF"
        )
        labelled.unlink()

    check(total == SOURCE.num_structures, f"{total} != {SOURCE.num_structures}")
    print("\n".join(summary))
    print("all splits verified against their source")


if __name__ == "__main__":
    main()
