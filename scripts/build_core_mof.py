"""Build the CoRE MOF 2019 dataset in HDF5 format.

The public structures of the Computation-Ready Experimental MOF database
(Chung et al., 2019), with all solvent removed (ASR) and with free solvent
removed (FSR), are downloaded, written without any filter and every structure
is checked against its CIF file. Run inside the project environment:

    uv run python scripts/build_core_mof.py datasets
"""

import argparse
import os
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _cif_reader import in_order, read_entry, read_reference, verify_structure
from _structures_writer import Entry, check, download, save, summary
from torchms.data.hdf5 import HDF5Dataset

# https://doi.org/10.5281/zenodo.7691378, version 1.1.4, CC BY 4.0
URL = "https://zenodo.org/api/records/7691378/files/{name}/content"


@dataclass(frozen=True)
class Split:
    """Files of a split of the database, with their MD5 and its number of structures."""

    name: str
    archive: str
    archive_md5: str
    table: str
    table_md5: str
    num_structures: int


SPLITS = (
    Split(
        "asr",
        "2019-11-01-ASR-public_12020.tar.gz",
        "e9858701b87c9d89e3b13a403da4a9c5",
        "2019-11-01-ASR-public_12020.csv",
        "0f9345ad8b085ccc64b84495515cd5fe",
        12020,
    ),
    Split(
        "fsr",
        "2019-11-01-FSR-public_7061.tar.gz",
        "2a374f67a756899f1607b7bea590e409",
        "2019-11-01-FSR-public_7061.csv",
        "415eccc07d65dfd6e5da4089bd820b27",
        7061,
    ),
)

# properties of the dataset and their column in the table of the source. The
# column "cm3_g" of the source holds the density, in g/cm3.
COLUMNS = {
    "lcd": "LCD",
    "pld": "PLD",
    "lfpd": "LFPD",
    "density": "cm3_g",
    "asa_m2_cm3": "ASA_m2_cm3",
    "asa_m2_g": "ASA_m2_g",
    "nasa_m2_cm3": "NASA_m2_cm3",
    "nasa_m2_g": "NASA_m2_g",
    "void_fraction": "AV_VF",
    "av_cm3_g": "AV_cm3_g",
    "nav_cm3_g": "NAV_cm3_g",
}
PROPERTIES = (*COLUMNS, "has_oms", "disorder")


def extract(archive: Path, directory: Path) -> list[Path]:
    """Extract the CIF files of an archive, and list them by name of structure."""
    if not directory.exists():
        partial = directory.with_name(directory.name + ".part")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(partial, filter="data")
        os.replace(partial, directory)

    return sorted(directory.rglob("*.cif"), key=lambda path: path.stem)


def read_table(file_name: Path) -> pd.DataFrame:
    """Table of the properties of the structures, by name of structure."""
    table = pd.read_csv(file_name)
    table = table.loc[:, ~table.columns.str.startswith("Unnamed")]
    check(table["filename"].is_unique, f"names of {file_name}")

    return table.set_index("filename")


def entries(paths: list[Path], table: pd.DataFrame, workers: int):
    """Entries of the structures, the row of the index of each one, and the skipped."""
    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []

    def iterate() -> Iterator[Entry]:
        reader = in_order(read_entry, paths, workers, "reading")

        for path, entry in zip(paths, reader):
            if isinstance(entry, str):
                skipped.append((path.stem, entry))
                continue

            source = table.loc[path.stem]
            disorder = path.parent.name.startswith("disorder")
            entry.properties = {
                **{name: float(source[column]) for name, column in COLUMNS.items()},
                "has_oms": float(source["Has_OMS"] == "Yes"),
                "disorder": float(disorder),
            }
            rows.append(
                {
                    "filename": path.stem,
                    "all_metals": source["All_Metals"],
                    "from_csd": source["from_CSD"] == "Y",
                    "doi": source["DOI_public"],
                }
            )

            yield entry

    return iterate(), rows, skipped


def verify(
    paths: list[Path], hdf5_file: Path, table: pd.DataFrame, workers: int
) -> None:
    """Compare every structure of the dataset with its CIF file and the table."""
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == len(paths), f"{len(dataset)} structures in the dataset")

    reader = in_order(read_reference, paths, workers, "verifying")

    for row, (path, reference) in enumerate(zip(paths, reader)):
        structures = dataset[row]
        verify_structure(structures, reference, path.name)

        source = table.loc[path.stem]
        for name in ("lcd", "density", "void_fraction"):
            stored = float(structures.properties[name])
            expected = float(source[COLUMNS[name]])
            check(
                abs(stored - expected) <= 1e-6 * max(1.0, abs(expected)),
                f"{path.name}: {name}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1)
    )
    args = parser.parse_args()

    root = Path(args.destination)
    raw = root / "raw" / "core_mof"

    for split in SPLITS:
        hdf5_file = root / "core_mof" / f"{split.name}.hdf5"

        download(URL.format(name=split.archive), raw / split.archive, split.archive_md5)
        download(URL.format(name=split.table), raw / split.table, split.table_md5)

        paths = extract(raw / split.archive, raw / split.name)
        table = read_table(raw / split.table)
        check(len(paths) == split.num_structures, f"{len(paths)} files in {split.name}")
        check(
            sorted(table.index) == sorted(path.stem for path in paths),
            f"the table of {split.name} has the structures of its archive",
        )

        iterator, rows, skipped = entries(paths, table, args.workers)
        save(hdf5_file, iterator, PROPERTIES, batch_size=256)
        pd.DataFrame(rows).to_parquet(
            hdf5_file.with_name(f"{split.name}_index.parquet"), index=False
        )

        names = {name for name, _ in skipped}
        kept = [path for path in paths if path.stem not in names]
        verify(kept, hdf5_file, table, args.workers)

        disorder = HDF5Dataset(hdf5_file).dataset("properties/disorder")[...]
        print(summary(hdf5_file))
        print(
            f"skipped: {len(skipped)} {skipped[:10]}; flagged as disordered by the "
            f"source: {int(disorder.sum())}"
        )
        print("every structure verified against its file read by pymatgen")


if __name__ == "__main__":
    main()
