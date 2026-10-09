"""Build the CoRE MOF 2024 dataset in HDF5 format.

The public part of the CoRE MOF database updated in 2024 (Zhao et al., 2025),
"CoRE MOF SI", is downloaded: its computation-ready structures with all solvent
removed (ASR), with free solvent removed (FSR) and with their ions (ION), and
its structures which are not computation-ready (NCR). They are written without
any filter and every structure is checked against its CIF file. The part of
the database derived from the CSD is distributed by the CCDC and is not in this
dataset. Run inside the project environment:

    uv run python scripts/build_core_mof_2024.py datasets
"""

import argparse
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from _cif_reader import in_order, read_entry, read_reference, verify_structure
from _structures_writer import Entry, check, download, save, summary
from torchms.data.hdf5 import HDF5Dataset

# https://doi.org/10.5281/zenodo.15055758, version 1.1, CC BY 4.0
URL = "https://zenodo.org/api/records/15055758/files/{name}/content"
ARCHIVE = ("CoREMOF2024DB_SI_20250204.zip", "240444c92c1868ee131ab7b059f45b05")

# computation-ready splits: directory of the archive, table, MD5 and structures
READY = {
    "asr": (
        "ASR",
        "ASR_data_SI_20250204.csv",
        "c02235ce5e3843257bb561911b07d0fa",
        1372,
    ),
    "fsr": (
        "FSR",
        "FSR_data_SI_20250204.csv",
        "5a7737add7b5e01f3cbf35aa8ca6fbb4",
        1192,
    ),
    "ion": ("Ion", "ION_data_SI_20250204.csv", "15a86925553b95d3ac81d0c8da1f07c7", 100),
}
NUM_NOT_READY = 5636

# properties of the dataset and their column in the tables of the source.
# Stabilities are predictions of the models of the source, "unknown" for some.
COLUMNS = {
    "lcd": "LCD (Å)",
    "pld": "PLD (Å)",
    "lfpd": "LFPD (Å)",
    "density": "Density (g/cm3)",
    "asa_m2_cm3": "ASA (m2/cm3)",
    "asa_m2_g": "ASA (m2/g)",
    "nasa_m2_cm3": "NASA (m2/cm3)",
    "nasa_m2_g": "NASA (m2/g)",
    "void_fraction": "VF",
    "pv_cm3_g": "PV (cm3/g)",
    "heat_capacity_300k": "Heat_capacity@300K (J/g/K)",
    "thermal_stability": "Thermal_stability (℃)",
    "solvent_stability": "Solvent_stability",
    "water_stability": "Water_stability",
}
PROPERTIES = (*COLUMNS, "has_oms", "unmodified")


def read_table(file_name: Path) -> pd.DataFrame:
    """Table of the properties of the structures of a split, by identifier."""
    table = pd.read_csv(file_name, encoding="utf-8-sig")
    check(table["coreid"].is_unique, f"identifiers of {file_name}")

    for column in COLUMNS.values():
        table[column] = pd.to_numeric(table[column], errors="coerce")

    return table.set_index("coreid")


def entries(paths: list[Path], table: pd.DataFrame | None, workers: int):
    """Entries of the structures, the row of the index of each one, and the skipped."""
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    def iterate() -> Iterator[Entry]:
        reader = in_order(read_entry, paths, workers, "reading")

        for path, entry in zip(paths, reader):
            if isinstance(entry, str):
                skipped.append({"name": path.stem, "reason": entry})
                continue

            if table is None:
                # not computation-ready: its directory says why
                rows.append({"name": path.stem, "not_ready": path.parent.name})
            else:
                source = table.loc[path.stem]
                entry.properties = {
                    **{name: float(source[column]) for name, column in COLUMNS.items()},
                    "has_oms": float(source["Has OMS"] == "Yes"),
                    "unmodified": float(bool(source["unmodified"])),
                }
                rows.append(
                    {
                        "coreid": path.stem,
                        "refcode": source["refcode"],
                        "metals": source["Metal Types"],
                        "topology": source["topology(AllNodes)"],
                        "doi": source["DOI"],
                    }
                )

            yield entry

    return iterate(), rows, skipped


def verify(
    paths: list[Path], hdf5_file: Path, table: pd.DataFrame | None, workers: int
) -> list[str]:
    """Compare every structure of the dataset with its CIF file and the table.

    Returns:
        Structures whose file has not the number of atoms of the table: the
        source disagrees with itself, their properties may be the ones of
        another structure.
    """
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == len(paths), f"{len(dataset)} structures in the dataset")

    reader = in_order(read_reference, paths, workers, "verifying")
    inconsistent = []

    for row, (path, reference) in enumerate(zip(paths, reader)):
        structures = dataset[row]
        verify_structure(structures, reference, path.name)

        if table is None:
            continue

        source = table.loc[path.stem]
        if int(structures.num_atoms) != int(source["natoms"]):
            inconsistent.append(path.stem)
        for name in ("lcd", "density", "void_fraction"):
            stored = float(structures.properties[name])
            expected = float(source[COLUMNS[name]])
            check(
                abs(stored - expected) <= 1e-6 * max(1.0, abs(expected)),
                f"{path.name}: {name}",
            )

    return inconsistent


def build(
    name: str,
    paths: list[Path],
    table: pd.DataFrame | None,
    root: Path,
    workers: int,
) -> None:
    """Build and verify the file of a split."""
    hdf5_file = root / "core_mof_2024" / f"{name}.hdf5"
    names = () if table is None else PROPERTIES

    iterator, rows, skipped = entries(paths, table, workers)
    save(hdf5_file, iterator, names, batch_size=256)
    pd.DataFrame(rows).to_parquet(
        hdf5_file.with_name(f"{name}_index.parquet"), index=False
    )
    pd.DataFrame(skipped, columns=["name", "reason"]).to_csv(
        hdf5_file.with_name(f"{name}_skipped.csv"), index=False
    )

    dropped = {row["name"] for row in skipped}
    kept = [path for path in paths if path.stem not in dropped]
    inconsistent = verify(kept, hdf5_file, table, workers)

    reasons = pd.Series([row["reason"].split(":")[0] for row in skipped], dtype=str)
    print(summary(hdf5_file))
    print(f"skipped: {reasons.value_counts().to_dict()}")
    print(f"number of atoms which is not the one of the table: {inconsistent}")
    print("every structure verified against its file read by pymatgen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1)
    )
    args = parser.parse_args()

    root = Path(args.destination)
    raw = root / "raw" / "core_mof_2024"

    download(URL.format(name=ARCHIVE[0]), raw / ARCHIVE[0], ARCHIVE[1])
    files = raw / "files"
    if not files.exists():
        partial = raw / "files.part"
        with zipfile.ZipFile(raw / ARCHIVE[0]) as archive:
            archive.extractall(partial)
        os.replace(partial, files)

    for name, (directory, table_name, md5, count) in READY.items():
        download(URL.format(name=table_name), raw / table_name, md5)
        table = read_table(raw / table_name)
        paths = sorted((files / "SI" / "CR" / directory).glob("*.cif"))

        check(len(paths) == count, f"{len(paths)} files in {name}")
        check(
            sorted(table.index) == sorted(path.stem for path in paths),
            f"the table of {name} has the structures of its directory",
        )
        build(name, paths, table, root, args.workers)

    paths = sorted((files / "SI" / "NCR").rglob("*.cif"), key=lambda path: path.stem)
    check(len(paths) == NUM_NOT_READY, f"{len(paths)} files in ncr")
    check(len({path.stem for path in paths}) == len(paths), "names of ncr")
    build("ncr", paths, None, root, args.workers)


if __name__ == "__main__":
    main()
