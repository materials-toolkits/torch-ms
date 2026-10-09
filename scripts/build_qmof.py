"""Build the QMOF dataset in HDF5 format.

The QMOF Database (Rosen et al., 2021 and 2022), also shown by the MOF
Explorer of the Materials Project, is downloaded, its structures relaxed with
PBE-D3(BJ) are written without any filter and every structure is checked
against its source. Run inside the project environment:

    uv run python scripts/build_qmof.py datasets
"""

import argparse
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from pymatgen.core import Structure

from _structures_writer import Entry, check, download, save, summary
from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset

# https://doi.org/10.6084/m9.figshare.13147324.v18, CC BY 4.0
URL = "https://ndownloader.figshare.com/files/59573735"
MD5 = "0d89aaf66f2c306e86e47fd91cb1346e"
MEMBERS = ("qmof_database/qmof.json", "qmof_database/qmof_structure_data.json")

# band gap at the other levels of theory, computed for some of the structures
THEORIES = ("hle17", "hse06_10hf", "hse06")
PROPERTIES = (
    "pld",
    "lcd",
    "density",
    "energy_vdw",
    "net_magmom",
    *(f"band_gap_{theory}" for theory in THEORIES),
    "synthesized",
)


def iter_json_list(file_name: Path, chunk_size: int = 1 << 24) -> Iterator[Any]:
    """Elements of a JSON list, read one after the other.

    The file of the structures is 3 GB: it is not read whole.
    """
    decoder = json.JSONDecoder()

    with open(file_name, encoding="utf-8") as file:
        buffer = file.read(chunk_size).lstrip()
        check(buffer.startswith("["), f"{file_name} is not a list")
        buffer = buffer[1:]

        while True:
            buffer = buffer.lstrip().removeprefix(",").lstrip()

            if buffer.startswith("]"):
                return

            try:
                element, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = file.read(chunk_size)
                check(len(chunk) > 0, f"{file_name} ends inside an element")
                buffer += chunk
                continue

            # a number at the end of the buffer may be cut: an element is
            # complete when something follows it
            if end == len(buffer):
                chunk = file.read(chunk_size)
                if len(chunk) > 0:
                    buffer += chunk
                    continue

            yield element
            buffer = buffer[end:]


def read_properties(file_name: Path) -> dict[str, dict[str, Any]]:
    """Tabulated properties of each material, by identifier."""
    with open(file_name, encoding="utf-8") as file:
        return {row["qmof_id"]: row for row in json.load(file)}


def site_symbols(structure: dict[str, Any]) -> list[str] | None:
    """Element of each site, None if a site is not fully occupied by one element."""
    symbols = []

    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1 or species[0]["occu"] != 1:
            return None
        symbols.append(species[0]["element"])

    return symbols


def labels(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Labels and named properties of a material."""
    info, outputs = row["info"], row["outputs"]

    properties = {
        "pld": info["pld"],
        "lcd": info["lcd"],
        "density": info["density"],
        "energy_vdw": outputs["pbe"]["energy_vdw"],
        "net_magmom": outputs["pbe"]["net_magmom"],
        "synthesized": float(info["synthesized"]),
    }
    for theory in THEORIES:
        if theory in outputs:
            properties[f"band_gap_{theory}"] = outputs[theory]["bandgap"]

    return (
        {
            "total_energy": outputs["pbe"]["energy_total"],
            "band_gap": outputs["pbe"]["bandgap"],
        },
        properties,
    )


def entries(structures_file: Path, table: dict[str, dict[str, Any]]):
    """Entries of the structures, the row of the index of each one, and counts."""
    rows: list[dict[str, Any]] = []
    counts = {"disordered": 0}

    def iterate() -> Iterator[Entry]:
        for item in tqdm.tqdm(
            iter_json_list(structures_file), "reading", unit="struct"
        ):
            structure = item["structure"]
            symbols = site_symbols(structure)

            if symbols is None:
                counts["disordered"] += 1
                continue

            row = table[item["qmof_id"]]
            values, properties = labels(row)
            info = row["info"]
            rows.append(
                {
                    "qmof_id": item["qmof_id"],
                    "name": item["name"],
                    "formula": info["formula"],
                    "source": info["source"],
                    "doi": info["doi"],
                    "mofid": info["mofid"]["mofid"],
                }
            )

            yield Entry(
                numbers=np.array([symbol_to_number(symbol) for symbol in symbols]),
                coords=np.array([site["abc"] for site in structure["sites"]]),
                lattice=np.array(structure["lattice"]["matrix"]),
                labels=values,
                properties=properties,
            )

    return iterate(), rows, counts


def same(actual: float, expected: float | None) -> bool:
    """Whether a value of the dataset is the one of its source, NaN if missing."""
    if expected is None:
        return bool(np.isnan(actual))

    return abs(actual - expected) <= 1e-6 * max(1.0, abs(expected))


def verify_structure(
    structures: Structures, item: dict[str, Any], row: dict[str, Any]
) -> None:
    """Compare a structure of the dataset with its source, read by pymatgen."""
    where = item["qmof_id"]
    reference = Structure.from_dict(item["structure"])

    check(int(structures.num_atoms) == len(reference), f"{where}: number of atoms")
    check(structures.z.tolist() == list(reference.atomic_numbers), f"{where}: elements")
    check(bool(structures.pbc.all()), f"{where}: periodic")

    diff = np.abs(structures.x.double().numpy() - reference.frac_coords % 1.0)
    diff = np.minimum(diff, 1.0 - diff)
    check(diff.max() < 1e-5, f"{where}: coordinates differ by {diff.max():.2e}")

    # the lattice is rebuilt in the standard orientation: compare the metric
    matrix = reference.lattice.matrix
    gram = matrix @ matrix.T
    stored = structures.lattice[0].double().numpy()
    error = np.abs(stored @ stored.T - gram).max()
    check(error < 1e-4 * gram.max(), f"{where}: lattice")

    outputs, info = row["outputs"], row["info"]
    check(
        same(float(structures.total_energy), outputs["pbe"]["energy_total"]),
        f"{where}: total energy",
    )
    check(
        same(float(structures.band_gap), outputs["pbe"]["bandgap"]),
        f"{where}: band gap",
    )
    check(same(float(structures.properties["lcd"]), info["lcd"]), f"{where}: LCD")
    for theory in THEORIES:
        expected = outputs[theory]["bandgap"] if theory in outputs else None
        check(
            same(float(structures.properties[f"band_gap_{theory}"]), expected),
            f"{where}: band gap {theory}",
        )


def verify(
    structures_file: Path, hdf5_file: Path, table: dict[str, dict[str, Any]]
) -> int:
    """Compare every structure of the dataset with the database.

    Returns:
        Number of structures verified.
    """
    dataset = HDF5Dataset(hdf5_file)
    count = 0

    for item in tqdm.tqdm(iter_json_list(structures_file), "verifying", unit="struct"):
        if site_symbols(item["structure"]) is None:
            continue

        check(count < len(dataset), "dataset is shorter than its source")
        verify_structure(dataset[count], item, table[item["qmof_id"]])
        count += 1

    check(count == len(dataset), f"{len(dataset)} structures, {count} in the source")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    args = parser.parse_args()

    root = Path(args.destination)
    raw = root / "raw" / "qmof"
    hdf5_file = root / "qmof" / "all.hdf5"

    archive = raw / "qmof_database.zip"
    download(URL, archive, MD5)
    with zipfile.ZipFile(archive) as zip_file:
        for member in MEMBERS:
            if not (raw / member).exists():
                zip_file.extract(member, raw)

    table = read_properties(raw / MEMBERS[0])
    structures_file = raw / MEMBERS[1]

    iterator, rows, counts = entries(structures_file, table)
    save(hdf5_file, iterator, PROPERTIES, batch_size=256)
    pd.DataFrame(rows).to_parquet(hdf5_file.with_name("all_index.parquet"), index=False)

    verified = verify(structures_file, hdf5_file, table)
    check(verified + counts["disordered"] == len(table), "materials of the database")

    print(summary(hdf5_file))
    print(f'skipped: {counts["disordered"]} disordered')
    print("every structure verified against its source")


if __name__ == "__main__":
    main()
