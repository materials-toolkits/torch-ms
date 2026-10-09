"""Build the QM9 dataset in HDF5 format.

The 133,885 molecules of Ramakrishnan et al. (2014) are downloaded, written
without any filter (the 3,054 molecules which failed the consistency check of
the source are kept, and flagged) and every molecule is checked against its
source. Run inside the project environment:

    uv run python scripts/build_qm9.py datasets
"""

import argparse
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from _structures_writer import Entry, check, download, save, summary
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset

# https://doi.org/10.6084/m9.figshare.c.978904.v5, CC0
FILES = {
    "dsgdb9nsd.xyz.tar.bz2": ("3195389", "ad1ebd51ee7f5b3a6e32e974e5d54012"),
    "uncharacterized.txt": ("3195404", "a361887bacb427b8a0ce7903d92a53b4"),
    "atomref.txt": ("3195395", "2d30b2df8329d8fd805c0a4d158a0a0f"),
}
URL = "https://ndownloader.figshare.com/files/{file_id}"

NUM_MOLECULES = 133885
NUM_UNCHARACTERIZED = 3054

# CODATA 2018
HARTREE = 27.211386245988

# columns of the second line of a file after the index of the molecule, with
# the factor to the unit of the dataset: energies in eV, the rest as the source
COLUMNS = (
    ("rotational_a", 1.0),  # GHz
    ("rotational_b", 1.0),
    ("rotational_c", 1.0),
    ("dipole_moment", 1.0),  # Debye
    ("polarizability", 1.0),  # bohr^3
    ("homo", HARTREE),
    ("lumo", HARTREE),
    ("gap", HARTREE),
    ("r2", 1.0),  # bohr^2
    ("zpve", HARTREE),
    ("u0", HARTREE),
    ("u", HARTREE),
    ("h", HARTREE),
    ("g", HARTREE),
    ("cv", 1.0),  # cal/(mol K)
)
# energies with a reference per atom in atomref.txt, and their column there
ATOMIZATION = {"u0": 2, "u": 3, "h": 4, "g": 5}

PROPERTIES = (
    *(name for name, _ in COLUMNS if name not in ("gap", "u0")),
    *(f"{name}_atomization" for name in ATOMIZATION),
    "uncharacterized",
)


def number(text: str) -> float:
    """Read a number of the source, which writes 1e-5 as 1.*^-5."""
    return float(text.replace("*^", "e"))


def read_atomref(file_name: Path) -> dict[str, np.ndarray]:
    """Reference energies of the atoms in Hartree, by element: a row of the table."""
    reference = {}

    for line in file_name.read_text().splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0].isalpha() and len(fields[0]) <= 2:
            reference[fields[0]] = np.array([float(value) for value in fields[1:]])

    check(sorted(reference) == ["C", "F", "H", "N", "O"], "elements of atomref.txt")

    return reference


def read_uncharacterized(file_name: Path) -> set[int]:
    """Indices of the molecules which failed the consistency check of the source."""
    indices = set()

    for line in file_name.read_text().splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].isdigit():
            indices.add(int(fields[0]))

    check(len(indices) == NUM_UNCHARACTERIZED, "molecules of uncharacterized.txt")

    return indices


def iter_files(archive: Path) -> Iterator[tuple[int, str]]:
    """Index and content of the file of each molecule, in the order of the indices."""
    with tarfile.open(archive, "r:bz2") as tar:
        members = sorted(
            (member for member in tar if member.isfile()), key=lambda m: m.name
        )

        for member in tqdm.tqdm(members, desc="reading", unit="mol"):
            index = int(re.fullmatch(r"dsgdb9nsd_(\d{6})\.xyz", member.name).group(1))
            file = tar.extractfile(member)
            assert file is not None
            yield index, file.read().decode()


def parse(text: str) -> tuple[list[str], np.ndarray, dict[str, float], list[str]]:
    """Read the file of a molecule.

    Returns:
        Symbols, positions in Å, columns of the second line in the units of the
        source, and the SMILES and InChI lines.
    """
    lines = text.split("\n")
    n_atoms = int(lines[0])

    header = lines[1].split()
    values = {name: number(v) for (name, _), v in zip(COLUMNS, header[2:])}
    values["index"] = int(header[1])

    atoms = [line.split() for line in lines[2 : 2 + n_atoms]]
    symbols = [atom[0] for atom in atoms]
    positions = np.array([[number(v) for v in atom[1:4]] for atom in atoms])

    return symbols, positions, values, lines[3 + n_atoms : 5 + n_atoms]


def entries(archive: Path, reference: dict[str, np.ndarray], flagged: set[int]):
    """Entries of the molecules, and for each one its row of the index."""
    rows = []

    def iterate() -> Iterator[Entry]:
        for index, text in iter_files(archive):
            symbols, positions, values, (smiles, inchi) = parse(text)
            check(values["index"] == index, f"index of molecule {index}")

            properties = {
                name: factor * values[name]
                for name, factor in COLUMNS
                if name not in ("gap", "u0")
            }
            # relative to the atoms, in 64 bits: a total energy of 10^4 eV is
            # only known to 1 meV in 32 bits
            atoms = sum(reference[symbol] for symbol in symbols)
            for name, column in ATOMIZATION.items():
                properties[f"{name}_atomization"] = HARTREE * (
                    values[name] - atoms[column - 1]
                )
            properties["uncharacterized"] = float(index in flagged)

            rows.append(
                {
                    "index": index,
                    "smiles_gdb9": smiles.split()[0],
                    "smiles_relaxed": smiles.split()[1],
                    "inchi_gdb9": inchi.split()[0],
                    "inchi_relaxed": inchi.split()[1],
                }
            )

            yield Entry(
                numbers=np.array([symbol_to_number(s) for s in symbols]),
                coords=positions,
                labels={
                    "total_energy": HARTREE * values["u0"],
                    "band_gap": HARTREE * values["gap"],
                },
                properties=properties,
            )

    return iterate(), rows


# a number of the source, with its exponent if it has one
NUMBER = r"-?\d+\.?\d*(?:\*\^-?\d+|[eE]-?\d+)?"
ATOM = re.compile(
    rf"^([A-Z][a-z]?)\t\s*({NUMBER})\t\s*({NUMBER})\t\s*({NUMBER})\t", re.M
)


def verify(archive: Path, hdf5_file: Path, reference: dict[str, np.ndarray]) -> None:
    """Compare every molecule of the dataset with its file, read another way."""
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == NUM_MOLECULES, f"{len(dataset)} molecules in the dataset")

    structures = dataset[:]
    check(not bool(structures.pbc.any()), "molecules are not periodic")
    check(bool((structures.lattice == np.eye(3, dtype=np.float32)).all()), "lattice")

    positions = structures.positions().double().numpy()
    z = structures.z.numpy()
    ptr = structures.ptr.numpy()
    u0 = structures.total_energy.double().numpy()
    gap = structures.band_gap.double().numpy()
    atomization = structures.properties["u0_atomization"].double().numpy()
    homo = structures.properties["homo"].double().numpy()

    for row, (index, text) in enumerate(iter_files(archive)):
        where = f"molecule {index}"
        check(row == index - 1, f"{where}: row {row} of the dataset")

        atoms = ATOM.findall(text)
        header = text.split("\n")[1].split("\t")
        check(len(atoms) == int(text.split("\n")[0]), f"{where}: atoms of the file")
        check(ptr[row + 1] - ptr[row] == len(atoms), f"{where}: number of atoms")

        expected = np.array([[number(v) for v in atom[1:]] for atom in atoms])
        diff = np.abs(positions[ptr[row] : ptr[row + 1]] - expected).max()
        check(diff < 2e-6, f"{where}: positions differ by {diff:.2e}")

        numbers = [symbol_to_number(atom[0]) for atom in atoms]
        check(z[ptr[row] : ptr[row + 1]].tolist() == numbers, f"{where}: elements")

        # fields separated by tabs: "gdb <index>", A, B, C, mu, alpha, HOMO,
        # LUMO, gap, r2, ZPVE, U0, ...
        energy = number(header[11])
        check(abs(u0[row] / HARTREE - energy) < 1e-7 * abs(energy), f"{where}: U0")
        check(abs(gap[row] / HARTREE - number(header[8])) < 1e-6, f"{where}: gap")
        check(abs(homo[row] / HARTREE - number(header[6])) < 1e-6, f"{where}: HOMO")

        atoms_energy = sum(reference[atom[0]][1] for atom in atoms)
        check(
            abs(atomization[row] / HARTREE - (energy - atoms_energy)) < 2e-6,
            f"{where}: atomization energy",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    args = parser.parse_args()

    root = Path(args.destination)
    raw = root / "raw" / "qm9"
    hdf5_file = root / "qm9" / "all.hdf5"

    for name, (file_id, md5) in FILES.items():
        download(URL.format(file_id=file_id), raw / name, md5)

    archive = raw / "dsgdb9nsd.xyz.tar.bz2"
    reference = read_atomref(raw / "atomref.txt")
    flagged = read_uncharacterized(raw / "uncharacterized.txt")

    iterator, rows = entries(archive, reference, flagged)
    save(hdf5_file, iterator, PROPERTIES)
    pd.DataFrame(rows).to_parquet(hdf5_file.with_name("all_index.parquet"), index=False)

    verify(archive, hdf5_file, reference)

    print(summary(hdf5_file))
    print(f"kept and flagged as uncharacterized: {len(flagged)}")
    print("every molecule verified against its source")


if __name__ == "__main__":
    main()
