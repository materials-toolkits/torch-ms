"""Read the CIF files of a database of MOFs and check what was read.

Shared by the scripts building the datasets of experimental MOFs. The atoms of
a file which lists every atom of its cell (space group P1) are read as they
are listed, in their order, by a few lines of code. The other files are read
by pymatgen, which applies their symmetry operations. Every structure is then
compared with the reading of pymatgen.
"""

import re
import shlex
import warnings
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm
from pymatgen.core import Lattice
from pymatgen.io.cif import CifParser
from scipy.spatial import cKDTree

from _cif_writer import cell_parameters
from _structures_writer import Entry, check
from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number

CELL_TAGS = (
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
)
SYMBOL = re.compile(r"[A-Z][a-z]?")


def number(text: str) -> float:
    """Read a number of a CIF file, without its uncertainty: 1.234(5) is 1.234."""
    return float(text.split("(")[0])


def listed_atoms(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, bool, bool]:
    """Atoms listed in a CIF file, without applying its symmetry operations.

    Returns:
        Lengths and angles of the cell, shape (6,), symbol of each atom
        listed, its fractional coordinates, shape (n, 3), whether every atom
        is fully occupied, and whether the file lists every atom of the cell
        (space group P1).
    """
    lines = path.read_text(errors="replace").splitlines()

    cell = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in CELL_TAGS:
            cell[fields[0]] = number(fields[1])

    # the loop of the atoms: its tags, then one line per atom
    start = next(i for i, line in enumerate(lines) if "_atom_site_fract_x" in line)
    while lines[start - 1].strip().startswith("_atom_site_"):
        start -= 1

    tags = []
    while lines[start].strip().startswith("_atom_site_"):
        tags.append(lines[start].strip())
        start += 1

    columns = [tags.index(f"_atom_site_fract_{axis}") for axis in "xyz"]
    if "_atom_site_type_symbol" in tags:
        symbol_column = tags.index("_atom_site_type_symbol")
    else:
        symbol_column = tags.index("_atom_site_label")
    occupancy_column = None
    if "_atom_site_occupancy" in tags:
        occupancy_column = tags.index("_atom_site_occupancy")

    symbols, coords, occupied = [], [], True
    for line in lines[start:]:
        fields = shlex.split(line)
        if len(fields) != len(tags):
            break
        match = SYMBOL.match(fields[symbol_column])
        assert match is not None
        symbols.append(match.group(0))
        coords.append([number(fields[column]) for column in columns])
        if occupancy_column is not None:
            occupied &= number(fields[occupancy_column]) == 1.0

    operations = sum(
        bool(re.search(r"[xX]\s*,.*[yY]\s*,.*[zZ]", line)) for line in lines
    )
    spacegroup = [
        line.split()[-1]
        for line in lines
        if re.match(
            r"\s*_(symmetry_int_tables_number|space_group_it_number)\s", line, re.I
        )
    ]
    p1 = operations <= 1 and spacegroup in ([], ["1"])
    cell_array = np.array([cell[tag] for tag in CELL_TAGS])

    return cell_array, symbols, np.array(coords), occupied, p1


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | str:
    """Read the structure of a CIF file with pymatgen.

    Returns:
        Atomic numbers, fractional coordinates and lattice of the structure,
        its atoms in the order of pymatgen. Or the reason why it has none:
        "disordered" for a site which is not fully occupied by one element,
        "unreadable: ..." for a file pymatgen cannot read.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            structure = CifParser(path).parse_structures(primitive=False)[0]
        except Exception as error:  # pylint: disable=broad-exception-caught
            return f"unreadable: {type(error).__name__}: {str(error)[:80]}"

    if not structure.is_ordered:
        return "disordered"

    numbers = np.array(structure.atomic_numbers)

    return numbers, structure.frac_coords, structure.lattice.matrix


def read_entry(path: Path) -> Entry | str:
    """Read the structure of a CIF file.

    Returns:
        The entry of the structure, or the reason why it is skipped, see
        read_reference.
    """
    try:
        cell, symbols, coords, occupied, p1 = listed_atoms(path)
    except Exception:  # pylint: disable=broad-exception-caught
        # not written as the files of these databases: left to pymatgen
        p1 = False

    if p1:
        if not occupied:
            return "disordered"

        return Entry(
            numbers=np.array([symbol_to_number(symbol) for symbol in symbols]),
            coords=coords,
            lattice=Lattice.from_parameters(*cell).matrix,
        )

    reference = read_reference(path)
    if isinstance(reference, str):
        return reference

    return Entry(numbers=reference[0], coords=reference[1], lattice=reference[2])


def in_order(function, paths: list[Path], workers: int, description: str) -> Iterator:
    """Result of a function on each file, in the order of the files."""
    with Pool(workers) as pool:
        yield from tqdm.tqdm(
            pool.imap(function, paths, chunksize=8),
            total=len(paths),
            desc=description,
            unit="struct",
        )


def verify_structure(
    structures: Structures,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | str,
    where: str,
) -> None:
    """Compare a structure of a dataset with the reading of its file by pymatgen.

    The atoms are not in the same order: each atom of one must be at the
    position of an atom of the other, of the same element, and only one.
    """
    check(not isinstance(reference, str), f"{where}: {reference}")
    assert not isinstance(reference, str)
    numbers, coords, matrix = reference

    lengths, angles = cell_parameters(structures.lattice[0].double().numpy())
    expected_lengths, expected_angles = cell_parameters(matrix)
    check(
        bool(np.allclose(lengths, expected_lengths, rtol=1e-5, atol=1e-4)),
        f"{where}: lengths of the cell",
    )
    check(
        bool(np.allclose(angles, expected_angles, atol=1e-3)),
        f"{where}: angles of the cell",
    )

    z = structures.z.numpy()
    check(len(z) == len(numbers), f"{where}: {len(z)} atoms, {len(numbers)} expected")
    check(sorted(z.tolist()) == sorted(numbers.tolist()), f"{where}: elements")

    # nearest atom of the same element, in the periodic cell. pymatgen moves
    # a coordinate closer than 1e-4 to a fraction such as 1/3 onto it.
    stored = structures.x.double().numpy() % 1.0
    expected = coords % 1.0
    for element in np.unique(z):
        tree = cKDTree(stored[z == element] % 1.0, boxsize=1.0)
        distance, index = tree.query(expected[numbers == element] % 1.0)
        check(distance.max() < 2e-4, f"{where}: an atom is {distance.max():.2e} away")
        check(len(np.unique(index)) == len(index), f"{where}: an atom is read twice")
