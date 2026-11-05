"""Crystal structures from the POSCAR and CONTCAR files of VASP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset

from .batching import Structures
from .constants import symbol_to_number

# names of the structure files of VASP in a directory, searched recursively
VASP_PATTERNS = ("POSCAR*", "CONTCAR*", "*.vasp", "*.poscar")


def parse_poscar(text: str, species: Sequence[str] | None = None) -> Structures:
    """Read a crystal from the content of a POSCAR or CONTCAR file of VASP.

    The format of VASP 5 and 6: a comment, the scaling factor, the three
    lattice vectors, the elements, their numbers of atoms, an optional
    "Selective dynamics" line, "Direct" or "Cartesian", then one line per
    atom. What follows the positions (velocities, predictor-corrector) is not
    read, nor the flags of selective dynamics.

    The scaling factor multiplies the lattice and the Cartesian positions; a
    negative one is the volume of the cell in Å³; three of them scale the x,
    y and z components (VASP 6). The lattice is the one of the file, not
    reoriented, and fractional coordinates are wrapped in [0, 1).

    Args:
        text: Content of the file.
        species: Element of each group of atoms, for the format of VASP 4,
            which has no line of elements; else taken from the comment when it
            lists them, as ASE does. Ignored when the file has the line.

    Returns:
        Structures with one crystal.

    Raises:
        ValueError: If the file is not a POSCAR, or its elements are unknown.

    Examples:
        >>> crystal = parse_poscar('''NaCl
        ... 5.64
        ...   0.0 0.5 0.5
        ...   0.5 0.0 0.5
        ...   0.5 0.5 0.0
        ... Na Cl
        ... 1 1
        ... Direct
        ... 0.0 0.0 0.0
        ... 0.5 0.5 0.5
        ... ''')
        >>> crystal.z.tolist(), crystal.lattice[0, 0].tolist()
        ([11, 17], [0.0, 2.819999933242798, 2.819999933242798])
    """
    lines = [line.strip() for line in text.splitlines()]
    if len(lines) < 8:
        raise ValueError("a POSCAR has at least 8 lines")

    scaling = _numbers(lines[1], "scaling factor")
    lattice = torch.tensor(
        [_numbers(line, "lattice vector")[:3] for line in lines[2:5]],
        dtype=torch.float64,
    )
    if lattice.shape != (3, 3):
        raise ValueError("a lattice vector has not three components")

    # VASP 4 has no line of elements: the counts follow the lattice
    row = 5
    words = lines[row].split()
    if words and not _is_integer(words[0]):
        symbols = [_element(word) for word in words]
        row += 1
    else:
        symbols = None
    counts = [int(word) for word in lines[row].split() if _is_integer(word)]
    row += 1
    if not counts:
        raise ValueError("no numbers of atoms in the POSCAR")
    if symbols is None:
        symbols = _species_of_vasp4(species, lines[0], len(counts))
    if len(symbols) != len(counts):
        raise ValueError(f"{len(symbols)} elements for {len(counts)} groups of atoms")

    if lines[row][:1] in ("s", "S"):
        row += 1
    cartesian = lines[row][:1] in ("c", "C", "k", "K")
    row += 1

    num_atoms = sum(counts)
    if len(lines) < row + num_atoms:
        raise ValueError(f"{num_atoms} atoms announced, fewer positions in the POSCAR")
    positions = torch.tensor(
        [_numbers(line, "position")[:3] for line in lines[row : row + num_atoms]],
        dtype=torch.float64,
    )
    if positions.shape != (num_atoms, 3):
        raise ValueError("a position has not three coordinates")

    lattice, factor = _scaled(lattice, scaling)
    if cartesian:
        # pylint: disable-next=not-callable
        positions = torch.linalg.solve(lattice.T, (positions * factor).T).T
    x = torch.remainder(positions, 1.0)

    z = torch.tensor(
        [symbol_to_number(symbol) for symbol in symbols], dtype=torch.int64
    ).repeat_interleave(torch.tensor(counts))

    return Structures(lattice=lattice.unsqueeze(0).float(), x=x.float(), z=z)


def _scaled(
    lattice: torch.Tensor, scaling: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The lattice in Å and the factor of the Cartesian components, shape (3,)."""
    if len(scaling) >= 3:
        factor = torch.tensor(scaling[:3], dtype=torch.float64)
    elif scaling and scaling[0] < 0:
        volume = torch.linalg.det(lattice).abs()  # pylint: disable=not-callable
        factor = (-scaling[0] / volume) ** (1.0 / 3.0) * torch.ones(
            3, dtype=torch.float64
        )
    elif scaling:
        factor = scaling[0] * torch.ones(3, dtype=torch.float64)
    else:
        raise ValueError("no scaling factor in the POSCAR")

    return lattice * factor, factor


def _species_of_vasp4(
    species: Sequence[str] | None, comment: str, num_groups: int
) -> list[str]:
    """Elements of a POSCAR of VASP 4: given, or the words of its comment."""
    if species is not None:
        return [_element(symbol) for symbol in species]

    words = comment.split()[:num_groups]
    try:
        symbols = [_element(word) for word in words]
        for symbol in symbols:
            symbol_to_number(symbol)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "the POSCAR has no line of elements (VASP 4): give them as species"
        ) from error
    if len(symbols) != num_groups:
        raise ValueError("the POSCAR has no line of elements (VASP 4): give species")

    return symbols


def _element(word: str) -> str:
    """Element of a word of the line of elements: "Fe", "Fe_pv", "Fe/1a2b3c"."""
    match = re.match(r"[A-Z][a-z]?", word)
    if match is None:
        raise ValueError(f"{word!r} is not an element")

    return match.group(0)


def _numbers(line: str, what: str) -> list[float]:
    """The numbers at the start of a line, up to the first word which is not one."""
    numbers = []
    for word in line.split():
        try:
            numbers.append(float(word))
        except ValueError:
            break
    if not numbers:
        raise ValueError(f"no {what} in the line {line!r} of the POSCAR")

    return numbers


def _is_integer(word: str) -> bool:
    return re.fullmatch(r"\d+", word) is not None


class VaspDataset(Dataset):
    """Crystals of the POSCAR and CONTCAR files of a directory.

    Args:
        root_dir: Directory searched recursively, as a directory per
            calculation of VASP.
        transform: Optional transform of each crystal.
        patterns: Names of the files read, VASP_PATTERNS by default.
        species: Elements of the files of VASP 4, see parse_poscar.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Structures], Structures] | None = None,
        patterns: Sequence[str] = VASP_PATTERNS,
        species: Sequence[str] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.species = species
        self.files = sorted(
            {
                path
                for pattern in patterns
                for path in self.root_dir.rglob(pattern)
                if path.is_file()
            }
        )
        if not self.files:
            raise ValueError(f"No POSCAR or CONTCAR files found in {self.root_dir}")

    def __len__(self) -> int:
        """Return the number of files."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Structures:
        """The crystal of a file."""
        structures = parse_poscar(self.files[idx].read_text(), self.species)
        if self.transform is not None:
            structures = self.transform(structures)

        return structures
