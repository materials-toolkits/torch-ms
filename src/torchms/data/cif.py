"""CIF dataset implementation for loading crystal structures from CIF files."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
from torch.utils.data import Dataset

from .batching import Structures
from .constants import symbol_to_number
from .constants import get_symmetry_operations
from .symmetry import MERGE_TOLERANCE, expand_symmetry

logger = logging.getLogger(__name__)


class NoStructure(ValueError):
    """A data block which describes no structure: no atom site of known place."""


# a token of CIF: a text field between lines starting with ";", a quoted value
# (the quote ends before a blank), a comment, or a bare value
_TOKEN = re.compile(
    r"^;(?P<text>.*?)^;"
    r"|'(?P<single>.*?)'(?=\s|$)"
    r'|"(?P<double>.*?)"(?=\s|$)'
    r"|(?P<comment>#[^\n]*)"
    r"|(?P<bare>\S+)",
    re.MULTILINE | re.DOTALL,
)

# columns of the symmetry operations, "-x+1/2, y, z", in CIF 2 then CIF 1
_SYMMETRY_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")


@dataclass(frozen=True)
class CifBlock:
    """Tags and loops of a CIF data block, tags in lower case.

    Attributes:
        items: Value of each tag outside of a loop.
        loops: Columns of each loop, by tag.
    """

    items: dict[str, str]
    loops: list[dict[str, list[str]]]

    @classmethod
    def read(cls, block_string: str) -> CifBlock:
        r"""Read the tags and the loops of a data block.

        A loop ends at the next loop_, tag or data block.

        Examples:
            >>> block = CifBlock.read(
            ...     "data_x\n_cell_length_a 3.0\nloop_\n_a\n_B\n1 'p q'\n2 r\n"
            ...     "loop_\n_c\n;\ntext\n;\n"
            ... )
            >>> block.items
            {'_cell_length_a': '3.0'}
            >>> block.loops
            [{'_a': ['1', '2'], '_b': ['p q', 'r']}, {'_c': ['\ntext\n']}]
        """
        tokens = [
            (match.lastgroup, match.group(match.lastgroup))
            for match in _TOKEN.finditer(block_string)
            if match.lastgroup is not None and match.lastgroup != "comment"
        ]

        items: dict[str, str] = {}
        loops: list[dict[str, list[str]]] = []
        i = 0
        while i < len(tokens):
            kind, value = tokens[i]
            keyword = kind == "bare" and value.lower()
            if keyword and keyword.startswith("loop_"):
                headers = []
                i += 1
                while (
                    i < len(tokens)
                    and tokens[i][0] == "bare"
                    and tokens[i][1][0] == "_"
                ):
                    headers.append(tokens[i][1].lower())
                    i += 1
                values = []
                while i < len(tokens) and not _ends_loop(*tokens[i]):
                    values.append(tokens[i][1])
                    i += 1
                rows = len(values) // max(len(headers), 1)
                loops.append(
                    {
                        header: values[column : rows * len(headers) : len(headers)]
                        for column, header in enumerate(headers)
                    }
                )
            elif keyword and keyword.startswith("_") and i + 1 < len(tokens):
                items[keyword] = tokens[i + 1][1]
                i += 2
            else:
                i += 1

        return cls(items, loops)

    def column(self, tag: str) -> list[str] | None:
        """Values of a tag: its column in a loop, or its single value."""
        for loop in self.loops:
            if tag in loop:
                return loop[tag]
        if tag in self.items:
            return [self.items[tag]]

        return None

    def table(self, tag: str) -> dict[str, list[str]] | None:
        """The loop which has a column of this tag."""
        return next((loop for loop in self.loops if tag in loop), None)


def _ends_loop(kind: str | None, value: str) -> bool:
    """Whether a token ends the values of a loop: a tag or a keyword."""
    if kind != "bare":
        return False
    lower = value.lower()

    return lower.startswith(("_", "loop_", "data_", "global_", "save_", "stop_"))


def parse_symmetry_operation(operation: str) -> torch.Tensor:
    """Affine matrix of a symmetry operation of CIF on fractional coordinates.

    Args:
        operation: The images of x, y and z separated by commas, as in
            _space_group_symop_operation_xyz: "-x+1/2, y, z-1/4", "x-y,x,z+1/2".

    Returns:
        Matrix [R|t; 0 0 0 1], shape (4, 4), float64.

    Raises:
        ValueError: If the operation is not three affine expressions.

    Examples:
        >>> parse_symmetry_operation("-x+1/2, x-y, 0.25+z")[:3].tolist()
        [[-1.0, 0.0, 0.0, 0.5], [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.25]]
    """
    rows = operation.replace(" ", "").lower().split(",")
    if len(rows) != 3:
        raise ValueError(f"symmetry operation {operation!r} has not three coordinates")

    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = 0.0
    for row, expression in enumerate(rows):
        terms = re.findall(r"([+-]?)([^+-]+)", expression)
        if not terms or "".join(sign + term for sign, term in terms) != expression:
            raise ValueError(f"cannot read symmetry operation {operation!r}")
        for sign, term in terms:
            factor = -1.0 if sign == "-" else 1.0
            axis = term[-1]
            if axis in "xyz":
                coefficient = term[:-1].rstrip("*")
                factor *= _fraction(coefficient) if coefficient else 1.0
                matrix[row, "xyz".index(axis)] += factor
            else:
                matrix[row, 3] += factor * _fraction(term)

    return matrix


def _fraction(value: str) -> float:
    """A number written as a fraction or a decimal: "1/2", "0.25"."""
    if "/" in value:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator)

    return float(value)


@dataclass
class ParseResult:
    """Result of parsing a single CIF data block.

    Attributes:
        structures: Structures object if parsing succeeded, None otherwise.
        error: Error message if parsing failed, None otherwise.
        block_name: Name of the data block (if available).
    """

    structures: Structures | None = None
    error: str | None = None
    block_name: str | None = None


class CifParser:
    """Parser for CIF crystallographic files.

    The atoms of a block are its atom sites and their images by the symmetry
    operations it lists (_space_group_symop_operation_xyz or
    _symmetry_equiv_pos_as_xyz), in its own setting. Without them, the
    operations of the standard setting of its space-group number are applied.
    Images closer than 1e-3 in fractional coordinates are one atom, as in ASE.
    Sites of several elements at one place (partial occupancies) are one atom
    of the most occupied element; occupancies are not kept otherwise.

    Args:
        regex_overrides: Optional dict mapping pattern names to custom regex strings.
            If provided, these override the default patterns. Invalid regex will
            raise re.error and should be handled by the caller.
            Available keys: space_group, cell_length, cell_angle, atom_row,
            data_block, total_energy, energy_above_hull, band_gap,
            formation_energy_per_atom.
    """

    def __init__(self, regex_overrides: dict[str, str] | None = None):
        regex_overrides = regex_overrides or {}

        self._space_group = re.compile(
            regex_overrides.get("space_group")
            or r"(?:_space_group_IT_number|_symmetry_Int_Tables_number)\s+(\d+)"
        )
        self._cell_length = re.compile(
            regex_overrides.get("cell_length")
            or r"_cell_length_([abc])\s+([\d.]+(?:\([\d]+\))?)"
        )
        self._cell_angle = re.compile(
            regex_overrides.get("cell_angle")
            or r"_cell_angle_(alpha|beta|gamma)\s+([\d.]+(?:\([\d]+\))?)"
        )
        self._atom_row = re.compile(
            regex_overrides.get("atom_row")
            or r"^(\w+)\s+([\d./-]+)\s+([\d./-]+)\s+([\d./-]+)$",
            re.MULTILINE,
        )
        self._data_block = re.compile(
            regex_overrides.get("data_block") or r"^data_(\S+)", re.MULTILINE
        )
        self._total_energy = re.compile(
            regex_overrides.get("total_energy")
            or r"(?:_dft_cell_energy|_dft_energy|_energy|_energy_total|_calculated_energy)\s+([\d.-]+)"
        )
        self._energy_above_hull = re.compile(
            regex_overrides.get("energy_above_hull")
            or r"(?:_energy_above_hull|_e_above_hull)\s+([\d.-]+)"
        )
        self._band_gap = re.compile(
            regex_overrides.get("band_gap")
            or r"(?:_dft_band_gap|_band_gap|_bandgap)\s+([\d.-]+)"
        )
        self._formation_energy_per_atom = re.compile(
            regex_overrides.get("formation_energy_per_atom")
            or r"_formation_energy_per_atom\s+([\d.-]+)"
        )

    def parse(self, cif_string: str) -> Structures:
        """Parse a CIF string into a Structures object.

        Args:
            cif_string: CIF file content as a string. May contain one or
                    multiple data blocks. Each data block is parsed as a
                    separate structure and batched together.

        Returns:
            Structures object containing all parsed structures batched together.

        Blocks which describe no structure, without atom sites or with
        unknown ones only, are left out.

        Raises:
            ValueError: If no block describes a structure, or required data is
                missing.
        """
        blocks = self._find_data_blocks(cif_string)
        if not blocks:
            raise ValueError("No data blocks found in CIF")

        structures = []
        for block in blocks:
            try:
                structures.append(self._parse_single_block(block))
            except NoStructure:
                continue
        if not structures:
            raise ValueError("No structure found in CIF")

        return Structures.collate(structures)

    def parse_with_errors(self, cif_string: str) -> list[ParseResult]:
        """Parse CIF string, yielding ParseResult for each block with error handling.

        This method parses each data block individually and returns both the
        structure (on success) or error message (on failure), allowing
        downstream code to handle failures gracefully.

        Args:
            cif_string: CIF file content as a string.

        Yields:
            ParseResult for each data block. Check .structures for successful
            parse, or .error for failure reason.
        """
        blocks = self._find_data_blocks(cif_string)
        if not blocks:
            return []

        result = []
        for block in blocks:
            match = re.search(r"^data_(\S+)", block, re.MULTILINE)
            block_name = match.group(1) if match else None

            try:
                structures = self._parse_single_block(block)
                parsed = ParseResult(structures=structures, block_name=block_name)
                result.append(parsed)
            except (ValueError, RuntimeError, NotImplementedError) as e:
                parsed = ParseResult(error=str(e), block_name=block_name)
                result.append(parsed)
                logger.warning("Failed to parse block %s: %s", block_name, e)

        return result

    def _find_data_blocks(self, cif_string: str) -> list[str]:
        """Find all data blocks in CIF string.

        Args:
            cif_string: CIF file content.

        Returns:
            List of CIF strings, each containing one data block.
        """
        matches = list(self._data_block.finditer(cif_string))
        if not matches:
            return [cif_string]

        blocks = []
        for i, match in enumerate(matches):
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(cif_string)
            blocks.append(cif_string[start:end])

        return blocks

    def _parse_single_block(self, block_string: str) -> Structures:
        """Parse a single CIF data block into a Structures object.

        Args:
            block_string: CIF data block content.

        Returns:
            Structures object with a single structure.
        """
        block = CifBlock.read(block_string)
        coords, atomic_numbers = self._parse_atoms(block)
        operations = self._parse_symmetry_operations(block, block_string)
        cell_params = self._parse_cell_parameters(block_string)
        lattice = self._build_lattice_vectors(cell_params)
        total_energy, energy_above_hull, band_gap, formation_energy_per_atom = (
            self._parse_properties(block_string)
        )

        x = torch.tensor(coords, dtype=torch.float64)
        z = torch.tensor(atomic_numbers, dtype=torch.int64)

        structures = expand_symmetry(lattice, x, z, operations)

        n_atoms = structures.z.shape[0]
        return Structures(
            lattice=structures.lattice.float(),
            x=structures.x.float(),
            z=structures.z.long(),
            num_atoms=torch.tensor([n_atoms], dtype=torch.int64),
            batch=torch.zeros(n_atoms, dtype=torch.int64),
            ptr=torch.tensor([0, n_atoms], dtype=torch.int64),
            total_energy=torch.tensor([total_energy], dtype=torch.float32),
            energy_above_hull=torch.tensor([energy_above_hull], dtype=torch.float32),
            band_gap=torch.tensor([band_gap], dtype=torch.float32),
            formation_energy_per_atom=torch.tensor(
                [formation_energy_per_atom], dtype=torch.float32
            ),
            _validate=True,
        )

    def _parse_symmetry_operations(
        self, block: CifBlock, cif_string: str
    ) -> torch.Tensor:
        """Symmetry operations of a block, shape (n_ops, 4, 4), float64.

        The ones it lists, else the ones of the standard setting of its
        space-group number.

        Raises:
            ValueError: If the block has neither.
        """
        for tag in _SYMMETRY_TAGS:
            listed = block.column(tag)
            if listed and all(op not in ("?", ".") for op in listed):
                return torch.stack([parse_symmetry_operation(op) for op in listed])

        space_group_match = self._space_group.search(cif_string)
        if space_group_match is None:
            raise ValueError("Space group number not found in CIF")

        return get_symmetry_operations(int(space_group_match.group(1))).double()

    def _parse_properties(self, cif_string: str) -> tuple[float, float, float, float]:
        """Parse energy and band gap properties from CIF."""
        total_energy = self._parse_property(self._total_energy, cif_string)
        energy_above_hull = self._parse_property(self._energy_above_hull, cif_string)
        band_gap = self._parse_property(self._band_gap, cif_string)
        formation_energy_per_atom = self._parse_property(
            self._formation_energy_per_atom, cif_string
        )
        return total_energy, energy_above_hull, band_gap, formation_energy_per_atom

    def _parse_property(self, pattern: re.Pattern, cif_string: str) -> float:
        """Parse a single property, returning NaN if not found."""
        match = pattern.search(cif_string)
        return float(match.group(1)) if match else float("nan")

    def _parse_cell_parameters(self, cif_string: str) -> dict[str, float]:
        """Parse cell length and angle parameters from CIF."""
        cell_lengths: dict[str, float] = {}
        for match in self._cell_length.finditer(cif_string):
            key = match.group(1)
            value = self._strip_uncertainty(match.group(2))
            cell_lengths[key] = float(value)

        if len(cell_lengths) != 3:
            raise ValueError(f"Expected 3 cell lengths, got {len(cell_lengths)}")

        cell_angles: dict[str, float] = {}
        for match in self._cell_angle.finditer(cif_string):
            key = match.group(1)
            value = self._strip_uncertainty(match.group(2))
            cell_angles[key] = float(value)

        if len(cell_angles) != 3:
            raise ValueError(f"Expected 3 cell angles, got {len(cell_angles)}")

        return {
            "a": cell_lengths["a"],
            "b": cell_lengths["b"],
            "c": cell_lengths["c"],
            "alpha": cell_angles["alpha"],
            "beta": cell_angles["beta"],
            "gamma": cell_angles["gamma"],
        }

    def _strip_uncertainty(self, value: str) -> str:
        """Strip uncertainty in parentheses, e.g. '110.12(5)' -> '110.12'."""
        return re.sub(r"\([\d]+\)$", "", value)

    def _parse_fractional(self, value: str) -> float:
        """Parse fractional value (int, float, fraction, or with uncertainty)."""
        value = re.sub(r"\([\d]+\)$", "", value)
        if "/" in value:
            num, denom = value.split("/")
            return float(num) / float(denom)
        return float(value)

    def _extract_element_symbol(self, symbol: str) -> str:
        """The first element symbol in a type symbol or a label, as ASE.

        "Fe2+" and "Fe1" are Fe, a mixed site "0.8Nb + 0.2Zr" is Nb, deuterium
        is hydrogen.
        """
        match = re.search(r"[A-Z][a-z]?", symbol)
        if match is None:
            return symbol
        element = match.group(0)

        return "H" if element == "D" else element

    def _build_lattice_vectors(self, cell: dict[str, float]) -> torch.Tensor:
        """Build lattice vectors from cell parameters (in degrees).

        Args:
            cell: Dictionary with keys a, b, c, alpha, beta, gamma.

        Returns:
            Lattice vectors with shape (3, 3).
        """
        a, b, c = cell["a"], cell["b"], cell["c"]
        alpha, beta, gamma = cell["alpha"], cell["beta"], cell["gamma"]
        alpha_rad = math.radians(alpha)
        beta_rad = math.radians(beta)
        gamma_rad = math.radians(gamma)

        cos_alpha = math.cos(alpha_rad)
        cos_beta = math.cos(beta_rad)
        cos_gamma = math.cos(gamma_rad)
        sin_gamma = math.sin(gamma_rad)

        v = math.sqrt(
            1.0
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2.0 * cos_alpha * cos_beta * cos_gamma
        )

        lattice = torch.tensor(
            [
                [a, 0.0, 0.0],
                [b * cos_gamma, b * sin_gamma, 0.0],
                [
                    c * cos_beta,
                    c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                    c * v / sin_gamma,
                ],
            ],
            dtype=torch.float32,
        )

        return lattice

    def _parse_atoms(self, block: CifBlock) -> tuple[list[list[float]], list[int]]:
        """Fractional coordinates and atomic numbers of the atom sites of a block.

        The element is the one of _atom_site_type_symbol, else of the label.
        Sites of unknown coordinates ("?" or ".") are left out.

        Raises:
            NoStructure: If the block has no atom site of known coordinates.
            ValueError: If a site has no element.
        """
        sites = block.table("_atom_site_fract_x")
        if sites is None:
            raise NoStructure("_atom_site_fract_x not found in CIF")
        for key in ("_atom_site_fract_y", "_atom_site_fract_z"):
            if key not in sites:
                raise ValueError(f"Missing required atom site column: {key[11:]}")
        symbols = sites.get("_atom_site_type_symbol") or sites.get("_atom_site_label")
        if symbols is None:
            raise ValueError("Missing atom site column: type_symbol or label")

        coords = []
        atomic_numbers = []
        columns = (
            sites["_atom_site_fract_x"],
            sites["_atom_site_fract_y"],
            sites["_atom_site_fract_z"],
        )
        occupancy = sites.get("_atom_site_occupancy") or ["1"] * len(symbols)
        occupancies = []
        for symbol, occupied, *values in zip(symbols, occupancy, *columns):
            if any(value in ("?", ".") for value in values):
                continue
            element = self._extract_element_symbol(symbol)
            try:
                number = symbol_to_number(element)
            except KeyError as error:
                raise ValueError(f"unknown element {symbol!r} in CIF") from error
            coords.append([self._parse_fractional(value) % 1.0 for value in values])
            atomic_numbers.append(number)
            occupancies.append(
                1.0 if occupied in ("?", ".") else self._parse_fractional(occupied)
            )

        if not coords:
            raise NoStructure("No atoms found in CIF")

        return coords, self._dominant_elements(coords, atomic_numbers, occupancies)

    @staticmethod
    def _dominant_elements(
        coords: list[list[float]], atomic_numbers: list[int], occupancies: list[float]
    ) -> list[int]:
        """Element of each site, the most occupied one of the sites at its place.

        Sites closer than MERGE_TOLERANCE share a place, a mixed site: each
        takes the element of largest occupancy among them, the last one on a
        tie, as ase.spacegroup.crystal. The images of the sites are merged
        after, keeping the first site.
        """
        x = torch.tensor(coords, dtype=torch.float64)
        difference = x.unsqueeze(1) - x.unsqueeze(0)
        difference = difference - torch.round(difference)
        same = torch.sum(difference**2, dim=-1) < MERGE_TOLERANCE**2

        occupancy = torch.tensor(occupancies, dtype=torch.float64)
        shared = torch.where(same, occupancy, -torch.inf)
        largest = shared == shared.max(dim=1, keepdim=True).values
        index = torch.arange(len(coords))
        choice = torch.where(largest, index, -1).argmax(dim=1)

        return [atomic_numbers[i] for i in choice.tolist()]


_parser = CifParser()


def parse_cif(cif_string: str) -> Structures:
    """Parse a CIF string into a Structures object.

    Args:
        cif_string: CIF file content as a string. May contain one or
                multiple data blocks. Each data block is parsed as a
                separate structure and batched together.

    Returns:
        Structures object containing all parsed structures batched together.

    Raises:
        ValueError: If no block describes a structure, or required data is
            missing.
    """
    return _parser.parse(cif_string)


def parse_blocks_with_errors(cif_string: str) -> List[ParseResult]:
    """Parse CIF string, yielding ParseResult for each block with error handling.

    Args:
        cif_string: CIF file content as a string.

    Yields:
        ParseResult for each data block.
    """
    return _parser.parse_with_errors(cif_string)


class CifDataset(Dataset):
    """Dataset for loading crystal structures from CIF files.

    Args:
        root_dir: Directory containing CIF files.
        transform: Optional transform to apply to structures.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Structures], Structures] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cif_files = sorted(self.root_dir.glob("*.cif"))
        if not self.cif_files:
            raise ValueError(f"No CIF files found in {self.root_dir}")

    def __len__(self) -> int:
        """Return the number of CIF files in the dataset."""
        return len(self.cif_files)

    def __getitem__(self, idx: int) -> Structures:
        """Get a structure by index.

        Args:
            idx: Index of the structure to retrieve.

        Returns:
            Structures object containing the parsed crystal structure.
        """
        cif_path = self.cif_files[idx]
        cif_string = cif_path.read_text()
        structures = parse_cif(cif_string)
        if self.transform is not None:
            structures = self.transform(structures)
        return structures
