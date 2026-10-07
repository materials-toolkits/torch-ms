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
from .symmetry import generate_symmetric_structure

logger = logging.getLogger(__name__)


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

    Support only P1 spacegroup.

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

        Raises:
            ValueError: If space group is not 1 or required data is missing.
        """
        blocks = self._find_data_blocks(cif_string)
        if not blocks:
            raise ValueError("No data blocks found in CIF")

        structures = [self._parse_single_block(block) for block in blocks]
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
        spacegroup = self._validate_space_group(block_string)
        cell_params = self._parse_cell_parameters(block_string)
        lattice = self._build_lattice_vectors(cell_params)
        coords, atomic_numbers = self._parse_atoms(block_string)
        total_energy, energy_above_hull, band_gap, formation_energy_per_atom = (
            self._parse_properties(block_string)
        )

        x = torch.tensor(coords, dtype=torch.float64)
        z = torch.tensor(atomic_numbers, dtype=torch.int64)

        structures = generate_symmetric_structure(lattice, x, z, spacegroup)

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

    def _validate_space_group(self, cif_string: str) -> int:
        """Validate and return space group number."""
        space_group_match = self._space_group.search(cif_string)
        if space_group_match is None:
            raise ValueError("Space group number not found in CIF")

        space_group = int(space_group_match.group(1))

        if space_group != 1:
            raise NotImplementedError("Spacegroups other that P1 are not supported")

        return space_group

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
        """Extract element symbol from symbol with optional oxidation state."""
        match = re.match(r"^([A-Z][a-z]?)", symbol)
        if match:
            return match.group(1)
        return symbol

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

    def _find_atom_site_loop(self, cif_string: str) -> tuple[int, int, list[str]]:
        """Find the atom site loop and return (start, end, headers).

        Args:
            cif_string: CIF file content.

        Returns:
            Tuple of (loop_start, loop_end, list of header column names).
        """
        fract_x_match = re.search(r"_atom_site_fract_x", cif_string)
        if fract_x_match is None:
            raise ValueError("_atom_site_fract_x not found in CIF")

        fract_x_pos = fract_x_match.start()

        loop_positions = []
        search_start = 0
        while True:
            pos = cif_string.find("loop_", search_start)
            if pos == -1 or pos >= fract_x_pos:
                break
            loop_positions.append(pos)
            search_start = pos + 1

        if not loop_positions:
            raise ValueError("loop_ not found before _atom_site_fract_x")

        atom_loop_start = loop_positions[-1]

        loop_start_line_end = cif_string.find("\n", atom_loop_start)
        lines_after_loop = cif_string[loop_start_line_end + 1 :].split("\n")

        headers = []
        for line in lines_after_loop:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("_"):
                headers.append(stripped)
                continue
            break

        num_header_lines = len(headers)
        loop_data_start = loop_start_line_end + 1
        for _ in range(num_header_lines):
            loop_data_start = cif_string.find("\n", loop_data_start) + 1

        next_data = re.search(r"\ndata_", cif_string[loop_data_start:])
        if next_data:
            loop_end = loop_data_start + next_data.start()
        else:
            loop_end = len(cif_string)

        return loop_data_start, loop_end, headers

    def _parse_atom_header_indices(self, headers: list[str]) -> dict[str, int]:
        """Get indices for required atom site headers."""
        header_map: dict[str, int] = {}
        for i, header in enumerate(headers):
            if header == "_atom_site_label":
                header_map["label"] = i
            elif header == "_atom_site_type_symbol":
                header_map["type_symbol"] = i
            elif header == "_atom_site_fract_x":
                header_map["fract_x"] = i
            elif header == "_atom_site_fract_y":
                header_map["fract_y"] = i
            elif header == "_atom_site_fract_z":
                header_map["fract_z"] = i
        return header_map

    def _parse_atoms(self, cif_string: str) -> tuple[list[list[float]], list[int]]:
        """Parse atomic coordinates and atomic numbers from CIF."""
        loop_start, loop_end, headers = self._find_atom_site_loop(cif_string)
        indices = self._parse_atom_header_indices(headers)

        required = ["fract_x", "fract_y", "fract_z"]
        for key in required:
            if key not in indices:
                raise ValueError(f"Missing required atom site column: {key}")

        loop_content = cif_string[loop_start:loop_end]
        lines = loop_content.strip().split("\n")

        coords = []
        atomic_numbers = []
        fract_indices = (indices["fract_x"], indices["fract_y"], indices["fract_z"])
        for line in lines:
            atom_data = self._parse_atom_line(
                line, fract_indices, indices.get("type_symbol"), indices.get("label")
            )
            if atom_data is not None:
                coords.append(atom_data[0])
                atomic_numbers.append(atom_data[1])

        if not coords:
            raise ValueError("No atoms found in CIF")

        return coords, atomic_numbers

    def _parse_atom_line(
        self,
        line: str,
        fract_indices: tuple[int, int, int],
        type_symbol_idx: int | None,
        label_idx: int | None,
    ) -> tuple[list[float], int] | None:
        """Parse a single atom line from CIF.

        Returns:
            Tuple of (coords, atomic_number) or None if line should be skipped.
        """
        stripped = line.strip()
        if not stripped or stripped.startswith(("_", "loop_", "#")):
            return None

        parts = stripped.split()
        fract_x_idx, fract_y_idx, fract_z_idx = fract_indices

        if len(parts) <= max(fract_x_idx, fract_y_idx, fract_z_idx):
            return None

        if type_symbol_idx is not None and len(parts) > type_symbol_idx:
            element = self._extract_element_symbol(parts[type_symbol_idx])
        elif label_idx is not None:
            element = self._extract_element_symbol(parts[label_idx])
        else:
            return None

        x = self._parse_fractional(parts[fract_x_idx]) % 1.0
        y = self._parse_fractional(parts[fract_y_idx]) % 1.0
        z = self._parse_fractional(parts[fract_z_idx]) % 1.0

        return [x, y, z], symbol_to_number(element)


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
        ValueError: If space group is not 1 or required data is missing.
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
