import json
from pathlib import Path

import numpy as np
import pytest
import spglib
import torch

from torchms.data.constants import get_symmetry_operations, symbol_to_number
from torchms.data.symmetry import generate_symmetric_structure
from torchms.data.compare import compare_structures
from torchms.data.batching import Structures


@pytest.fixture
def cubic_lattice():
    return torch.tensor(
        [[3.567, 0.0, 0.0], [0.0, 3.567, 0.0], [0.0, 0.0, 3.567]],
        dtype=torch.float64,
    )


@pytest.fixture
def diamond_basis():
    positions = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
    atomic_numbers = torch.tensor([6], dtype=torch.int64)
    return positions, atomic_numbers


def load_expected(cif_path: Path) -> (dict[str, float], torch.Tensor, torch.Tensor):
    """Load expected values from JSON file with same name as CIF."""
    json_path = cif_path.with_suffix(".json")
    with open(json_path) as f:
        data = json.load(f)

    coords = torch.tensor([atom["coords"] for atom in data["atoms"]])
    atomic_numbers = torch.tensor(
        [symbol_to_number(atom["element"]) for atom in data["atoms"]]
    )

    return (data["cell_parameters"], coords, atomic_numbers)


class TestDiamondStructure:
    """Test diamond crystal structure generation using spglib.

    Diamond structure (Fd-3m) has 8 atoms in the primitive cell.
    Basis: (0,0,0) and (1/4,1/4,1/4) in FCC lattice.
    """

    def test_diamond_symmetrized(self, cubic_lattice, diamond_basis):
        positions, atomic_numbers = diamond_basis
        struct = generate_symmetric_structure(
            cubic_lattice, positions, atomic_numbers, 227
        )

        expected_cell, expected_x, expected_z = load_expected(
            Path("tests/cif/diamond.json")
        )

        expected = Structures(struct.lattice, expected_x, expected_z)

        errors = compare_structures(struct, expected)
        assert not errors, "\n".join(errors)
