import json
from pathlib import Path

import numpy as np
import pytest
import spglib
import torch

from torchms.data.constants import (
    get_spacegroup,
    get_symmetry_operations,
    symbol_to_number,
)
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


def as_set(rotations: np.ndarray, translations: np.ndarray) -> set[tuple]:
    """Affine operations as a set, translations modulo 1, rounded."""
    return {
        (tuple(np.rint(r).astype(int).ravel()), tuple(np.round(t % 1.0, 4) % 1.0))
        for r, t in zip(rotations, translations)
    }


# spglib 2.7 warns at each call that its old handling of errors is deprecated
@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING:DeprecationWarning")
@pytest.mark.parametrize("number", range(1, 231))
def test_space_group_table_is_the_one_of_the_international_tables(number):
    """Number, Hall setting, symbol and operations of a group agree with spglib."""
    group = get_spacegroup(number)
    kind = spglib.get_spacegroup_type(group.hall_number)

    assert group.number == kind.number == number
    assert group.hall_symbol == kind.hall_symbol
    assert group.symbol == kind.international_short
    assert get_spacegroup(group.symbol).number == number

    affine = get_symmetry_operations(group.symbol).numpy()
    expected = spglib.get_symmetry_from_database(group.hall_number)
    assert as_set(affine[:, :3, :3], affine[:, :3, 3]) == as_set(
        expected["rotations"], expected["translations"]
    )


def test_space_group_symbols():
    assert get_spacegroup(225).symbol == "Fm-3m"
    assert get_spacegroup(194).symbol == "P6_3/mmc"
    assert get_spacegroup("P2_1/c").number == 14
    assert get_spacegroup(166).crystal_system == "trigonal"
