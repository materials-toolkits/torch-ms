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


# Wyckoff positions, from pyxtal


def cell_of_the_family(number: int) -> torch.Tensor:
    """A cell with the metric of the crystal family of a space group, no more."""
    from torchms.graph.geometry import lattice_from_parameters

    if number <= 2:
        lengths, angles = [4.1, 5.3, 6.2], [81.0, 97.0, 104.0]
    elif number <= 15:
        lengths, angles = [4.1, 5.3, 6.2], [90.0, 104.0, 90.0]
    elif number <= 74:
        lengths, angles = [4.1, 5.3, 6.2], [90.0, 90.0, 90.0]
    elif number <= 142:
        lengths, angles = [4.1, 4.1, 6.2], [90.0, 90.0, 90.0]
    elif number <= 194:
        lengths, angles = [4.1, 4.1, 6.2], [90.0, 90.0, 120.0]
    else:
        lengths, angles = [5.3, 5.3, 5.3], [90.0, 90.0, 90.0]

    return lattice_from_parameters(
        torch.tensor([lengths], dtype=torch.float64),
        torch.tensor([angles], dtype=torch.float64),
    )[0]


# spglib 2.7 warns at each call that its old handling of errors is deprecated
@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING:DeprecationWarning")
@pytest.mark.parametrize("number", range(1, 231))
def test_general_position_is_the_space_group(number):
    """The general position has the operations of the group, in a standard setting.

    The one of the Hall symbol of torchms, but the origin 2 for the 24 groups
    with two origins: pyxtal takes the origin on a center of inversion.
    """
    from torchms.data.constants import get_wyckoff_positions

    positions = get_wyckoff_positions(number)
    general = positions[0].operations.numpy()
    assert torch.equal(positions[0].operations[0], torch.eye(4, dtype=torch.float64))
    assert [p.letter for p in positions][-1] == "a"
    assert all(p.multiplicity == len(p.operations) for p in positions)

    hall = get_spacegroup(number).hall_number
    if spglib.get_spacegroup_type(hall).choice == "1":
        hall += 1
    kind = spglib.get_spacegroup_type(hall)
    assert kind.number == number and kind.choice in ("", "b", "b1", "2", "H")

    expected = spglib.get_symmetry_from_database(hall)
    assert as_set(general[:, :3, :3], general[:, :3, 3]) == as_set(
        expected["rotations"], expected["translations"]
    )


# spglib 2.7 warns at each call that its old handling of errors is deprecated
@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING:DeprecationWarning")
@pytest.mark.parametrize("number", range(1, 231))
def test_wyckoff_positions_are_the_ones_spglib_finds(number):
    """An atom on each position, with one on the general position: spglib agrees.

    The atom on the general position makes the structure of the group, and no
    more; the atoms of the other position keep it if the position is one of
    the group. spglib can name a position by another letter of the same
    multiplicity, an image by a symmetry of the group of the cell.
    """
    from torchms.data.constants import get_wyckoff_positions
    from torchms.data.symmetry import WyckoffSites

    positions = get_wyckoff_positions(number)
    multiplicity = {p.letter: p.multiplicity for p in positions}
    lattice = cell_of_the_family(number).numpy()
    random = torch.Generator().manual_seed(number)

    for position in positions:
        sites = WyckoffSites.from_letters(
            [number], [[positions[0].letter, position.letter]], dtype=torch.float64
        )
        # free points drawn again until no two atoms are at the same place
        while True:
            x = sites.place(
                torch.rand(len(sites.site), 3, generator=random, dtype=torch.float64)
            )
            difference = x[:, None] - x[None]
            distance = (difference - difference.round()).norm(dim=-1)
            if (distance + torch.eye(len(x))).min() > 0.02:
                break
        z = sites.unpool_sites(torch.tensor([1, 2]))
        dataset = spglib.get_symmetry_dataset(
            (lattice, x.numpy(), z.numpy()), symprec=1e-4
        )

        assert dataset.number == number, position.letter
        found = {dataset.wyckoffs[i] for i in range(len(z)) if z[i] == 2}
        assert len(found) == 1, position.letter
        assert multiplicity[found.pop()] == position.multiplicity, position.letter
