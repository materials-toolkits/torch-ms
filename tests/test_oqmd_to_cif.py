"""Tests for the selection of OQMD entries and their conversion to CIF."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from oqmd_to_cif import (  # noqa: E402
    LATTICE,
    rejection,
    select_calculations,
    write_structures,
)

from pycrystalgen.data import parse_cif  # noqa: E402

CUBIC = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0]


def make_structures(rows):
    """rows: (id, natoms)"""
    return pd.DataFrame(
        [[i, None, "", natoms] + CUBIC for i, natoms in rows],
        columns=["id", "entry_id", "label", "natoms"] + LATTICE,
    )


def make_calculations(rows):
    """rows: (id, entry_id, label, output_id, energy_pa, band_gap, converged)"""
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "entry_id",
            "label",
            "output_id",
            "energy_pa",
            "band_gap",
            "converged",
        ],
    )


def make_formation_energies(rows):
    """rows: (id, entry_id, calculation_id, fit_id, delta_e, stability)"""
    return pd.DataFrame(
        rows,
        columns=["id", "entry_id", "calculation_id", "fit_id", "delta_e", "stability"],
    )


def select(structures, calculations, formation_energies=(), atom_counts=None):
    structures = make_structures(structures)
    if atom_counts is None:
        atom_counts = pd.Series(
            structures["natoms"].to_numpy(), index=structures["id"].to_numpy()
        )
    return select_calculations(
        structures,
        make_calculations(calculations),
        make_formation_energies(list(formation_energies)),
        atom_counts,
    )


def test_lowest_energy_is_selected():
    selected, report = select(
        [(100, 2), (101, 2), (102, 2)],
        [
            (1, 7, "standard", 100, -1.0, 0.5, 1),
            (2, 7, "static", 101, -2.0, 1.5, 1),
            (3, 8, "static", 102, -3.0, None, 1),
        ],
    )

    assert selected["entry_id"].tolist() == [7, 8]
    assert selected["calculation_id"].tolist() == [2, 3]
    assert selected["structure_id"].tolist() == [101, 102]
    assert selected["energy_pa"].tolist() == [-2.0, -3.0]
    assert selected["band_gap"].iloc[0] == 1.5
    assert np.isnan(selected["band_gap"].iloc[1])
    assert report["entries_selected"] == 2


def test_equal_energies_select_lowest_calculation_id():
    selected, _ = select(
        [(100, 2), (101, 2)],
        [(5, 7, "static", 101, -2.0, None, 1), (4, 7, "standard", 100, -2.0, None, 1)],
    )

    assert selected["calculation_id"].tolist() == [4]


def test_rejected_calculations():
    selected, report = select(
        [(100, 2), (101, 2), (102, 2), (103, 2), (104, 2)],
        [
            (1, 1, "static", 100, -9.0, None, 0),  # not converged
            (2, 2, "relaxation", 101, -9.0, None, 1),  # not static or standard
            (3, 3, "static_lda", 102, -9.0, None, 1),  # lda
            (4, 4, "static", 103, None, None, 1),  # no energy
            (5, 5, "static", 999, -9.0, None, 1),  # output is not a structure
            (6, 6, "static", None, -9.0, None, 1),  # no output
            (7, 7, "standard", 104, -1.0, None, 1),
        ],
    )

    assert selected["entry_id"].tolist() == [7]
    assert report["calculations_not_converged"] == 1
    assert report["calculations_not_static_or_standard"] == 1
    assert report["calculations_lda"] == 1
    assert report["calculations_without_energy"] == 1
    assert report["calculations_without_valid_output"] == 2


def test_inconsistent_number_of_atoms():
    selected, report = select(
        [(100, 2), (101, 3), (102, 2)],
        [
            (1, 7, "static", 100, -5.0, None, 1),  # atoms table has 3 rows
            (2, 7, "static", 101, -1.0, None, 1),
            (3, 8, "static", 102, -1.0, None, 1),  # no row in the atoms table
        ],
        atom_counts=pd.Series([3, 3], index=[100, 101]),
    )

    assert selected["structure_id"].tolist() == [101]
    assert report["structures_with_inconsistent_natoms"] == 2


def test_formation_energy():
    selected, report = select(
        [(100, 2), (101, 2), (102, 2), (103, 2)],
        [
            (1, 7, "static", 100, -1.0, None, 1),
            (2, 8, "static", 101, -1.0, None, 1),
            (3, 9, "static", 102, -1.0, None, 1),
            (4, 10, "static", 103, -1.0, None, 1),
        ],
        [
            (1, 7, 1, "standard", -0.5, 0.01),  # matches the calculation
            (2, 7, 99, "standard", -0.9, 0.09),
            (3, 7, 1, "other", -0.7, 0.07),  # other fit
            (4, 8, None, "standard", -0.4, -0.02),  # only one for the entry
            (5, 9, 98, "standard", -0.3, 0.03),  # ambiguous: two other calculations
            (6, 9, 97, "standard", -0.2, 0.02),
        ],
    )

    assert selected["delta_e"].tolist()[:2] == [-0.5, -0.4]
    assert selected["stability"].tolist()[:2] == [0.01, -0.02]
    assert selected["delta_e"].isna().tolist() == [False, False, True, True]
    assert report["entries_with_formation_energy"] == 2


def make_atoms(rows):
    return pd.DataFrame(
        rows, columns=["structure_id", "element_id", "x", "y", "z", "occupancy"]
    )


def test_write_structures():
    selected, _ = select(
        [(100, 2), (101, 1)],
        [(1, 7, "static", 100, -2.5, 1.2, 1), (2, 8, "static", 101, -1.0, None, 1)],
        [(1, 7, 1, "standard", -0.5, -0.01)],
    )
    atoms = make_atoms(
        [
            (101, "Fe", 0.0, 0.0, 0.0, 1.0),
            (100, "Na", 0.0, 0.0, 0.0, 1.0),
            (999, "Xe", 0.0, 0.0, 0.0, 1.0),
            (100, "Cl", 0.5, 0.25, -0.25, 1.0),
        ]
    )

    fp = io.StringIO()
    written, rejected = write_structures(fp, selected, atoms)
    structures = parse_cif(fp.getvalue())

    assert written.tolist() == [True, True]
    assert rejected == {}
    assert structures.num_atoms.tolist() == [2, 1]
    assert structures.z.tolist() == [11, 17, 26]
    assert torch.allclose(structures.x[1], torch.tensor([0.5, 0.25, 0.75]))
    assert torch.allclose(structures.total_energy, torch.tensor([-5.0, -1.0]))
    assert torch.isclose(structures.formation_energy_per_atom[0], torch.tensor(-0.5))
    assert torch.isclose(structures.energy_above_hull[0], torch.tensor(-0.01))
    assert torch.isclose(structures.band_gap[0], torch.tensor(1.2))
    assert torch.isnan(structures.band_gap[1])
    assert torch.isnan(structures.formation_energy_per_atom[1])


@pytest.mark.parametrize(
    "lattice,frac,symbols,occupancy,expected",
    [
        (np.eye(3) * 4, [[0.0, 0.0, 0.0]], ["Na"], [1.0], None),
        (np.zeros((3, 3)), [[0.0, 0.0, 0.0]], ["Na"], [1.0], "lattice_degenerate"),
        (
            [[4.0, 0, 0], [8.0, 0, 0], [0, 0, 4.0]],
            [[0.0, 0.0, 0.0]],
            ["Na"],
            [1.0],
            "lattice_degenerate",
        ),
        (np.eye(3) * np.nan, [[0.0, 0.0, 0.0]], ["Na"], [1.0], "lattice_not_finite"),
        (np.eye(3) * 4, [[np.nan, 0.0, 0.0]], ["Na"], [1.0], "coordinates_not_finite"),
        (np.eye(3) * 4, [[0.0, 0.0, 0.0]], ["Na"], [0.5], "partial_occupancy"),
        (np.eye(3) * 4, [[0.0, 0.0, 0.0]], [None], [1.0], "unknown_element"),
        (np.eye(3) * 4, [[0.0, 0.0, 0.0]], ["D"], [1.0], "unknown_element"),
    ],
)
def test_rejection(lattice, frac, symbols, occupancy, expected):
    reason = rejection(
        np.array(lattice, dtype=float),
        np.array(frac),
        symbols,
        np.array(occupancy),
        valid_symbols={"Na", "Cl"},
    )

    assert reason == expected


def make_selected(rows):
    """rows: (entry_id, energy_pa)"""
    return pd.DataFrame(rows, columns=["entry_id", "energy_pa"])


def test_drop_duplicates_keeps_lowest_energy():
    from oqmd_to_cif import drop_duplicates

    selected = make_selected([(1, -1.0), (2, -3.0), (3, -2.0), (4, -5.0), (5, -4.0)])
    # 2 and 3 are duplicates of 1, 4 has no information, 5 references itself
    duplicate_of = pd.Series([1.0, 1.0, 1.0, None, 5.0], index=[1, 2, 3, 4, 5])

    unique = drop_duplicates(selected, duplicate_of)

    assert unique["entry_id"].tolist() == [2, 4, 5]
    assert unique["energy_pa"].tolist() == [-3.0, -5.0, -4.0]


def test_drop_duplicates_ties_keep_the_original():
    from oqmd_to_cif import drop_duplicates

    selected = make_selected([(7, -2.0), (8, -2.0), (9, -2.0), (10, -1.0), (11, -1.0)])
    # original 8 is in the middle of its group, original 12 is not selected
    duplicate_of = pd.Series([8.0, 8.0, 8.0, 12.0, 12.0], index=[7, 8, 9, 10, 11])

    unique = drop_duplicates(selected, duplicate_of)

    assert unique["entry_id"].tolist() == [8, 10]


def test_drop_duplicates_keeps_order_and_other_columns():
    from oqmd_to_cif import drop_duplicates

    selected = make_selected([(1, -1.0), (2, -2.0), (3, -3.0)]).assign(natoms=[4, 5, 6])
    duplicate_of = pd.Series([3.0, 2.0, 3.0], index=[1, 2, 3])

    unique = drop_duplicates(selected, duplicate_of)

    assert unique["entry_id"].tolist() == [2, 3]
    assert unique["natoms"].tolist() == [5, 6]
    assert list(unique.columns) == ["entry_id", "energy_pa", "natoms"]
    assert unique.index.tolist() == [0, 1]
