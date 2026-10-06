"""Tests for the Materials Project conversion script."""

import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from materials_project_to_cif import (  # noqa: E402
    DisorderedStructure,
    site_symbols,
    write_row,
)

from pycrystalgen.data import parse_cif  # noqa: E402


def make_row(**overrides):
    row = {
        "material_id": "mp-test",
        "deprecated": False,
        "nsites": 2,
        "energy_per_atom": -5.25,
        "formation_energy_per_atom": -1.5,
        "energy_above_hull": 0.02,
        "band_gap": 1.1,
        "structure": {
            "lattice": {"matrix": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 0.0, 6.0]]},
            "sites": [
                {
                    "abc": [0.0, 0.0, 0.0],
                    "species": [{"element": "Na", "occu": 1.0}],
                },
                {
                    "abc": [0.5, 0.25, 0.75],
                    "species": [{"element": "Cl", "occu": 1.0}],
                },
            ],
        },
    }
    row.update(overrides)
    return row


def convert_row(row):
    fp = io.StringIO()
    write_row(fp, row)
    return parse_cif(fp.getvalue())


def test_structure():
    structures = convert_row(make_row())

    assert structures.z.tolist() == [11, 17]
    assert torch.allclose(
        structures.x, torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.25, 0.75]]), atol=1e-6
    )

    lattice = torch.tensor([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 0.0, 6.0]])
    parsed = structures.lattice[0]
    assert torch.allclose(parsed @ parsed.T, lattice @ lattice.T, atol=1e-4)


def test_labels():
    structures = convert_row(make_row())

    assert torch.isclose(structures.total_energy[0], torch.tensor(-10.5))
    assert torch.isclose(structures.formation_energy_per_atom[0], torch.tensor(-1.5))
    assert torch.isclose(structures.energy_above_hull[0], torch.tensor(0.02))
    assert torch.isclose(structures.band_gap[0], torch.tensor(1.1))


def test_missing_labels():
    structures = convert_row(
        make_row(
            energy_per_atom=None,
            formation_energy_per_atom=None,
            energy_above_hull=None,
            band_gap=None,
        )
    )

    assert torch.isnan(structures.total_energy).all()
    assert torch.isnan(structures.formation_energy_per_atom).all()
    assert torch.isnan(structures.energy_above_hull).all()
    assert torch.isnan(structures.band_gap).all()


@pytest.mark.parametrize(
    "species",
    [
        [{"element": "Na", "occu": 0.5}],
        [{"element": "Na", "occu": 0.5}, {"element": "K", "occu": 0.5}],
    ],
)
def test_disordered_site_is_rejected(species):
    row = make_row()
    row["structure"]["sites"][0]["species"] = species

    with pytest.raises(DisorderedStructure):
        site_symbols(row)

    with pytest.raises(DisorderedStructure):
        write_row(io.StringIO(), row)
