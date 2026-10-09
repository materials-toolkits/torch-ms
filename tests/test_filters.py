import dataclasses

import pytest
import torch

from torchms.data import Structures
from torchms.data.filters import Filter


def make_batch():
    batch = Structures.collate(
        [
            Structures.diamond(),
            Structures.zincblende(),
            Structures.diamond(),
        ]
    )
    batch.energy_above_hull = torch.tensor([0.0, 0.05, float("nan")])
    return batch


def test_no_filter_keeps_everything():
    assert Filter()(make_batch()).tolist() == [True, True, True]


def test_max_atoms():
    batch = make_batch()
    n = int(batch.num_atoms.max())
    assert Filter(max_atoms=n)(batch).tolist() == [True, True, True]
    assert Filter(max_atoms=n - 1)(batch).tolist() == (batch.num_atoms < n).tolist()


def test_exclude_elements():
    assert Filter(exclude_elements=(16,))(make_batch()).tolist() == [True, False, True]
    assert Filter(exclude_elements=(2, 6))(make_batch()).tolist() == [
        False,
        True,
        False,
    ]


def test_max_energy_above_hull_drops_unknown():
    assert Filter(max_energy_above_hull=0.0)(make_batch()).tolist() == [
        True,
        False,
        False,
    ]
    assert Filter(max_energy_above_hull=0.1)(make_batch()).tolist() == [
        True,
        True,
        False,
    ]


def test_combined():
    filt = Filter(exclude_elements=(16,), max_energy_above_hull=0.1)
    assert filt(make_batch()).tolist() == [True, False, False]


def labelled() -> Structures:
    """Diamond, zincblende, water and a molecule of hydrogen."""
    batch = Structures.collate(
        [
            Structures.diamond(),
            Structures.zincblende(),
            Structures.water(),
            Structures.molecule([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], [1, 1]),
        ]
    )

    return dataclasses.replace(
        batch,
        band_gap=torch.tensor([5.5, 3.7, torch.nan, 11.0]),
        properties={"disorder": torch.tensor([0.0, 1.0, 0.0, torch.nan])},
    )


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"min_atoms": 3}, [True, True, True, False]),
        ({"min_atoms": 3, "max_atoms": 3}, [False, False, True, False]),
        ({"elements": (1, 8)}, [False, False, True, True]),
        ({"elements": (6, 16, 30)}, [True, True, False, False]),
        ({"elements": ()}, [False, False, False, False]),
        ({"elements": (1, 8), "exclude_elements": (8,)}, [False, False, False, True]),
        ({"periodic": True}, [True, True, False, False]),
        ({"periodic": False}, [False, False, True, True]),
        ({"minimum": {"band_gap": 4.0}}, [True, False, False, True]),
        ({"maximum": {"band_gap": 5.5}}, [True, True, False, False]),
        (
            {"minimum": {"band_gap": 4.0}, "maximum": {"band_gap": 6.0}},
            [True, False, False, False],
        ),
        ({"equal": {"disorder": 0}}, [True, False, True, False]),
        ({"known": ("band_gap",)}, [True, True, False, True]),
        ({"known": ("band_gap", "disorder")}, [True, True, False, False]),
        ({"known": ("total_energy",)}, [False, False, False, False]),
        ({"periodic": False, "equal": {"disorder": 0}}, [False, False, True, False]),
    ],
)
def test_settings(settings, expected):
    assert Filter(**settings)(labelled()).tolist() == expected


def test_partly_periodic_is_neither():
    slab = dataclasses.replace(
        Structures.diamond(), pbc=torch.tensor([[True, True, False]])
    )

    assert Filter(periodic=True)(slab).tolist() == [False]
    assert Filter(periodic=False)(slab).tolist() == [False]


def test_unknown_name():
    with pytest.raises(KeyError, match="unknown label or property"):
        Filter(minimum={"homo": 0.0})(labelled())
    with pytest.raises(KeyError, match="disorder"):
        Filter(known=("gap",))(labelled())


def test_settings_are_shown():
    assert "max_atoms=64" in repr(Filter(max_atoms=64, equal={"disorder": 0}))
