import torch

from pycrystalgen.data import Structures
from pycrystalgen.data.filters import Filter


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
