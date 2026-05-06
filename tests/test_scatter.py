import torch
import pytest

from pycrystalgen.graph.scatter import scatter_mean, scatter_sum


# --- scatter_sum ---

def test_scatter_sum_1d():
    src   = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1, 2])
    out   = scatter_sum(src, index, n_out=3)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([3.0, 3.0, 4.0]))


def test_scatter_sum_2d():
    src   = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    out   = scatter_sum(src, index, n_out=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))


def test_scatter_sum_empty_slot():
    src   = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 0])
    out   = scatter_sum(src, index, n_out=3)
    assert torch.allclose(out, torch.tensor([3.0, 0.0, 0.0]))


def test_scatter_sum_variable_batch():
    # two structures: first has 3 atoms, second has 2 atoms
    src   = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    out   = scatter_sum(src, index, n_out=2)
    assert torch.allclose(out, torch.tensor([3.0, 2.0]))


def test_scatter_sum_preserves_device():
    src   = torch.ones(4)
    index = torch.tensor([0, 0, 1, 1])
    out   = scatter_sum(src, index, n_out=2)
    assert out.device == src.device


def test_scatter_sum_preserves_dtype():
    src   = torch.ones(4, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1])
    out   = scatter_sum(src, index, n_out=2)
    assert out.dtype == torch.float64


# --- scatter_mean ---

def test_scatter_mean_1d():
    src   = torch.tensor([1.0, 3.0, 5.0, 7.0])
    index = torch.tensor([0, 0, 1, 1])
    out   = scatter_mean(src, index, n_out=2)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([2.0, 6.0]))


def test_scatter_mean_2d():
    src   = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = torch.tensor([0, 0, 1, 1])
    out   = scatter_mean(src, index, n_out=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))


def test_scatter_mean_empty_slot_is_zero():
    src   = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 0])
    out   = scatter_mean(src, index, n_out=3)
    assert torch.allclose(out, torch.tensor([1.5, 0.0, 0.0]))


def test_scatter_mean_single_element():
    src   = torch.tensor([42.0])
    index = torch.tensor([0])
    out   = scatter_mean(src, index, n_out=1)
    assert torch.allclose(out, torch.tensor([42.0]))


def test_scatter_mean_variable_batch():
    # structure 0: atoms with values [1, 2, 3] → mean 2
    # structure 1: atoms with values [4, 6]    → mean 5
    src   = torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    out   = scatter_mean(src, index, n_out=2)
    assert torch.allclose(out, torch.tensor([2.0, 5.0]))
