import torch
import pytest

from pycrystalgen.graph.scatter import (
    batched_cartesian_product_2,
    batched_cartesian_product_3,
    scatter_mean,
    scatter_sum,
)

# --- scatter_sum ---


def test_scatter_sum_1d():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1, 2])
    out = scatter_sum(src, index, n_out=3)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([3.0, 3.0, 4.0]))


def test_scatter_sum_2d():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    out = scatter_sum(src, index, n_out=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))


def test_scatter_sum_empty_slot():
    src = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 0])
    out = scatter_sum(src, index, n_out=3)
    assert torch.allclose(out, torch.tensor([3.0, 0.0, 0.0]))


def test_scatter_sum_variable_batch():
    # two structures: first has 3 atoms, second has 2 atoms
    src = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = scatter_sum(src, index, n_out=2)
    assert torch.allclose(out, torch.tensor([3.0, 2.0]))


def test_scatter_sum_preserves_device():
    src = torch.ones(4)
    index = torch.tensor([0, 0, 1, 1])
    out = scatter_sum(src, index, n_out=2)
    assert out.device == src.device


def test_scatter_sum_preserves_dtype():
    src = torch.ones(4, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1])
    out = scatter_sum(src, index, n_out=2)
    assert out.dtype == torch.float64


# --- scatter_mean ---


def test_scatter_mean_1d():
    src = torch.tensor([1.0, 3.0, 5.0, 7.0])
    index = torch.tensor([0, 0, 1, 1])
    out = scatter_mean(src, index, n_out=2)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([2.0, 6.0]))


def test_scatter_mean_2d():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = torch.tensor([0, 0, 1, 1])
    out = scatter_mean(src, index, n_out=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))


def test_scatter_mean_empty_slot_is_zero():
    src = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 0])
    out = scatter_mean(src, index, n_out=3)
    assert torch.allclose(out, torch.tensor([1.5, 0.0, 0.0]))


def test_scatter_mean_single_element():
    src = torch.tensor([42.0])
    index = torch.tensor([0])
    out = scatter_mean(src, index, n_out=1)
    assert torch.allclose(out, torch.tensor([42.0]))


def test_scatter_mean_variable_batch():
    # structure 0: atoms with values [1, 2, 3] → mean 2
    # structure 1: atoms with values [4, 6]    → mean 5
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = scatter_mean(src, index, n_out=2)
    assert torch.allclose(out, torch.tensor([2.0, 5.0]))


# --- batched_cartesian_product_2 ---

def test_batched_cartesian_product_2_uniform():
    # both structures: 2×3 → 6 pairs each, 12 total
    sizes_a = torch.tensor([2, 2])
    sizes_b = torch.tensor([3, 3])
    indices, batch, ptr = batched_cartesian_product_2(sizes_a, sizes_b)
    assert indices.shape == (12, 2)
    assert ptr.tolist() == [0, 6, 12]
    assert indices[:6, 0].tolist() == [0, 0, 0, 1, 1, 1]
    assert indices[:6, 1].tolist() == [0, 1, 2, 0, 1, 2]


def test_batched_cartesian_product_2_variable_sizes():
    # struct 0: 2×2=4 pairs, struct 1: 3×1=3 pairs → 7 total
    sizes_a = torch.tensor([2, 3])
    sizes_b = torch.tensor([2, 1])
    indices, batch, ptr = batched_cartesian_product_2(sizes_a, sizes_b)
    assert indices.shape == (7, 2)
    assert ptr.tolist() == [0, 4, 7]
    assert batch.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert indices[4:, 0].tolist() == [0, 1, 2]
    assert indices[4:, 1].tolist() == [0, 0, 0]


def test_batched_cartesian_product_2_indices_in_range():
    sizes_a = torch.tensor([3, 5])
    sizes_b = torch.tensor([4, 2])
    indices, batch, ptr = batched_cartesian_product_2(sizes_a, sizes_b)
    for i, (sa, sb) in enumerate(zip(sizes_a.tolist(), sizes_b.tolist())):
        mask = batch == i
        assert indices[mask, 0].max() < sa
        assert indices[mask, 1].max() < sb


# --- batched_cartesian_product_3 ---

def test_batched_cartesian_product_3_uniform():
    # 2×2×2 = 8 triplets per structure, 2 structures → 16 total
    sizes = torch.tensor([2, 2])
    indices, batch, ptr = batched_cartesian_product_3(sizes, sizes, sizes)
    assert indices.shape == (16, 3)
    assert ptr.tolist() == [0, 8, 16]


def test_batched_cartesian_product_3_variable_sizes():
    # struct 0: 2×3×1=6, struct 1: 1×2×4=8 → 14 total
    sizes_a = torch.tensor([2, 1])
    sizes_b = torch.tensor([3, 2])
    sizes_c = torch.tensor([1, 4])
    indices, batch, ptr = batched_cartesian_product_3(sizes_a, sizes_b, sizes_c)
    assert indices.shape == (14, 3)
    assert ptr.tolist() == [0, 6, 14]


def test_batched_cartesian_product_3_indices_in_range():
    sizes_a = torch.tensor([2, 3])
    sizes_b = torch.tensor([3, 2])
    sizes_c = torch.tensor([4, 5])
    indices, batch, ptr = batched_cartesian_product_3(sizes_a, sizes_b, sizes_c)
    for i, (sa, sb, sc) in enumerate(zip(sizes_a.tolist(), sizes_b.tolist(), sizes_c.tolist())):
        mask = batch == i
        assert indices[mask, 0].max() < sa
        assert indices[mask, 1].max() < sb
        assert indices[mask, 2].max() < sc


def test_batched_cartesian_product_3_no_duplicates():
    # all (a,b,c) triplets for a single structure must be unique
    sizes = torch.tensor([3])
    indices, batch, ptr = batched_cartesian_product_3(sizes, sizes, sizes)
    assert indices.unique(dim=0).shape[0] == 27
