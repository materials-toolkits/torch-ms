import pytest
import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.atomic_radius import atomic_radius


@pytest.fixture
def zincblende():
    return Structures.zincblende()


def test_atomic_radius_scaling_1_1_no_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=1.1, ensure_gap=0.0)
    assert graph.num_edges.tolist() == [32]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426]), atol=1e-3)


def test_atomic_radius_scaling_2_0_with_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=2.0, ensure_gap=0.5)
    assert graph.num_edges.tolist() == [32]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426]), atol=1e-3)


def test_atomic_radius_scaling_2_0_no_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=2.0, ensure_gap=0.0)
    assert graph.num_edges.tolist() == [176]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426, 3.8254, 4.4857]), atol=1e-3)
