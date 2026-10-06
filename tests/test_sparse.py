import pytest
import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.atomic_radius import atomic_radius
from pycrystalgen.graph.sparse import scatter, equivalent_partition


def mix(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    x &= 0xFFFFFFFF
    return x


def hash_single(x):
    h1 = 0
    h2 = 0
    h3 = 0
    for i in x:
        h = mix(int(i.item()))
        h1 = (h1 + h) & 0xFFFFFFFF
        h2 = (h2 + (h * h)) & 0xFFFFFFFF
        h3 = (h3 + (h * h * h)) & 0xFFFFFFFF

    return mix(h1 ^ h2 ^ h3)


def test_atomic_radius_bidi_filter_requires_negated_shift():
    """The bidi filter must reject (i,j,s)+(j,i,s) pairs where the shift is not negated.

    Edges 0-1 are proper reverses: (0,1,[+1,0,0]) and (1,0,[-1,0,0]) → both kept.
    Edges 2-3 share the same shift:  (0,2,[+1,0,0]) and (2,0,[+1,0,0]) → both dropped,
    because (2,0,[-1,0,0]) is missing.  A filter that only checks node pairs would
    wrongly keep them.
    """
    pair_idx = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    edge_shift = torch.tensor([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]])

    n_edges = pair_idx.shape[1]
    fwd = torch.cat([pair_idx.t(), edge_shift], dim=1)
    rev = torch.cat([pair_idx[[1, 0]].t(), -edge_shift], dim=1)
    _, uid = torch.unique(torch.cat([fwd, rev], dim=0), dim=0, return_inverse=True)
    bidi_mask = torch.isin(uid[:n_edges], uid[n_edges:])

    assert bidi_mask.tolist() == [True, True, False, False]


def test_atomic_radius_undirected():
    """Every edge (i, j, s) must have a reverse edge (j, i) with shift -s."""
    from pycrystalgen.data.batching import Structures
    from pycrystalgen.graph.atomic_radius import atomic_radius

    graph = atomic_radius(Structures.zincblende(), scaling=2.0)
    senders = graph.edge_index[0]
    receivers = graph.edge_index[1]
    es = graph.edge_shift  # (n_edges, 3)

    # For each edge e, find its reverse candidates (j -> i) and check that
    # at least one carries exactly -s (independent of s itself).
    for e in range(senders.shape[0]):
        i, j, s = senders[e], receivers[e], es[e]
        rev = (senders == j) & (receivers == i)
        rev_shifts = es[rev]  # shifts of all (j -> i) edges
        has_neg_s = (rev_shifts == -s).all(dim=1).any()
        assert (
            has_neg_s
        ), f"No reverse edge ({j},{i},{(-s).tolist()}) for edge ({i},{j},{s.tolist()})"


def test_scatter_hash():
    x0 = torch.tensor([6, 4, 5, 8, 6, 2, 3, 6, 8])
    x1 = torch.tensor([6, 3, 1, 7, 9, 5])
    x2 = torch.tensor([3, 6, 4, 9, 6, 2, 1, 4])

    x = torch.cat([x0, x1, x2])
    batch = torch.cat(
        [
            torch.full_like(x0, fill_value=0, dtype=torch.int64),
            torch.full_like(x1, fill_value=1, dtype=torch.int64),
            torch.full_like(x2, fill_value=2, dtype=torch.int64),
        ]
    )

    original_hash = scatter(x, batch, n_out=3, reduce="hash")

    perm = torch.randperm(x0.shape[0] + x1.shape[0] + x2.shape[0])
    x = x[perm]
    batch = batch[perm]

    permuted_hashed = scatter(x, batch, n_out=3, reduce="hash")

    assert (original_hash == permuted_hashed).all()

    assert original_hash[0] == hash_single(x0)
    assert original_hash[1] == hash_single(x1)
    assert original_hash[2] == hash_single(x2)


def test_equivalent_partition_true():
    # a and b define the same grouping: {0,1} and {2} in graph 0
    a = torch.tensor([0, 0, 1])
    b = torch.tensor([7, 7, 3])
    batch = torch.zeros(3, dtype=torch.long)
    result = equivalent_partition(a, b, batch)
    assert result.tolist() == [True]


def test_equivalent_partition_false_same_counts():
    # a-class 0 maps to b-classes {0,1} — not a bijection, but count_a == count_b
    a = torch.tensor([0, 0, 1])
    b = torch.tensor([0, 1, 1])
    batch = torch.zeros(3, dtype=torch.long)
    result = equivalent_partition(a, b, batch)
    assert result.tolist() == [False]


def test_equivalent_partition_batched():
    # graph 0: equivalent, graph 1: not equivalent
    a = torch.tensor([0, 0, 1,  0, 0, 1])
    b = torch.tensor([5, 5, 9,  5, 9, 9])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    result = equivalent_partition(a, b, batch)
    assert result.tolist() == [True, False]
