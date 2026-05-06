import torch
import pytest

from pycrystalgen.graph.graph import Graph, InvalidGraph

# TODO: check this tests

def _make_graph(n_atoms=4, n_edges=6, n_struct=1):
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    edge_shift = torch.zeros(n_edges, 3, dtype=torch.long)
    num_atoms = torch.tensor([n_atoms] * n_struct)
    num_edges = torch.tensor([n_edges // n_struct] * n_struct)
    return Graph(edge_index=edge_index, edge_shift=edge_shift, num_atoms=num_atoms, num_edges=num_edges)


# --- construction and derived fields ---

def test_graph_derives_edge_batch():
    g = _make_graph(n_edges=6, n_struct=1)
    assert g.edge_batch.shape == (6,)
    assert (g.edge_batch == 0).all()


def test_graph_derives_edge_ptr():
    g = _make_graph(n_edges=6, n_struct=1)
    assert g.edge_ptr.tolist() == [0, 6]


def test_graph_multi_struct_edge_batch():
    edge_index = torch.zeros(2, 5, dtype=torch.long)
    edge_shift = torch.zeros(5, 3, dtype=torch.long)
    num_atoms = torch.tensor([3, 4])
    num_edges = torch.tensor([2, 3])
    g = Graph(edge_index=edge_index, edge_shift=edge_shift, num_atoms=num_atoms, num_edges=num_edges)
    assert g.edge_batch.tolist() == [0, 0, 1, 1, 1]
    assert g.edge_ptr.tolist() == [0, 2, 5]


def test_graph_invalid_edge_index_shape():
    with pytest.raises(InvalidGraph):
        Graph(
            edge_index=torch.zeros(2, 5, dtype=torch.long),
            edge_shift=torch.zeros(5, 3, dtype=torch.long),
            num_atoms=torch.tensor([3]),
            num_edges=torch.tensor([4]),   # mismatch: 4 edges but edge_index has 5
        )


def test_graph_invalid_no_structures():
    with pytest.raises((InvalidGraph, Exception)):
        Graph(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_shift=torch.zeros(0, 3, dtype=torch.long),
            num_atoms=torch.tensor([], dtype=torch.long),
            num_edges=torch.tensor([], dtype=torch.long),
        )


# --- collate ---

def test_graph_collate_edge_index_offset():
    # graph 0: atom 0→1, 2 atoms; graph 1: atom 0→1, 3 atoms
    # after collate: graph 1 edge should be 2→3 (offset by 2)
    g0 = Graph(
        edge_index=torch.tensor([[0], [1]]),
        edge_shift=torch.zeros(1, 3, dtype=torch.long),
        num_atoms=torch.tensor([2]),
        num_edges=torch.tensor([1]),
    )
    g1 = Graph(
        edge_index=torch.tensor([[0], [1]]),
        edge_shift=torch.zeros(1, 3, dtype=torch.long),
        num_atoms=torch.tensor([3]),
        num_edges=torch.tensor([1]),
    )
    batched = Graph.collate([g0, g1])
    assert batched.edge_index[:, 0].tolist() == [0, 1]
    assert batched.edge_index[:, 1].tolist() == [2, 3]


def test_graph_collate_shapes():
    g0 = _make_graph(n_atoms=4, n_edges=6)
    g1 = _make_graph(n_atoms=3, n_edges=4)
    batched = Graph.collate([g0, g1])
    assert batched.edge_index.shape == (2, 10)
    assert batched.num_atoms.tolist() == [4, 3]
    assert batched.num_edges.tolist() == [6, 4]
    assert batched.edge_ptr.tolist() == [0, 6, 10]


def test_graph_collate_edge_batch():
    g0 = _make_graph(n_atoms=4, n_edges=6)
    g1 = _make_graph(n_atoms=3, n_edges=4)
    batched = Graph.collate([g0, g1])
    assert batched.edge_batch[:6].tolist() == [0] * 6
    assert batched.edge_batch[6:].tolist() == [1] * 4


def test_graph_collate_empty_raises():
    with pytest.raises(ValueError):
        Graph.collate([])


# --- separate ---

def test_graph_separate_roundtrip():
    g0 = _make_graph(n_atoms=4, n_edges=6)
    g1 = _make_graph(n_atoms=3, n_edges=4)
    batched = Graph.collate([g0, g1])
    parts = Graph.separate(batched)
    assert len(parts) == 2
    assert parts[0].edge_index.shape == (2, 6)
    assert parts[1].edge_index.shape == (2, 4)


def test_graph_separate_resets_edge_index():
    g = Graph(
        edge_index=torch.tensor([[0], [1]]),
        edge_shift=torch.zeros(1, 3, dtype=torch.long),
        num_atoms=torch.tensor([2]),
        num_edges=torch.tensor([1]),
    )
    batched = Graph.collate([g, g])
    parts = Graph.separate(batched)
    # both structures: edge 0→1 in local indices
    assert parts[0].edge_index.tolist() == [[0], [1]]
    assert parts[1].edge_index.tolist() == [[0], [1]]


def test_graph_separate_edge_batch_is_zero():
    g0 = _make_graph(n_atoms=4, n_edges=6)
    g1 = _make_graph(n_atoms=3, n_edges=4)
    batched = Graph.collate([g0, g1])
    parts = Graph.separate(batched)
    assert (parts[0].edge_batch == 0).all()
    assert (parts[1].edge_batch == 0).all()


# --- aggregation helpers ---

def _graph_with_known_edges():
    # 3 atoms, edges: 0→1, 1→2, 0→2
    # senders=[0,1,0], receivers=[1,2,2]
    return Graph(
        edge_index=torch.tensor([[0, 1, 0], [1, 2, 2]]),
        edge_shift=torch.zeros(3, 3, dtype=torch.long),
        num_atoms=torch.tensor([3]),
        num_edges=torch.tensor([3]),
    )


def test_aggregate_edges_to_nodes_sum():
    g = _graph_with_known_edges()
    # edge features: all ones
    src = torch.ones(3, 1)
    out = g.aggregate_edges_to_nodes(src, reduce="sum")
    assert out.shape == (3, 1)
    # node 0: no incoming edges → 0
    # node 1: 1 incoming edge  → 1
    # node 2: 2 incoming edges → 2
    assert out.squeeze().tolist() == [0.0, 1.0, 2.0]


def test_aggregate_edges_to_nodes_mean():
    g = _graph_with_known_edges()
    src = torch.tensor([[1.0], [3.0], [5.0]])  # edge values 1, 3, 5
    out = g.aggregate_edges_to_nodes(src, reduce="mean")
    # node 1: receives edge 0 (value 1) → mean 1.0
    # node 2: receives edges 1,2 (values 3,5) → mean 4.0
    assert torch.allclose(out[1], torch.tensor([1.0]))
    assert torch.allclose(out[2], torch.tensor([4.0]))


def test_aggregate_nodes_to_structures_sum():
    # 2 structures: 2 atoms and 3 atoms
    g = Graph(
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_shift=torch.zeros(0, 3, dtype=torch.long),
        num_atoms=torch.tensor([2, 3]),
        num_edges=torch.tensor([0, 0]),
    )
    src = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    out = g.aggregate_nodes_to_structures(src, reduce="sum")
    assert out.shape == (2, 1)
    assert out[0].item() == 3.0   # 1+2
    assert out[1].item() == 12.0  # 3+4+5


def test_aggregate_nodes_to_structures_mean():
    g = Graph(
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_shift=torch.zeros(0, 3, dtype=torch.long),
        num_atoms=torch.tensor([2, 3]),
        num_edges=torch.tensor([0, 0]),
    )
    src = torch.tensor([[1.0], [3.0], [2.0], [4.0], [6.0]])
    out = g.aggregate_nodes_to_structures(src, reduce="mean")
    assert torch.allclose(out[0], torch.tensor([2.0]))   # (1+3)/2
    assert torch.allclose(out[1], torch.tensor([4.0]))   # (2+4+6)/3


def test_aggregate_edges_to_structures_sum():
    g = Graph(
        edge_index=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        edge_shift=torch.zeros(4, 3, dtype=torch.long),
        num_atoms=torch.tensor([2, 2]),
        num_edges=torch.tensor([2, 2]),
    )
    src = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    out = g.aggregate_edges_to_structures(src, reduce="sum")
    assert out[0].item() == 3.0   # 1+2
    assert out[1].item() == 7.0   # 3+4


def test_aggregate_invalid_reduce():
    g = _graph_with_known_edges()
    src = torch.ones(3)
    with pytest.raises(ValueError):
        g.aggregate_edges_to_nodes(src, reduce="max")
