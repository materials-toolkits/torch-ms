import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.cutoff import centered_supercell, cutoff, lattice_heights


def test_lattice_heights_monoclinic():
    # a=[3,0,0], b=[1,2,0], c=[0,0,4]
    # h_a = 6/√5 (less than |a|=3 due to shear), h_b = 2, h_c = 4
    L = torch.tensor([[[3.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]]])
    expected = torch.tensor([[6.0 / 5.0**0.5, 2.0, 4.0]])
    assert torch.allclose(lattice_heights(L), expected, atol=1e-5)


def test_centered_supercell_shape():
    # cubic cell a=2, 1 atom at origin, radius=2.5
    # n = ceil(2.5/2.0) = 2 in each direction → 5^3 = 125 tiles
    a = 2.0
    s = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.zeros(1, 3),
        z=torch.tensor([6]),
    )
    positions, offsets, atom_index = centered_supercell(s, radius=2.5)
    assert positions.shape == (125, 3)
    assert offsets.shape == (125, 3)
    assert atom_index.shape == (125,)


def test_centered_supercell_central_cell_included():
    # the (0, 0, 0) shift must always be present
    a = 3.0
    s = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.25, 0.25, 0.25]]),
        z=torch.tensor([6]),
    )
    positions, offsets, atom_index = centered_supercell(s, radius=1.0)
    assert (offsets == 0).all(dim=-1).any()


def test_centered_supercell_cartesian_positions():
    # 1 atom at fractional [0.5, 0.5, 0.5] in cubic cell a=4 → Cartesian [2, 2, 2]
    # radius=0 → n=0 in all directions → only the (0,0,0) central cell tile
    a = 4.0
    s = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.5, 0.5, 0.5]]),
        z=torch.tensor([6]),
    )
    positions, offsets, atom_index = centered_supercell(s, radius=0.0)
    assert positions.shape == (1, 3)
    assert torch.allclose(positions, torch.tensor([[2.0, 2.0, 2.0]]))


def test_centered_supercell_atom_index():
    # atom_index must point back to the correct original atom
    a = 3.0
    s = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        z=torch.tensor([6, 14]),
    )
    positions, offsets, atom_index = centered_supercell(s, radius=0.0)
    # radius=0 → 1 tile → atom_index must be [0, 1]
    assert atom_index.tolist() == [0, 1]


def test_centered_supercell_batched():
    # two structures with different numbers of atoms
    a = 3.0
    s0 = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.0, 0.0, 0.0]]),
        z=torch.tensor([6]),
    )
    s1 = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        z=torch.tensor([6, 14]),
    )
    batch = Structures.collate([s0, s1])
    positions, offsets, atom_index = centered_supercell(batch, radius=0.0)
    # radius=0 → 1 tile per structure → 1 + 2 = 3 output atoms
    assert positions.shape == (3, 3)
    assert atom_index.tolist() == [0, 1, 2]


# --- cutoff ---

def _simple_cubic(a: float) -> Structures:
    return Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.zeros(1, 3),
        z=torch.tensor([6]),
    )


def test_cutoff_self_loop():
    s = _simple_cubic(3.0)
    g = cutoff(s, radius=0.0)
    assert g.edge_index.shape[1] == 1
    assert g.edge_shift[0].tolist() == [0, 0, 0]
    assert g.edge_index[0, 0] == g.edge_index[1, 0] == 0


def test_cutoff_simple_cubic_first_shell():
    # simple cubic a=3, radius=3.5 → self + 6 face neighbours = 7 edges
    s = _simple_cubic(3.0)
    g = cutoff(s, radius=3.5)
    assert int(g.num_edges[0]) == 7
    assert g.edge_index.shape == (2, 7)
    assert g.edge_shift.shape == (7, 3)


def test_cutoff_returns_graph():
    from pycrystalgen.graph.graph import Graph
    s = _simple_cubic(3.0)
    g = cutoff(s, radius=3.5)
    assert isinstance(g, Graph)


def test_cutoff_edge_index_global():
    s = _simple_cubic(3.0)
    g = cutoff(s, radius=3.5)
    assert g.edge_index.min() >= 0
    assert g.edge_index.max() < int(s.num_atoms[0])


def test_cutoff_ptr_consistency():
    s = _simple_cubic(3.0)
    g = cutoff(s, radius=3.5)
    assert int(g.edge_ptr[-1]) == g.edge_index.shape[1]
    assert (g.edge_ptr[1:] >= g.edge_ptr[:-1]).all()


def test_cutoff_batched():
    s = _simple_cubic(3.0)
    batch = Structures.collate([s, s])
    g = cutoff(batch, radius=3.5)
    assert g.num_edges.tolist() == [7, 7]
    assert g.edge_index.shape[1] == 14
    assert g.edge_ptr.tolist() == [0, 7, 14]
    assert g.edge_index[:, :7].max() == 0
    assert g.edge_index[:, 7:].min() == 1


def test_cutoff_batched_variable_structures():
    a = 3.0
    s0 = _simple_cubic(a)
    s1 = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        z=torch.tensor([6, 6]),
    )
    batch = Structures.collate([s0, s1])
    g = cutoff(batch, radius=3.5)
    assert int(g.num_edges[0]) == 7
    mask1 = g.edge_batch == 1
    assert g.edge_index[:, mask1].min() >= 1
    assert g.edge_index[:, mask1].max() <= 2
