import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.cutoff import centered_supercell, lattice_heights


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
    positions, offsets = centered_supercell(s, radius=2.5)
    assert positions.shape == (125, 3)
    assert offsets.shape == (125, 3)


def test_centered_supercell_central_cell_included():
    # the (0, 0, 0) shift must always be present
    a = 3.0
    s = Structures(
        lattice=torch.tensor([[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]]),
        x=torch.tensor([[0.25, 0.25, 0.25]]),
        z=torch.tensor([6]),
    )
    positions, offsets = centered_supercell(s, radius=1.0)
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
    positions, offsets = centered_supercell(s, radius=0.0)
    assert positions.shape == (1, 3)
    assert torch.allclose(positions, torch.tensor([[2.0, 2.0, 2.0]]))
