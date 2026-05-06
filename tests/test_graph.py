import torch

from pycrystalgen.graph.cutoff import lattice_heights


def test_lattice_heights_monoclinic():
    # a=[3,0,0], b=[1,2,0], c=[0,0,4]
    # h_a = 6/√5 (less than |a|=3 due to shear), h_b = 2, h_c = 4
    L = torch.tensor([[[3.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]]])
    expected = torch.tensor([[6.0 / 5.0 ** 0.5, 2.0, 4.0]])
    assert torch.allclose(lattice_heights(L), expected, atol=1e-5)
