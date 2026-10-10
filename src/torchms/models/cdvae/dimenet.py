# Derived from cdvae (https://github.com/txie-93/cdvae), Copyright (c) 2021
# Tian Xie, Xiang Fu, MIT license, whose DimeNet++ is adapted from the Open
# Catalyst Project, Copyright (c) Facebook, Inc. and its affiliates, and from
# PyTorch Geometric, Copyright (c) 2021 Matthias Fey, Jiaxuan You, both under
# the MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the DimeNet++ encoder of cdvae
# (cdvae/pl_modules/gnn.py and the layers of torch_geometric 2.0.1 it uses),
# for inference only. The graph and its pairs of bonds are read from torchms
# Structures instead of a batch of PyTorch Geometric, the bases are computed
# with torch instead of sympy, and the option building a radius graph is
# removed. Names of modules and parameters follow cdvae to load its weights.
"""DimeNet++ encoder of CDVAE.

Features live on the bonds of a graph of the atoms. A block updates a bond
i → j from the bonds j → k leaving its receiver towards another atom than i,
in a basis of spherical Bessel functions of the length of j → k and of
spherical harmonics of the angle between the two bonds. After each block the
bonds are summed on their atom, and the atoms averaged over the structure.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.estimator.config import load_config
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.sparse import Reduce
from torchms.models.basis import SphericalBesselBasis, legendre, polynomial_cutoff

# every value of the model is in config.yaml, next to this module
CONFIG = load_config(__file__)

HIDDEN_DIM: int = CONFIG["encoder"]["hidden_dim"]
NUM_BLOCKS: int = CONFIG["encoder"]["num_blocks"]
TRIPLET_DIM: int = CONFIG["encoder"]["triplet_dim"]
BASIS_DIM: int = CONFIG["encoder"]["basis_dim"]
OUTPUT_DIM: int = CONFIG["encoder"]["output_dim"]
NUM_SPHERICAL: int = CONFIG["encoder"]["num_spherical"]
NUM_RADIAL: int = CONFIG["encoder"]["num_radial"]
RADIUS: float = CONFIG["encoder"]["radius"]
ENVELOPE_EXPONENT: int = CONFIG["encoder"]["envelope_exponent"]
NUM_BEFORE_SKIP: int = CONFIG["encoder"]["num_before_skip"]
NUM_AFTER_SKIP: int = CONFIG["encoder"]["num_after_skip"]
NUM_OUTPUT_LAYERS: int = CONFIG["encoder"]["num_output_layers"]
READOUT: Reduce = CONFIG["encoder"]["readout"]
NUM_ELEMENTS: int = CONFIG["encoder"]["num_elements"]


def envelope(scaled: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n 1"]:
    """Envelope of the bases of a distance over the radius: a smooth cutoff over it.

    It grows as the inverse of the distance. It is zero after the radius, where
    the one of cdvae is not: its graphs have no bond that long.
    """
    cutoff = polynomial_cutoff(scaled, 1.0, p=ENVELOPE_EXPONENT + 1)

    return (cutoff / scaled).unsqueeze(-1)


class BesselBasisLayer(nn.Module):
    """Radial basis of the length of a bond, with learned frequencies."""

    def __init__(self):
        super().__init__()
        self.freq = nn.Parameter(math.pi * torch.arange(1.0, NUM_RADIAL + 1.0))

    def forward(
        self, distance: Float[torch.Tensor, "n_edges"]
    ) -> Float[torch.Tensor, "n_edges num_radial"]:
        scaled = distance / RADIUS

        return envelope(scaled) * torch.sin(self.freq * scaled.unsqueeze(-1))


class SphericalBasisLayer(nn.Module):
    """Basis of a pair of bonds: Bessel functions times harmonics of their angle.

    The spherical Bessel functions are the ones of SphericalBesselBasis for a
    radius of 1: its normalization is undone.
    """

    def __init__(self):
        super().__init__()
        self.basis = SphericalBesselBasis(NUM_SPHERICAL, NUM_RADIAL, RADIUS)

    def radial(
        self, distance: Float[torch.Tensor, "n_edges"]
    ) -> Float[torch.Tensor, "n_edges num_spherical*num_radial"]:
        """Radial part of the basis, for the length of each bond."""
        return self.basis(distance) * RADIUS**1.5 * envelope(distance / RADIUS)

    def forward(
        self,
        radial: Float[torch.Tensor, "n_pairs num_spherical*num_radial"],
        cosine: Float[torch.Tensor, "n_pairs"],
    ) -> Float[torch.Tensor, "n_pairs num_spherical*num_radial"]:
        # the harmonic of degree l multiplies the num_radial functions of degree l
        harmonics = legendre(NUM_SPHERICAL, cosine).repeat_interleave(NUM_RADIAL, dim=1)

        return radial * harmonics


class EmbeddingBlock(nn.Module):
    """Initial features of the bonds, from their two atoms and their length."""

    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
        # indexed by the atomic number itself
        self.emb = nn.Embedding(NUM_ELEMENTS, HIDDEN_DIM)
        self.lin_rbf = nn.Linear(NUM_RADIAL, HIDDEN_DIM)
        self.lin = nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM)

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        rbf: Float[torch.Tensor, "n_edges num_radial"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_edges d"]:
        atoms = self.emb(z)
        rbf = self.act(self.lin_rbf(rbf))

        return self.act(
            self.lin(
                torch.cat(
                    [graph.unpool_senders(atoms), graph.unpool_receivers(atoms), rbf],
                    dim=-1,
                )
            )
        )


class ResidualLayer(nn.Module):
    """Two layers added to their input."""

    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
        self.lin1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.lin2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(nn.Module):
    """Update of a bond from the bonds which follow it."""

    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

        self.lin_rbf1 = nn.Linear(NUM_RADIAL, BASIS_DIM, bias=False)
        self.lin_rbf2 = nn.Linear(BASIS_DIM, HIDDEN_DIM, bias=False)
        self.lin_sbf1 = nn.Linear(NUM_SPHERICAL * NUM_RADIAL, BASIS_DIM, bias=False)
        self.lin_sbf2 = nn.Linear(BASIS_DIM, TRIPLET_DIM, bias=False)

        self.lin_kj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.lin_ji = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.lin_down = nn.Linear(HIDDEN_DIM, TRIPLET_DIM, bias=False)
        self.lin_up = nn.Linear(TRIPLET_DIM, HIDDEN_DIM, bias=False)

        self.layers_before_skip = nn.ModuleList(
            ResidualLayer() for _ in range(NUM_BEFORE_SKIP)
        )
        self.lin = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.layers_after_skip = nn.ModuleList(
            ResidualLayer() for _ in range(NUM_AFTER_SKIP)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "n_edges d"],
        rbf: Float[torch.Tensor, "n_edges num_radial"],
        sbf: Float[torch.Tensor, "n_pairs num_spherical*num_radial"],
        pairs: EdgeTuples,
    ) -> Float[torch.Tensor, "n_edges d"]:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        x_kj = x_kj * self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = self.act(self.lin_down(x_kj))

        # a pair i → j, j → k sends to its first bond the features of the second
        x_kj = pairs.unpool_edges(x_kj, position=1) * self.lin_sbf2(self.lin_sbf1(sbf))
        x_kj = pairs.aggregate_to_edges(x_kj, position=0)
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(nn.Module):
    """Output of each atom from the bonds leaving it."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.act = nn.SiLU()

        self.lin_rbf = nn.Linear(NUM_RADIAL, HIDDEN_DIM, bias=False)
        self.lin_up = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=True)
        self.lins = nn.ModuleList(
            nn.Linear(OUTPUT_DIM, OUTPUT_DIM) for _ in range(NUM_OUTPUT_LAYERS)
        )
        self.lin = nn.Linear(OUTPUT_DIM, out_channels, bias=False)

    def forward(
        self,
        x: Float[torch.Tensor, "n_edges d"],
        rbf: Float[torch.Tensor, "n_edges num_radial"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_atoms out_channels"]:
        x = graph.aggregate_edges_to_nodes(self.lin_rbf(rbf) * x, to="sender")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))

        return self.lin(x)


class DimeNetPlusPlus(nn.Module):
    """DimeNet++ computing a vector for each structure.

    Args:
        out_channels: Size of the vector of a structure.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.rbf = BesselBasisLayer()
        self.sbf = SphericalBasisLayer()
        self.emb = EmbeddingBlock()
        self.output_blocks = nn.ModuleList(
            OutputPPBlock(out_channels) for _ in range(NUM_BLOCKS + 1)
        )
        self.interaction_blocks = nn.ModuleList(
            InteractionPPBlock() for _ in range(NUM_BLOCKS)
        )

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        graph: Graph,
        pairs: EdgeTuples,
        distance: Float[torch.Tensor, "n_edges"],
        cosine: Float[torch.Tensor, "n_pairs"],
    ) -> Float[torch.Tensor, "n_struct out_channels"]:
        """Vector of each structure.

        Args:
            z: Atomic number of each atom, below the number of rows of the
                embedding.
            graph: Graph of the bonds.
            pairs: Pairs of bonds i → j, j → k of the graph.
            distance: Length of each bond in Å.
            cosine: Cosine of the angle between the two bonds of each pair, as
                vectors: 1 when k continues the bond i → j.
        """
        rbf = self.rbf(distance)
        # the basis of a pair uses the length of its second bond
        sbf = self.sbf(
            pairs.unpool_edges(self.sbf.radial(distance), position=1), cosine
        )

        x = self.emb(z, rbf, graph)
        output = self.output_blocks[0](x, rbf, graph)

        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, pairs)
            output = output + output_block(x, rbf, graph)

        return graph.aggregate_nodes_to_structures(output, READOUT)
