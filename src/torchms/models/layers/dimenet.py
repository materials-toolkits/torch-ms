# Derived from cdvae (https://github.com/txie-93/cdvae), Copyright (c) 2021
# Tian Xie, Xiang Fu, MIT license, whose DimeNet++ is adapted from the Open
# Catalyst Project, Copyright (c) Facebook, Inc. and its affiliates, and from
# PyTorch Geometric, Copyright (c) 2021 Matthias Fey, Jiaxuan You, both under
# the MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the DimeNet++ of cdvae
# (cdvae/pl_modules/gnn.py and the layers of torch_geometric 2.0.1 it uses),
# for inference only. The graph and its pairs of bonds are read from torchms
# Structures instead of a batch of PyTorch Geometric, the bases are computed
# with torch instead of sympy, and the option building a radius graph is
# removed. Names of modules and parameters follow cdvae to load its weights.
"""DimeNet++, a network on the bonds of a graph and on their angles.

Features live on the bonds of a graph of the atoms. A block updates a bond
i → j from the bonds j → k leaving its receiver towards another atom than i,
in a basis of spherical Bessel functions of the length of j → k and of
spherical harmonics of the angle between the two bonds. After each block the
bonds are summed on their atom, and the atoms reduced over the structure.

DimeNet++ is published in "Fast and Uncertainty-Aware Directional Message
Passing for Non-Equilibrium Molecules", Gasteiger, Giri, Margraf and Günnemann
(arXiv:2011.14115), after DimeNet, "Directional Message Passing for Molecular
Graphs", Gasteiger, Groß and Günnemann (arXiv:2003.03123).

It is rewritten here from the code of cdvae, which uses it as the encoder of
CDVAE, see torchms.models.cdvae.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.sparse import Reduce
from torchms.models.layers.basis import (
    SphericalBesselBasis,
    legendre,
    polynomial_cutoff,
)


@dataclass(frozen=True)
class Config:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of a DimeNet++.

    Args:
        out_channels: Size of the vector of a structure.
        hidden_dim: Features of the bonds.
        num_blocks: Number of interaction blocks.
        triplet_dim: Features of the pairs of bonds.
        basis_dim: Projection of the bases.
        output_dim: Hidden layers of the outputs.
        num_spherical: Degrees of the basis of the angles.
        num_radial: Radial functions.
        radius: Radius of the bases in Å: a longer bond contributes nothing.
        envelope_exponent: Exponent of the smooth cutoff of the bases.
        num_before_skip: Residual layers of a block before its skip connection.
        num_after_skip: Residual layers of a block after its skip connection.
        num_output_layers: Hidden layers of an output.
        readout: Reduction of the atoms of a structure, "mean" or "sum".
        num_elements: Rows of the embedding, indexed by atomic number.
    """

    out_channels: int
    hidden_dim: int
    num_blocks: int
    triplet_dim: int
    basis_dim: int
    output_dim: int
    num_spherical: int
    num_radial: int
    radius: float
    envelope_exponent: int
    num_before_skip: int
    num_after_skip: int
    num_output_layers: int
    readout: Reduce
    num_elements: int


def envelope(
    scaled: Float[torch.Tensor, "n"], exponent: int
) -> Float[torch.Tensor, "n 1"]:
    """Envelope of the bases of a distance over the radius: a smooth cutoff over it.

    It grows as the inverse of the distance. It is zero after the radius, where
    the one of cdvae is not: its graphs have no bond that long.
    """
    cutoff = polynomial_cutoff(scaled, 1.0, p=exponent + 1)

    return (cutoff / scaled).unsqueeze(-1)


class BesselBasisLayer(nn.Module):
    """Radial basis of the length of a bond, with learned frequencies."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.freq = nn.Parameter(math.pi * torch.arange(1.0, config.num_radial + 1.0))

    def forward(
        self, distance: Float[torch.Tensor, "n_edges"]
    ) -> Float[torch.Tensor, "n_edges num_radial"]:
        scaled = distance / self.config.radius

        return envelope(scaled, self.config.envelope_exponent) * torch.sin(
            self.freq * scaled.unsqueeze(-1)
        )


class SphericalBasisLayer(nn.Module):
    """Basis of a pair of bonds: Bessel functions times harmonics of their angle.

    The spherical Bessel functions are the ones of SphericalBesselBasis for a
    radius of 1: its normalization is undone.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.basis = SphericalBesselBasis(
            config.num_spherical, config.num_radial, config.radius
        )

    def radial(
        self, distance: Float[torch.Tensor, "n_edges"]
    ) -> Float[torch.Tensor, "n_edges num_spherical*num_radial"]:
        """Radial part of the basis, for the length of each bond."""
        config = self.config
        cutoff = envelope(distance / config.radius, config.envelope_exponent)

        return self.basis(distance) * config.radius**1.5 * cutoff

    def forward(
        self,
        radial: Float[torch.Tensor, "n_pairs num_spherical*num_radial"],
        cosine: Float[torch.Tensor, "n_pairs"],
    ) -> Float[torch.Tensor, "n_pairs num_spherical*num_radial"]:
        # the harmonic of degree l multiplies the num_radial functions of degree l
        harmonics = legendre(self.config.num_spherical, cosine).repeat_interleave(
            self.config.num_radial, dim=1
        )

        return radial * harmonics


class EmbeddingBlock(nn.Module):
    """Initial features of the bonds, from their two atoms and their length."""

    def __init__(self, config: Config):
        super().__init__()
        self.act = nn.SiLU()
        # indexed by the atomic number itself
        self.emb = nn.Embedding(config.num_elements, config.hidden_dim)
        self.lin_rbf = nn.Linear(config.num_radial, config.hidden_dim)
        self.lin = nn.Linear(3 * config.hidden_dim, config.hidden_dim)

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

    def __init__(self, config: Config):
        super().__init__()
        self.act = nn.SiLU()
        self.lin1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(nn.Module):
    """Update of a bond from the bonds which follow it."""

    def __init__(self, config: Config):
        super().__init__()
        self.act = nn.SiLU()

        self.lin_rbf1 = nn.Linear(config.num_radial, config.basis_dim, bias=False)
        self.lin_rbf2 = nn.Linear(config.basis_dim, config.hidden_dim, bias=False)
        self.lin_sbf1 = nn.Linear(
            config.num_spherical * config.num_radial, config.basis_dim, bias=False
        )
        self.lin_sbf2 = nn.Linear(config.basis_dim, config.triplet_dim, bias=False)

        self.lin_kj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.lin_ji = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.lin_down = nn.Linear(config.hidden_dim, config.triplet_dim, bias=False)
        self.lin_up = nn.Linear(config.triplet_dim, config.hidden_dim, bias=False)

        self.layers_before_skip = nn.ModuleList(
            ResidualLayer(config) for _ in range(config.num_before_skip)
        )
        self.lin = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layers_after_skip = nn.ModuleList(
            ResidualLayer(config) for _ in range(config.num_after_skip)
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

    def __init__(self, config: Config):
        super().__init__()
        self.act = nn.SiLU()

        self.lin_rbf = nn.Linear(config.num_radial, config.hidden_dim, bias=False)
        self.lin_up = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.lins = nn.ModuleList(
            nn.Linear(config.output_dim, config.output_dim)
            for _ in range(config.num_output_layers)
        )
        self.lin = nn.Linear(config.output_dim, config.out_channels, bias=False)

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
        config: Hyperparameters of the network.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rbf = BesselBasisLayer(config)
        self.sbf = SphericalBasisLayer(config)
        self.emb = EmbeddingBlock(config)
        self.output_blocks = nn.ModuleList(
            OutputPPBlock(config) for _ in range(config.num_blocks + 1)
        )
        self.interaction_blocks = nn.ModuleList(
            InteractionPPBlock(config) for _ in range(config.num_blocks)
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

        return graph.aggregate_nodes_to_structures(output, self.config.readout)
