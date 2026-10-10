# Derived from cdvae (https://github.com/txie-93/cdvae), Copyright (c) 2021
# Tian Xie, Xiang Fu, MIT license, whose GemNet-T is adapted from the Open
# Catalyst Project, Copyright (c) Facebook, Inc. and its affiliates, MIT
# license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the GemNet-T of cdvae
# (cdvae/pl_modules/gemnet), for inference only. The graph, its triplets and
# its pairs of reverse edges are read from torchms Structures instead of being
# built by the network, the sums over the triplets are aggregations instead of
# padded matrices, the harmonics are computed with torch instead of sympy, and
# the energy read-out, the fitting of the scaling factors and the other bases
# are removed. Names of modules and parameters follow cdvae to load its
# weights.
"""GemNet-T, a network on the edges of a symmetric graph and on their triplets.

Features live on the atoms and on the edges of a symmetric graph. A block
updates an edge towards an atom from the other edges of this atom, in a basis
of gaussians of its length and of spherical harmonics of their angle, then the
atoms from their edges and the edges from their atoms. Each edge gives a
scalar: their sum along the directions of the edges of an atom is a vector.
The network also reads a vector of each structure, given to all its atoms.

GemNet is published in "GemNet: Universal Directional Graph Neural Networks
for Molecules", Gasteiger, Becker and Günnemann (arXiv:2106.08903). GemNet-T
is its variant with triplets of atoms.

It is rewritten here from the code of cdvae, which uses it as the decoder of
CDVAE, see torchms.models.cdvae: the vector of an atom is the direction it
should move in.
"""

# attributes are named as in cdvae, to load its weights
# pylint: disable=invalid-name

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.graph.graph import EdgeTuples, Graph
from torchms.models.layers.basis import GaussianBasis, legendre, polynomial_cutoff

INV_SQRT_2 = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class GemNetConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of a GemNet-T.

    Args:
        latent_dim: Size of the vector of a structure given to its atoms.
        num_elements: Rows of the embedding of the elements, from hydrogen.
        radius: Radius of the graph and of the bases, in Å.
        hidden_dim: Features of the atoms and of the edges.
        num_blocks: Number of interaction blocks.
        num_spherical: Degrees of the basis of the angles.
        num_radial: Gaussians of the lengths of the edges.
        triplet_dim: Features of the edges in a triplet.
        radial_dim: Projection of the radial basis.
        circular_dim: Projection of the basis of the angles.
        bilinear_dim: Features of the edges after the triplets.
        num_before_skip: Residual layers of a block before its skip connection.
        num_after_skip: Residual layers of a block after its skip connection.
        num_concat: Residual layers after the update of the edges from the atoms.
        num_atom_layers: Residual layers of the updates of the atoms.
        envelope_exponent: Exponent of the smooth cutoff of the radial basis.
        activation_scale: The SiLU is divided by it.
    """

    latent_dim: int
    num_elements: int
    radius: float
    hidden_dim: int
    num_blocks: int
    num_spherical: int
    num_radial: int
    triplet_dim: int
    radial_dim: int
    circular_dim: int
    bilinear_dim: int
    num_before_skip: int
    num_after_skip: int
    num_concat: int
    num_atom_layers: int
    envelope_exponent: int
    activation_scale: float


def reverse_features(
    src: Float[torch.Tensor, "n_edges ..."], bonds: EdgeTuples
) -> Float[torch.Tensor, "n_edges ..."]:
    """Per-edge features of the reverse of each edge.

    Args:
        src: Per-edge features, shape (n_edges, *).
        bonds: Pairs of an edge and its reverse, every edge in one pair.
    """
    return bonds.aggregate_to_edges(
        bonds.unpool_edges(src, position=1), position=0
    ) + bonds.aggregate_to_edges(bonds.unpool_edges(src, position=0), position=1)


class ScaledSiLU(nn.Module):
    """SiLU divided by a constant, which keeps the variance of its input."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return nn.functional.silu(x) / self.scale


class Dense(nn.Module):
    """Linear layer without bias, followed by a scaled SiLU if its scale is given."""

    def __init__(
        self, in_features: int, out_features: int, activation: float | None = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self._activation = (
            nn.Identity() if activation is None else ScaledSiLU(activation)
        )

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self._activation(self.linear(x))


class ResidualLayer(nn.Module):
    """Two layers added to their input, the sum divided by sqrt(2)."""

    def __init__(self, units: int, activation: float):
        super().__init__()
        self.dense_mlp = nn.Sequential(
            Dense(units, units, activation), Dense(units, units, activation)
        )

    def forward(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        return (x + self.dense_mlp(x)) * INV_SQRT_2


class ScalingFactor(nn.Module):
    """Constant fitted by cdvae to keep the variance of what it multiplies."""

    scale_factor: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("scale_factor", torch.tensor(1.0))

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return x * self.scale_factor


def residual_mlp(units_in: int, units: int, config: GemNetConfig) -> nn.ModuleList:
    """A layer then the residual layers of the updates of the atoms."""
    scale = config.activation_scale

    return nn.ModuleList(
        [Dense(units_in, units, scale)]
        + [ResidualLayer(units, scale) for _ in range(config.num_atom_layers)]
    )


class RadialBasis(nn.Module):
    """Gaussians of the length of an edge over the radius, with a smooth cutoff."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.config = config
        # the width of a gaussian is the distance between two centers
        self.rbf = GaussianBasis(
            0.0, 1.0, config.num_radial, gamma=0.5 * (config.num_radial - 1) ** 2
        )

    def forward(
        self, distance: Float[torch.Tensor, "n_edges"]
    ) -> Float[torch.Tensor, "n_edges num_radial"]:
        scaled = distance / self.config.radius
        cutoff = polynomial_cutoff(scaled, 1.0, p=self.config.envelope_exponent)

        return cutoff.unsqueeze(-1) * self.rbf(scaled)


class CircularBasisLayer(nn.Module):
    """Bases of a triplet: gaussians of a length and harmonics of the angle."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.config = config
        self.radial_basis = RadialBasis(config)

    def forward(
        self,
        distance: Float[torch.Tensor, "n_edges"],
        cosine: Float[torch.Tensor, "n_triplets"],
    ) -> tuple[
        Float[torch.Tensor, "n_edges num_radial"],
        Float[torch.Tensor, "n_triplets num_spherical"],
    ]:
        return self.radial_basis(distance), legendre(self.config.num_spherical, cosine)


class EfficientInteractionDownProjection(nn.Module):
    """Projection of the radial basis of an edge, one per degree of the harmonics."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_spherical, config.num_radial, config.circular_dim)
        )
        nn.init.orthogonal_(self.weight)

    def forward(
        self, rbf: Float[torch.Tensor, "n_edges num_radial"]
    ) -> Float[torch.Tensor, "n_edges circular_dim num_spherical"]:
        return torch.einsum("er,sri->eis", rbf, self.weight)


class EfficientInteractionBilinear(nn.Module):
    """Sum over the triplets of an edge, then bilinear layer with its radial basis."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.triplet_dim, config.circular_dim, config.bilinear_dim)
        )
        nn.init.orthogonal_(self.weight)

    def forward(
        self,
        rbf: Float[torch.Tensor, "n_edges circular_dim num_spherical"],
        cbf: Float[torch.Tensor, "n_triplets num_spherical"],
        m: Float[torch.Tensor, "n_triplets triplet_dim"],
        angles: EdgeTuples,
    ) -> Float[torch.Tensor, "n_edges bilinear_dim"]:
        # an edge sums, over the other edges of its atom, their features times
        # the harmonics of the angle they form with it
        summed = angles.aggregate_to_edges(
            cbf.unsqueeze(2) * m.unsqueeze(1), position=0
        )

        return torch.einsum("eji,ijo->eo", torch.matmul(rbf, summed), self.weight)


class AtomEmbedding(nn.Module):
    """Initial features of the atoms, from their element."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.num_elements, config.hidden_dim)

    def forward(
        self, element: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms d"]:
        return self.embeddings(element)


class EdgeEmbedding(nn.Module):
    """Features of an edge from the ones of its two atoms and its own."""

    def __init__(self, edge_features: int, config: GemNetConfig):
        super().__init__()
        self.dense = Dense(
            2 * config.hidden_dim + edge_features,
            config.hidden_dim,
            activation=config.activation_scale,
        )

    def forward(
        self,
        h: Float[torch.Tensor, "n_atoms d"],
        m: Float[torch.Tensor, "n_edges f"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_edges d"]:
        # an edge of cdvae goes to the atom which lists it: here its sender
        return self.dense(
            torch.cat([graph.unpool_receivers(h), graph.unpool_senders(h), m], dim=-1)
        )


class AtomUpdateBlock(nn.Module):
    """Features of an atom from the edges it lists."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.dense_rbf = Dense(config.radial_dim, config.hidden_dim)
        self.scale_sum = ScalingFactor()
        self.layers = residual_mlp(config.hidden_dim, config.hidden_dim, config)

    def forward(
        self,
        m: Float[torch.Tensor, "n_edges d"],
        rbf: Float[torch.Tensor, "n_edges radial_dim"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_atoms d"]:
        x = graph.aggregate_edges_to_nodes(m * self.dense_rbf(rbf), to="sender")
        x = self.scale_sum(x)
        for layer in self.layers:
            x = layer(x)

        return x


class OutputBlock(nn.Module):
    """Scalar of each edge, the length of the move of its atom along it.

    The read-out of an energy per atom of GemNet is in the weights and is not
    computed: cdvae does not use it.
    """

    def __init__(self, config: GemNetConfig):
        super().__init__()
        # read-out of the energy, as an update of the atoms
        self.dense_rbf = Dense(config.radial_dim, config.hidden_dim)
        self.scale_sum = ScalingFactor()
        self.layers = residual_mlp(config.hidden_dim, config.hidden_dim, config)
        self.seq_energy = self.layers
        self.out_energy = Dense(config.hidden_dim, 1)

        self.scale_rbf_F = ScalingFactor()
        self.seq_forces = residual_mlp(config.hidden_dim, config.hidden_dim, config)
        self.out_forces = Dense(config.hidden_dim, 1)
        self.dense_rbf_F = Dense(config.radial_dim, config.hidden_dim)

    def forward(
        self,
        m: Float[torch.Tensor, "n_edges d"],
        rbf: Float[torch.Tensor, "n_edges radial_dim"],
    ) -> Float[torch.Tensor, "n_edges"]:
        x = m
        for layer in self.seq_forces:
            x = layer(x)
        x = self.scale_rbf_F(x * self.dense_rbf_F(rbf))

        return self.out_forces(x).squeeze(-1)


class TripletInteraction(nn.Module):
    """Update of an edge from the other edges of its atom, and of its reverse."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.dense_ba = Dense(
            config.hidden_dim, config.hidden_dim, activation=config.activation_scale
        )
        self.mlp_rbf = Dense(config.radial_dim, config.hidden_dim)
        self.scale_rbf = ScalingFactor()
        self.mlp_cbf = EfficientInteractionBilinear(config)
        self.scale_cbf_sum = ScalingFactor()
        self.down_projection = Dense(
            config.hidden_dim, config.triplet_dim, activation=config.activation_scale
        )
        self.up_projection_ca = Dense(
            config.bilinear_dim, config.hidden_dim, activation=config.activation_scale
        )
        self.up_projection_ac = Dense(
            config.bilinear_dim, config.hidden_dim, activation=config.activation_scale
        )

    def forward(
        self,
        m: Float[torch.Tensor, "n_edges d"],
        rbf3: Float[torch.Tensor, "n_edges radial_dim"],
        rbf_cbf3: Float[torch.Tensor, "n_edges circular_dim num_spherical"],
        cbf3: Float[torch.Tensor, "n_triplets num_spherical"],
        angles: EdgeTuples,
        bonds: EdgeTuples,
    ) -> Float[torch.Tensor, "n_edges d"]:
        x_ba = self.dense_ba(m)
        x_ba = self.scale_rbf(x_ba * self.mlp_rbf(rbf3))
        x_ba = self.down_projection(x_ba)

        # a triplet gives the features of its second edge to its first one
        x = self.mlp_cbf(rbf_cbf3, cbf3, angles.unpool_edges(x_ba, position=1), angles)
        x = self.scale_cbf_sum(x)

        x_ca = self.up_projection_ca(x)
        x_ac = reverse_features(self.up_projection_ac(x), bonds)

        return (x_ca + x_ac) * INV_SQRT_2


class InteractionBlockTripletsOnly(nn.Module):
    """Update of the edges from the triplets, then of the atoms and of the edges."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.dense_ca = Dense(
            config.hidden_dim, config.hidden_dim, activation=config.activation_scale
        )
        self.trip_interaction = TripletInteraction(config)
        self.layers_before_skip = nn.ModuleList(
            ResidualLayer(config.hidden_dim, config.activation_scale)
            for _ in range(config.num_before_skip)
        )
        self.layers_after_skip = nn.ModuleList(
            ResidualLayer(config.hidden_dim, config.activation_scale)
            for _ in range(config.num_after_skip)
        )
        self.atom_update = AtomUpdateBlock(config)
        self.concat_layer = EdgeEmbedding(config.hidden_dim, config)
        self.residual_m = nn.ModuleList(
            ResidualLayer(config.hidden_dim, config.activation_scale)
            for _ in range(config.num_concat)
        )

    def forward(
        self,
        h: Float[torch.Tensor, "n_atoms d"],
        m: Float[torch.Tensor, "n_edges d"],
        rbf3: Float[torch.Tensor, "n_edges radial_dim"],
        rbf_cbf3: Float[torch.Tensor, "n_edges circular_dim num_spherical"],
        cbf3: Float[torch.Tensor, "n_triplets num_spherical"],
        rbf_h: Float[torch.Tensor, "n_edges radial_dim"],
        graph: Graph,
        angles: EdgeTuples,
        bonds: EdgeTuples,
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        x = self.dense_ca(m) + self.trip_interaction(
            m, rbf3, rbf_cbf3, cbf3, angles, bonds
        )
        x = x * INV_SQRT_2
        for layer in self.layers_before_skip:
            x = layer(x)

        m = (m + x) * INV_SQRT_2
        for layer in self.layers_after_skip:
            m = layer(m)

        h = (h + self.atom_update(m, rbf_h, graph)) * INV_SQRT_2

        m2 = self.concat_layer(h, m, graph)
        for layer in self.residual_m:
            m2 = layer(m2)

        return h, (m + m2) * INV_SQRT_2


class GemNetT(nn.Module):
    """GemNet-T computing features and a vector for each atom."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.radial_basis = RadialBasis(config)
        self.cbf_basis3 = CircularBasisLayer(config)

        # projections of the bases, shared by the blocks
        self.mlp_rbf3 = Dense(config.num_radial, config.radial_dim)
        self.mlp_cbf3 = EfficientInteractionDownProjection(config)
        self.mlp_rbf_h = Dense(config.num_radial, config.radial_dim)
        self.mlp_rbf_out = Dense(config.num_radial, config.radial_dim)

        self.atom_emb = AtomEmbedding(config)
        self.atom_latent_emb = nn.Linear(
            config.hidden_dim + config.latent_dim, config.hidden_dim
        )
        self.edge_emb = EdgeEmbedding(config.num_radial, config)

        self.out_blocks = nn.ModuleList(
            OutputBlock(config) for _ in range(config.num_blocks + 1)
        )
        self.int_blocks = nn.ModuleList(
            InteractionBlockTripletsOnly(config) for _ in range(config.num_blocks)
        )

    def forward(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        element: Int[torch.Tensor, "n_atoms"],
        graph: Graph,
        angles: EdgeTuples,
        bonds: EdgeTuples,
        vectors: Float[torch.Tensor, "n_edges 3"],
        cosine: Float[torch.Tensor, "n_triplets"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_atoms 3"]]:
        """Features and vector of each atom.

        Args:
            latent: Latent vector of each structure.
            element: Index of the element of each atom, from 0 for hydrogen.
            graph: Symmetric graph of the atoms.
            angles: Triplets of the graph, the pairs of edges of an atom.
            bonds: Pairs of an edge of the graph and its reverse.
            vectors: Vector of each edge in Å, from its sender.
            cosine: Cosine of the angle of each triplet.
        """
        distance = vectors.norm(dim=-1)

        rbf = self.radial_basis(distance)
        rbf_cbf3, cbf3 = self.cbf_basis3(distance, cosine)

        h = self.atom_emb(element)
        h = self.atom_latent_emb(torch.cat([h, graph.unpool_structures(latent)], dim=1))
        m = self.edge_emb(h, rbf, graph)

        rbf3 = self.mlp_rbf3(rbf)
        rbf_cbf3 = self.mlp_cbf3(rbf_cbf3)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        scalar = self.out_blocks[0](m, rbf_out)
        for int_block, out_block in zip(self.int_blocks, self.out_blocks[1:]):
            h, m = int_block(h, m, rbf3, rbf_cbf3, cbf3, rbf_h, graph, angles, bonds)
            scalar = scalar + out_block(m, rbf_out)

        # an atom is moved along its edges, towards itself for a positive scalar
        towards = -vectors / distance.unsqueeze(-1)
        move = graph.aggregate_edges_to_nodes(
            scalar.unsqueeze(-1) * towards, to="sender"
        )

        return h, move
