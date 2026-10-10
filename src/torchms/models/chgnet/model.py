# Derived from chgnet (https://github.com/CederGroupHub/chgnet), Copyright (c)
# 2023, The Regents of the University of California, through Lawrence Berkeley
# National Laboratory, modified BSD license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the energy prediction of CHGNet 0.3.0, for
# inference only. Magnetic moments, forces, stress, training, datasets and the
# conversion from pymatgen are removed. Graphs are built by torchms.graph
# instead of the graph converter of chgnet, and batches are torchms
# Structures. Names of modules and parameters follow chgnet to load its
# pretrained weights.
"""Energy prediction of the pretrained CHGNet model.

The network has features on atoms, on bonds and on angles. Atoms closer than
6 Å are bonded; the angles are the pairs of bonds shorter than 3 Å around an
atom. A bond and its reverse share their features. Each of the first three
layers updates the atoms from their bonds, the bonds from their angles, then
the angles; a last layer updates the atoms. The energy is a sum of an energy
per atom and of a reference energy per element.

Weights are the ones published with chgnet, see config.yaml.
"""

# attributes are named as in chgnet, to load its weights
# pylint: disable=invalid-name

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.models.chgnet.config import Config, NetworkConfig
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddBonds,
    AddRadiusGraph,
    AddTriplets,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.layers.basis import polynomial_cutoff


@dataclass
class UndirectedBonds:
    """Bonds of a graph: an edge and its reverse are one bond.

    The features of the bonds have one row per bond, shared by its two edges.

    Attributes:
        first: Index of one edge of each bond, shape (n_bonds,).
        edge_bond: Index of the bond of each edge, shape (n_edges,).
        num_bonds: Number of bonds per structure, shape (n_struct,).
    """

    first: Int[torch.Tensor, "n_bonds"]
    edge_bond: Int[torch.Tensor, "n_edges"]
    num_bonds: Int[torch.Tensor, "n_struct"]

    @classmethod
    def of(cls, bonds: EdgeTuples) -> UndirectedBonds:
        """Bonds from the pairs of an edge and its reverse attached by AddBonds."""
        n_bonds = bonds.index.shape[1]
        number = torch.arange(n_bonds, device=bonds.device())

        # every edge is in one bond, as its first or as its second edge
        edge_bond = number.new_empty(int(bonds.num_edges.sum()))
        edge_bond[bonds.index] = number

        return cls(bonds.index[0], edge_bond, bonds.num_tuples)

    def from_edges(
        self, src: Float[torch.Tensor, "n_edges ..."]
    ) -> Float[torch.Tensor, "n_bonds ..."]:
        """Per-bond values from per-edge ones which are the same on both edges."""
        return src[self.first]

    def unpool_bonds(
        self, src: Float[torch.Tensor, "n_bonds ..."]
    ) -> Float[torch.Tensor, "n_edges ..."]:
        """Per-bond features on the edges: the ones of the bond of each edge."""
        return src[self.edge_bond]

    def on_bonds(self, tuples: EdgeTuples) -> EdgeTuples:
        """The same tuples of edges, as tuples of the bonds of these edges."""
        return replace(
            tuples, index=self.edge_bond[tuples.index], num_edges=self.num_bonds
        )


class MLP(nn.Module):
    """Fully connected network with SiLU activations between its layers.

    The layer before the last one is the place of the dropout of chgnet: the
    indices of the layers are the ones of its weights.
    """

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int, bias=True
    ):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        layers: list[nn.Module] = []

        for size_in, size_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(size_in, size_out, bias=bias), nn.SiLU()]

        layers += [nn.Identity(), nn.Linear(sizes[-1], output_size, bias=bias)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self.layers(x)


class GatedMLP(nn.Module):
    """Product of a network and of a gate in [0, 1], both normalized."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.mlp_core = MLP(input_size, hidden_sizes, output_size)
        self.mlp_gate = MLP(input_size, hidden_sizes, output_size)
        self.bn1 = nn.LayerNorm(output_size)
        self.bn2 = nn.LayerNorm(output_size)

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        core = nn.functional.silu(self.bn1(self.mlp_core(x)))
        gate = torch.sigmoid(self.bn2(self.mlp_gate(x)))

        return core * gate


class RadialBessel(nn.Module):
    """Radial basis sqrt(2 / r_max) sin(f r / r_max) / r with a smooth cutoff.

    The frequencies f are parameters, n pi for n = 1..num_radial before training.
    """

    def __init__(self, radius: float, num_radial: int, cutoff_power: int):
        super().__init__()
        self.radius = radius
        self.cutoff_power = cutoff_power
        self.frequencies = nn.Parameter(
            math.pi * torch.arange(1.0, num_radial + 1.0), requires_grad=False
        )

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_radial"]:
        distance = distance.unsqueeze(-1)
        basis = (
            math.sqrt(2.0 / self.radius)
            * torch.sin(self.frequencies * distance / self.radius)
            / distance
        )

        return polynomial_cutoff(distance, self.radius, p=self.cutoff_power) * basis


class BondEncoder(nn.Module):
    """Radial bases of a bond for the cutoffs of the bonds and of the angles."""

    def __init__(self, config: Config):
        super().__init__()
        network = config.model
        self.rbf_expansion_ag = RadialBessel(
            config.graph.radius, network.num_radial, network.cutoff_power
        )
        self.rbf_expansion_bg = RadialBessel(
            config.graph.bond_radius, network.num_radial, network.cutoff_power
        )


class Fourier(nn.Module):
    """Fourier basis of an angle: a constant, then sines and cosines of f angle.

    The frequencies f are parameters, 1..order before training.
    """

    def __init__(self, num_angular: int):
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.arange(1.0, (num_angular - 1) // 2 + 1.0), requires_grad=False
        )

    def forward(
        self, angle: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_angular"]:
        phase = angle.unsqueeze(-1) * self.frequencies
        constant = torch.full_like(angle, math.sqrt(0.5)).unsqueeze(-1)

        return torch.cat(
            [constant, torch.sin(phase), torch.cos(phase)], dim=-1
        ) / math.sqrt(math.pi)


class AngleEncoder(nn.Module):
    """Fourier basis of the angle between two bonds."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.cosine_shrink = network.cosine_shrink
        self.fourier_expansion = Fourier(network.num_angular)

    def forward(
        self, cosine: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_angular"]:
        # as chgnet, the cosine is shrunk to keep the angle differentiable
        return self.fourier_expansion(torch.acos(cosine * (1.0 - self.cosine_shrink)))


class AtomEmbedding(nn.Module):
    """Learned features of each element, indexed by atomic number minus one."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.embedding = nn.Embedding(network.num_elements, network.latent_dim)

    def forward(
        self, element: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms d"]:
        return self.embedding(element)


class AtomConv(nn.Module):
    """Update of the atoms from their bonds and their neighbours."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.twoBody_atom = GatedMLP(3 * latent_dim, [latent_dim], latent_dim)
        self.mlp_out = MLP(latent_dim, [], latent_dim, bias=False)

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        weights: Float[torch.Tensor, "n_bonds d"],
        graph: Graph,
        undirected: UndirectedBonds,
    ) -> Float[torch.Tensor, "n_atoms d"]:
        messages = self.twoBody_atom(
            torch.cat(
                [
                    graph.unpool_senders(atoms),
                    undirected.unpool_bonds(bonds),
                    graph.unpool_receivers(atoms),
                ],
                dim=1,
            )
        )
        received = graph.aggregate_edges_to_nodes(
            messages * undirected.unpool_bonds(weights), to="sender"
        )

        return atoms + self.mlp_out(received)


class BondConv(nn.Module):
    """Update of the bonds from the angles they form with the other bonds."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.twoBody_bond = GatedMLP(4 * latent_dim, [latent_dim], latent_dim)
        self.mlp_out = MLP(latent_dim, [], latent_dim, bias=False)

    def forward(
        self,
        triplet_features: Float[torch.Tensor, "n_triplets 4*d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        weights: Float[torch.Tensor, "n_bonds d"],
        bond_triplets: EdgeTuples,
    ) -> Float[torch.Tensor, "n_bonds d"]:
        messages = (
            self.twoBody_bond(triplet_features)
            * bond_triplets.unpool_edges(weights, position=0)
            * bond_triplets.unpool_edges(weights, position=1)
        )
        received = bond_triplets.aggregate_to_edges(messages, position=0)

        return bonds + self.mlp_out(received)


class AngleUpdate(nn.Module):
    """Update of the angles from their two bonds and their atom."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.twoBody_bond = GatedMLP(4 * latent_dim, [], latent_dim)

    def forward(
        self,
        triplet_features: Float[torch.Tensor, "n_triplets 4*d"],
        angles: Float[torch.Tensor, "n_triplets d"],
    ) -> Float[torch.Tensor, "n_triplets d"]:
        return angles + self.twoBody_bond(triplet_features)


class AtomRef(nn.Module):
    """Reference energy of each element, indexed by atomic number minus one."""

    def __init__(self, num_elements: int):
        super().__init__()
        self.fc = nn.Linear(num_elements, 1, bias=False)

    def forward(
        self, element: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms"]:
        return self.fc.weight[0].double()[element]


# one module per module of chgnet
# pylint: disable-next=too-many-instance-attributes
class CHGNet(PretrainedModel[Config]):
    """Total energy of crystal structures predicted by CHGNet 0.3.0.

    Examples:
        >>> model = CHGNet()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    CONFIG = Config
    GRAPH_FIELDS = ("graph", "triplets", "bonds")

    element_table: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model
        latent_dim, num_layers = network.latent_dim, network.num_layers

        self.register_buffer(
            "element_table",
            element_table(torch.arange(1, network.num_elements + 1)),
            persistent=False,
        )
        self.composition_model = AtomRef(network.num_elements)
        self.atom_embedding = AtomEmbedding(network)
        self.bond_basis_expansion = BondEncoder(self.config)
        self.bond_embedding = nn.Linear(network.num_radial, latent_dim, bias=False)
        self.bond_weights_ag = nn.Linear(network.num_radial, latent_dim, bias=False)
        self.bond_weights_bg = nn.Linear(network.num_radial, latent_dim, bias=False)
        self.angle_basis_expansion = AngleEncoder(network)
        self.angle_embedding = nn.Linear(network.num_angular, latent_dim, bias=False)

        self.atom_conv_layers = nn.ModuleList(
            AtomConv(latent_dim) for _ in range(num_layers)
        )
        self.bond_conv_layers = nn.ModuleList(
            BondConv(latent_dim) for _ in range(num_layers - 1)
        )
        self.angle_layers = nn.ModuleList(
            AngleUpdate(latent_dim) for _ in range(num_layers - 1)
        )

        self.readout_norm = nn.LayerNorm(latent_dim)
        self.mlp = MLP(latent_dim, [latent_dim] * network.readout_layers, 1)

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them."""
        (path,) = download("chgnet", name, cache)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.load_pretrained(
            checkpoint["model"]["state_dict"], checkpoint["model"]["model_args"]
        )

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        model_args: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a CHGNet model of chgnet.

        Args:
            state_dict: Parameters of the model of chgnet.
            model_args: Arguments of the model of chgnet, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was built with other arguments than CHGNet 0.3.0.
        """
        if model_args is not None:
            different = different_arguments(model_args, self.config.upstream.model_args)
            if different:
                raise InvalidWeights(
                    f"weights of a model with other arguments: {different}"
                )

        self.load_parameters(state_dict)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the graph, the angles and the bonds of the model.

        Atoms closer than the radius, 6 Å, are connected, periodic images of
        the atom itself included. Angles are the pairs of edges shorter than
        the bond radius, 3 Å, starting from the same atom. An edge and its
        reverse are the same bond.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return Compose(
            [
                AddRadiusGraph(graph.radius, self_images=True),
                AddTriplets(radius=graph.bond_radius, strict=True),
                AddBonds(),
            ]
        )

    def graphs(self, structures: Structures) -> tuple[
        Graph,
        UndirectedBonds,
        Float[torch.Tensor, "n_bonds"],
        EdgeTuples,
        Float[torch.Tensor, "n_triplets"],
    ]:
        """Bonds and angles of the structures, with their geometry.

        They are the graph, the triplets and the bonds attached to the
        structures by transform. An edge and its reverse are the same bond.

        Returns:
            Graph of the atoms, its bonds, length of each bond in Å, shape
            (n_bonds,), triplets of the graph, the two edges of each angle,
            and cosine of each angle, shape (n_triplets,).
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        angles = attached_tuples(structures, "triplets", self)
        undirected = UndirectedBonds.of(attached_tuples(structures, "bonds", self))

        vectors = edges_vectors(graph)
        distance = vectors.norm(dim=-1)

        cosines = triplet_cosines(vectors, angles)

        return graph, undirected, undirected.from_edges(distance), angles, cosines

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        element = element_index(structures.z, self.element_table)
        graph, undirected, bond_length, edge_triplets, cosines = self.graphs(structures)
        bond_triplets = undirected.on_bonds(edge_triplets)

        basis = self.bond_basis_expansion.rbf_expansion_ag(bond_length)

        atoms = self.atom_embedding(element)
        bonds = self.bond_embedding(basis)
        weights = self.bond_weights_ag(basis)
        weights_angles = self.bond_weights_bg(
            self.bond_basis_expansion.rbf_expansion_bg(bond_length)
        )
        angles = self.angle_embedding(self.angle_basis_expansion(cosines))

        for atom_layer, bond_layer, angle_layer in zip(
            self.atom_conv_layers, self.bond_conv_layers, self.angle_layers
        ):
            atoms = atom_layer(atoms, bonds, weights, graph, undirected)

            bonds = bond_layer(
                self.triplet_features(
                    atoms, bonds, angles, graph, edge_triplets, bond_triplets
                ),
                bonds,
                weights_angles,
                bond_triplets,
            )
            angles = angle_layer(
                self.triplet_features(
                    atoms, bonds, angles, graph, edge_triplets, bond_triplets
                ),
                angles,
            )

        atoms = self.atom_conv_layers[-1](atoms, bonds, weights, graph, undirected)

        # energy of each atom, and reference energy of its element
        per_atom = self.mlp(self.readout_norm(atoms)).squeeze(-1).double()
        per_atom = per_atom + self.composition_model(element)

        return graph.aggregate_nodes_to_structures(per_atom)

    @staticmethod
    def triplet_features(
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        angles: Float[torch.Tensor, "n_triplets d"],
        graph: Graph,
        edge_triplets: EdgeTuples,
        bond_triplets: EdgeTuples,
    ) -> Float[torch.Tensor, "n_triplets 4*d"]:
        """Features of the two bonds, of the angle and of the atom of each angle.

        Args:
            atoms: Features of the atoms.
            bonds: Features of the bonds.
            angles: Features of the angles.
            graph: Graph of the atoms.
            edge_triplets: Angles as pairs of edges of the graph.
            bond_triplets: The same angles as pairs of bonds.
        """
        # both edges of an angle start from its atom
        centers = edge_triplets.unpool_edges(graph.unpool_senders(atoms), position=0)

        return torch.cat(
            [
                bond_triplets.unpool_edges(bonds, position=0),
                bond_triplets.unpool_edges(bonds, position=1),
                angles,
                centers,
            ],
            dim=1,
        )
