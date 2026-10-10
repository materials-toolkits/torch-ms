# Derived from orb-models (https://github.com/orbital-materials/orb-models),
# Copyright 2024 Orbital Materials, Inc., licensed under the Apache License,
# Version 2.0 (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the energy prediction of the orb-v3
# conservative model, for inference only. Forces, stress, charges, dispersion,
# training, datasets and the conversion from ASE are removed. Graphs are built
# by torchms.graph instead of nvalchemi-toolkit-ops, and batches are
# torchms Structures. Names of modules and parameters follow orb-models to
# load its pretrained weights.
#
# The radial basis, the smooth cutoff and the spherical harmonics of
# orb-models are in torchms/models/layers/basis.py.
"""Energy prediction of the pretrained orb-v3 conservative model.

The network is a graph neural network on the atoms closer than 6 Å: edge
features are the product of a radial basis and spherical harmonics, updated by
five attention layers. The energy is the sum of a head on the mean of the atom
features, a pair repulsion at short distance and a reference energy per atom.

Weights are the ones published by Orbital Materials, see config.yaml.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.models.estimator.config import load_config
from torchms.models.estimator.pretrained import InvalidWeights, PretrainedModel
from torchms.models.estimator.weights import download
from torchms.graph.graph import Graph
from torchms.graph.transforms import (
    AddRadiusGraph,
    KeepNearestNeighbors,
    attached_graph,
    edges_vectors,
)
from torchms.models.layers.basis import (
    NUM_HARMONICS,
    BesselBasis,
    polynomial_cutoff,
    spherical_harmonics,
)

# every value of the model is in config.yaml, next to this module
CONFIG = load_config(__file__)

RADIUS: float = CONFIG["graph"]["radius"]
MAX_NEIGHBORS: int = CONFIG["graph"]["max_neighbors"]

LATENT_DIM: int = CONFIG["model"]["latent_dim"]
HIDDEN_DIM: int = CONFIG["model"]["hidden_dim"]
NUM_ELEMENTS: int = CONFIG["model"]["num_elements"]
NUM_RADIAL: int = CONFIG["model"]["num_radial"]
NUM_LAYERS: int = CONFIG["model"]["num_layers"]
CUTOFF_POWER: int = CONFIG["model"]["cutoff_power"]

# parameters of the pretrained weights which are not used to predict the energy
UNUSED_PARAMETERS: tuple[str, ...] = tuple(CONFIG["unused_parameters"])

# Ziegler-Biersack-Littmark repulsion, see ZBLBasis
REPULSION = CONFIG["repulsion"]
COVALENT_RADII: tuple[float, ...] = tuple(REPULSION["covalent_radii"])


def build_mlp(
    input_size: int, hidden_sizes: list[int], output_size: int
) -> nn.Sequential:
    """Fully connected network with SiLU activations between its layers."""
    sizes = [input_size, *hidden_sizes, output_size]
    mlp = nn.Sequential()

    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        mlp.add_module(f"NN-{i}", nn.Linear(sizes[i], sizes[i + 1]))
        mlp.add_module(f"Act-{i}", nn.Identity() if last else nn.SiLU())

    return mlp


def mlp_and_norm(input_size: int, hidden_size: int = HIDDEN_DIM) -> nn.Sequential:
    """Network with two hidden layers followed by a normalization."""
    return nn.Sequential(
        OrderedDict(
            mlp=build_mlp(input_size, [hidden_size, hidden_size], LATENT_DIM),
            layer_norm=nn.RMSNorm(LATENT_DIM),
        )
    )


class AtomEmbedding(nn.Module):
    """Learned features of each element, indexed by atomic number."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(NUM_ELEMENTS + 1, LATENT_DIM)

    def forward(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms d"]:
        return self.embeddings(z)


class Encoder(nn.Module):
    """Initial features of atoms and edges."""

    def __init__(self, num_edge_features: int):
        super().__init__()
        self._node_fn = mlp_and_norm(LATENT_DIM)
        self._edge_fn = mlp_and_norm(num_edge_features)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_atoms d"],
        edges: Float[torch.Tensor, "n_edges f"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        return self._node_fn(nodes), self._edge_fn(edges)


class AttentionInteractionNetwork(nn.Module):
    """Message passing layer updating the features of atoms and edges.

    Messages are weighted by a sigmoid attention times the smooth cutoff of the
    edge. An atom is updated from the messages it sends and the ones it
    receives.
    """

    def __init__(self):
        super().__init__()
        self._node_mlp = mlp_and_norm(3 * LATENT_DIM)
        self._edge_mlp = mlp_and_norm(3 * LATENT_DIM)
        self._receive_attn = nn.Linear(LATENT_DIM, 1)
        self._send_attn = nn.Linear(LATENT_DIM, 1)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_atoms d"],
        edges: Float[torch.Tensor, "n_edges d"],
        graph: Graph,
        envelope: Float[torch.Tensor, "n_edges 1"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        receive_attn = torch.sigmoid(self._receive_attn(edges)) * envelope
        send_attn = torch.sigmoid(self._send_attn(edges)) * envelope

        updated_edges = self._edge_mlp(
            torch.cat(
                [edges, graph.unpool_senders(nodes), graph.unpool_receivers(nodes)],
                dim=1,
            )
        )

        sent = graph.aggregate_edges_to_nodes(updated_edges * send_attn, to="sender")
        received = graph.aggregate_edges_to_nodes(updated_edges * receive_attn)

        updated_nodes = self._node_mlp(torch.cat([nodes, received, sent], dim=1))

        return nodes + updated_nodes, edges + updated_edges


class MoleculeGNS(nn.Module):
    """Graph network computing the features of each atom."""

    def __init__(self, num_message_passing_steps: int = NUM_LAYERS):
        super().__init__()
        self.rbf_transform = BesselBasis(RADIUS, NUM_RADIAL)
        self.atom_emb = AtomEmbedding()
        self._encoder = Encoder(num_edge_features=NUM_RADIAL * NUM_HARMONICS)
        self.gnn_stacks = nn.ModuleList(
            AttentionInteractionNetwork() for _ in range(num_message_passing_steps)
        )

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        vectors: Float[torch.Tensor, "n_edges 3"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_atoms d"]:
        distance = vectors.norm(dim=1)
        envelope = polynomial_cutoff(distance, RADIUS, p=CUTOFF_POWER).unsqueeze(-1)

        # outer product of the radial and the angular bases
        radial = self.rbf_transform(distance)
        angular = spherical_harmonics(vectors)
        edges = envelope * (radial[:, :, None] * angular[:, None, :]).flatten(1)

        nodes, edges = self._encoder(self.atom_emb(z), edges)

        for layer in self.gnn_stacks:
            nodes, edges = layer(nodes, edges, graph, envelope)

        return nodes


class ScalarNormalizer(nn.Module):
    """Mean and standard deviation of the energy per atom seen in training."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1, affine=False, momentum=None)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        running_mean, running_var = self.bn.running_mean, self.bn.running_var
        if running_mean is None or running_var is None:
            raise InvalidWeights("the normalizer of the energy has no statistics")

        return values * torch.sqrt(running_var) + running_mean


class LinearReferenceEnergy(nn.Module):
    """Energy of a structure as a sum of an energy per atom of each element."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_ELEMENTS, 1, bias=False)

    def forward(
        self, z: Int[torch.Tensor, "n_atoms"], graph: Graph
    ) -> Float[torch.Tensor, "n_struct"]:
        return graph.aggregate_nodes_to_structures(self.linear.weight[0].double()[z])


class EnergyHead(nn.Module):
    """Energy of a structure from the mean of the features of its atoms."""

    def __init__(self):
        super().__init__()
        self.normalizer = ScalarNormalizer()
        self.mlp = build_mlp(LATENT_DIM, [LATENT_DIM], 1)
        self.reference = LinearReferenceEnergy()

    def forward(
        self, nodes: Float[torch.Tensor, "n_atoms d"], graph: Graph
    ) -> Float[torch.Tensor, "n_struct"]:
        num_atoms = graph.num_atoms
        mean = graph.aggregate_nodes_to_structures(nodes) / num_atoms.unsqueeze(-1)
        per_atom = self.normalizer.inverse(self.mlp(mean).squeeze(-1))

        return per_atom * num_atoms


class ZBLBasis(nn.Module):
    """Ziegler-Biersack-Littmark repulsion between close atoms.

    The repulsion of a pair is cut at the sum of the covalent radii of its
    atoms. As in the training of the pretrained weights, the energy of a
    structure is the mean over its atoms and atomic numbers are shifted by one.
    """

    c: torch.Tensor
    d: torch.Tensor
    covalent_radii: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "c", torch.tensor(REPULSION["coefficients"]), persistent=False
        )
        self.register_buffer(
            "d", torch.tensor(REPULSION["exponents"]), persistent=False
        )
        self.register_buffer(
            "covalent_radii", torch.tensor(COVALENT_RADII), persistent=False
        )

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        vectors: Float[torch.Tensor, "n_edges 3"],
        graph: Graph,
    ) -> Float[torch.Tensor, "n_struct"]:
        # orb-models reads the atomic number from a one-hot encoding indexed by
        # the atomic number and adds one: the repulsion of an element is the
        # one of the next element. The pretrained weights were trained with it.
        z_send = graph.unpool_senders(z) + 1
        z_receive = graph.unpool_receivers(z) + 1
        distance = vectors.norm(dim=1)

        # screening length of the pair
        power = REPULSION["screening_power"]
        length = (
            REPULSION["screening_length"]
            * REPULSION["bohr_radius"]
            / (torch.pow(z_send, power) + torch.pow(z_receive, power))
        )
        screening = (
            self.c * torch.exp(-self.d * (distance / length).unsqueeze(-1))
        ).sum(dim=-1)

        radius = self.covalent_radii[z_send] + self.covalent_radii[z_receive]
        pair = (
            0.5
            * (REPULSION["coulomb_constant"] * z_send * z_receive / distance)
            * screening
            * polynomial_cutoff(distance, radius, p=REPULSION["cutoff_power"])
        )

        per_atom = graph.aggregate_edges_to_nodes(pair, to="sender")
        return graph.aggregate_nodes_to_structures(per_atom) / graph.num_atoms


class OrbEnergyModel(PretrainedModel):
    """Total energy of crystal structures predicted by orb-v3 conservative.

    Examples:
        >>> model = OrbEnergyModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    DEFAULT_WEIGHTS = CONFIG["default_weights"]
    UNUSED_PARAMETERS = UNUSED_PARAMETERS

    def __init__(self):
        super().__init__()
        self.model = MoleculeGNS()
        self.heads = nn.ModuleDict({"energy": EnergyHead()})
        self.pair_repulsion_fn = ZBLBasis()

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them."""
        (path,) = download("orb", name, cache)
        self.load_pretrained(torch.load(path, map_location="cpu", weights_only=True))

    def load_pretrained(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the weights of an orb-v3 conservative model of orb-models.

        Raises:
            InvalidWeights: If parameters are missing or unknown.
        """
        parameters = dict(state_dict)

        reference = "heads.energy.reference.linear.weight"
        if reference in parameters:
            parameters[reference] = parameters[reference].reshape(1, NUM_ELEMENTS)

        self.load_parameters(parameters)

    @classmethod
    def transform(cls) -> Transform:
        """Transform attaching the graph of the model.

        Atoms are connected to their 120 nearest neighbours closer than 6 Å,
        periodic images of the atom itself included.
        """
        return Compose(
            [
                AddRadiusGraph(RADIUS, self_images=True),
                KeepNearestNeighbors(MAX_NEIGHBORS),
            ]
        )

    def edges(
        self, structures: Structures
    ) -> tuple[Graph, Float[torch.Tensor, "n_edges 3"]]:
        """Graph of the structures and vector of each edge, in Å.

        The graph is the one attached to the structures by transform. The
        vector of an edge goes from its sender to its receiver.

        Raises:
            ValueError: If the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)

        return graph, edges_vectors(graph)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision."""
        graph, vectors = self.edges(structures)
        z = structures.z

        nodes = self.model(z, vectors, graph)

        head = self.heads["energy"]
        if not isinstance(head, EnergyHead):
            raise TypeError("the energy head of the model has been replaced")

        energy = head(nodes, graph)
        energy = energy + self.pair_repulsion_fn(z, vectors, graph)

        return energy.double() + head.reference(z, graph)
