# Derived from alignn (https://github.com/usnistgov/alignn), developed by
# employees of the National Institute of Standards and Technology (see LICENSE
# in this directory).
#
# Changes: this file is a rewrite of the prediction of a property by the
# ALIGNN 2.0 model "alignn_atomwise_pure", for inference only. Forces, stress,
# the other convolutions and read-outs, training and the conversion from
# jarvis-tools are removed. Graphs are built by torchms.graph, and batches
# are torchms Structures. Names of modules and parameters follow alignn to
# load its pretrained weights.
"""Prediction of a property of a crystal by a pretrained ALIGNN model.

ALIGNN passes messages on two graphs. In the graph of the atoms, an atom is
connected to its 12 nearest neighbours closer than 5 Å. In the line graph, the
nodes are these bonds, and a bond u→v is connected to the bonds v→w starting
from the atom it arrives at, for bonds of at most 3.5 Å: an edge of the line
graph is the angle at v. Each of the first four layers updates the atoms and
the bonds on the graph of the atoms, then the bonds and the angles on the line
graph; four more layers work on the graph of the atoms alone. The prediction is
read from the mean of the features of the atoms.

Weights are the ones published with alignn, see weights.py.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.batching import Structures
from torchms.estimator.alignn.weights import download
from torchms.graph.cutoff import cutoff, nearest_neighbors
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import Graph
from torchms.graph.line_graph import line_graph
from torchms.graph.sparse import scatter

RADIUS = 5.0
THREE_BODY_RADIUS = 3.5
MAX_NEIGHBORS = 12
HIDDEN_DIM = 512
EMBEDDING_DIM = 64
NUM_EDGE_BASES = 80
NUM_ANGLE_BASES = 40
NUM_ELEMENT_FEATURES = 92

FEATURES = Path(__file__).parent / "cgcnn_features.npz"

# the table of the features of the elements is empty in the pretrained weights
UNUSED_PARAMETERS = ("_species_table",)

# configuration of the model the rewrite is written for, as stored with the weights
MODEL_ARGS = {
    "name": "alignn_atomwise_pure",
    "alignn_layers": 4,
    "gcn_layers": 4,
    "atom_input_features": NUM_ELEMENT_FEATURES,
    "edge_input_features": NUM_EDGE_BASES,
    "triplet_input_features": NUM_ANGLE_BASES,
    "embedding_features": EMBEDDING_DIM,
    "hidden_features": HIDDEN_DIM,
    "output_features": 1,
    "calculate_gradient": False,
    "energy_mult_natoms": False,
    "use_penalty": False,
    "link": "identity",
    "classification": False,
    "use_cutoff_function": False,
}
GRAPH_ARGS = {
    "atom_features": "cgcnn",
    "cutoff": RADIUS,
    "max_neighbors": MAX_NEIGHBORS,
    "three_body_cutoff": THREE_BODY_RADIUS,
}


class InvalidWeights(Exception):
    """Raised when weights do not match the parameters of the model."""


class RBFExpansion(nn.Module):
    """Gaussians of a value, centered on a regular grid."""

    centers: torch.Tensor

    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__()
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        # as alignn, the inverse of the step of the grid, not of its square
        self.gamma = (bins - 1) / (vmax - vmin)

    def forward(self, value: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n bins"]:
        return torch.exp(-self.gamma * (value.unsqueeze(1) - self.centers) ** 2)


class MLPLayer(nn.Module):
    """Linear layer followed by a normalization and an activation."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size), nn.LayerNorm(output_size), nn.SiLU()
        )

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self.layer(x)


class EdgeGatedGraphConv(nn.Module):
    """Update of the nodes and of the edges of a graph.

    A node receives the features of its neighbours, weighted by a gate computed
    from the two nodes and their edge, and normalized by the sum of the gates.
    """

    def __init__(self):
        super().__init__()
        self.src_gate = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dst_gate = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.edge_gate = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.bn_edges = nn.LayerNorm(HIDDEN_DIM)
        self.src_update = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dst_update = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.bn_nodes = nn.LayerNorm(HIDDEN_DIM)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_nodes d"],
        edges: Float[torch.Tensor, "n_edges d"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
    ) -> tuple[Float[torch.Tensor, "n_nodes d"], Float[torch.Tensor, "n_edges d"]]:
        sender, receiver = edge_index

        messages = (
            self.src_gate(nodes)[sender]
            + self.dst_gate(nodes)[receiver]
            + self.edge_gate(edges)
        )
        gate = torch.sigmoid(messages)

        received = scatter(self.dst_update(nodes)[sender] * gate, receiver, len(nodes))
        received = received / (scatter(gate, receiver, len(nodes)) + 1e-6)

        new_nodes = nn.functional.silu(self.bn_nodes(self.src_update(nodes) + received))
        new_edges = nn.functional.silu(self.bn_edges(messages))

        return nodes + new_nodes, edges + new_edges


class ALIGNNConv(nn.Module):
    """Update on the graph of the atoms, then on its line graph."""

    def __init__(self):
        super().__init__()
        self.node_update = EdgeGatedGraphConv()
        self.edge_update = EdgeGatedGraphConv()

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        angles: Float[torch.Tensor, "n_pairs d"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        pair_index: Int[torch.Tensor, "2 n_pairs"],
    ) -> tuple[
        Float[torch.Tensor, "n_atoms d"],
        Float[torch.Tensor, "n_edges d"],
        Float[torch.Tensor, "n_pairs d"],
    ]:
        atoms, bonds = self.node_update(atoms, bonds, edge_index)
        bonds, angles = self.edge_update(bonds, angles, pair_index)

        return atoms, bonds, angles


def element_features() -> Float[torch.Tensor, "119 92"]:
    """Features of each element, indexed by atomic number.

    They are the features "cgcnn" of jarvis-tools, 92 values in {0, 1} which
    encode properties of the element. Elements without features are zeros.

    Examples:
        >>> features = element_features()
        >>> features.shape
        torch.Size([119, 92])
        >>> features[6].sum()  # one value per property of carbon
        tensor(9.)
    """
    with np.load(FEATURES) as data:
        return torch.from_numpy(data["features"]).float()


class AlignnModel(nn.Module):
    """Property of crystal structures predicted by ALIGNN.

    The pretrained model predicts the formation energy in eV/atom.

    Examples:
        >>> model = AlignnModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with torch.no_grad():
        ...     prediction = model(structures)
        >>> prediction.shape
        torch.Size([2])
    """

    species_table: torch.Tensor

    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.atom_embedding = MLPLayer(NUM_ELEMENT_FEATURES, HIDDEN_DIM)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(0.0, 8.0, NUM_EDGE_BASES),
            MLPLayer(NUM_EDGE_BASES, EMBEDDING_DIM),
            MLPLayer(EMBEDDING_DIM, HIDDEN_DIM),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(-1.0, 1.0, NUM_ANGLE_BASES),
            MLPLayer(NUM_ANGLE_BASES, EMBEDDING_DIM),
            MLPLayer(EMBEDDING_DIM, HIDDEN_DIM),
        )
        self.alignn_layers = nn.ModuleList(ALIGNNConv() for _ in range(num_layers))
        self.gcn_layers = nn.ModuleList(EdgeGatedGraphConv() for _ in range(num_layers))
        self.fc = nn.Linear(HIDDEN_DIM, 1)

        self.register_buffer("species_table", element_features(), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        name: str = "alignn-formation-energy-radius",
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> AlignnModel:
        """Load a model with pretrained weights, downloaded on the first call.

        Args:
            name: Name of the weights, see torchms.estimator.alignn.weights.
            device: Device of the model.
            cache: Directory of the downloaded weights.
        """
        with zipfile.ZipFile(download(name, cache)) as archive:
            config = json.loads(archive.read("config.json"))
            state_dict = torch.load(
                io.BytesIO(archive.read("best_model.pt")),
                map_location="cpu",
                weights_only=True,
            )

        model = cls()
        model.load_pretrained(state_dict, config)

        return model.to(device).eval().requires_grad_(False)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a model "alignn_atomwise_pure" of alignn.

        Args:
            state_dict: Parameters of the model of alignn.
            config: Configuration of the training of alignn, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with another configuration.
        """
        if config is not None:
            model_config = config.get("model")
            if not isinstance(model_config, Mapping):
                raise InvalidWeights("the configuration has no model")

            different = {
                key: model_config.get(key)
                for key, value in MODEL_ARGS.items()
                if model_config.get(key, value) != value
            }
            different.update(
                {
                    key: config.get(key)
                    for key, value in GRAPH_ARGS.items()
                    if config.get(key, value) != value
                }
            )
            if different:
                raise InvalidWeights(
                    f"weights of a model with another configuration: {different}"
                )

        used = {
            name: values
            for name, values in state_dict.items()
            if not name.startswith(UNUSED_PARAMETERS)
        }

        try:
            self.load_state_dict(used, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e

    def neighbors(self, structures: Structures) -> Graph:
        """Graph of the atoms of the model.

        An atom is connected to its 12 nearest neighbours closer than 5 Å,
        periodic images of the atom itself included. Among neighbours at the
        same distance, the ones which are kept are not the ones of alignn.
        """
        return nearest_neighbors(
            cutoff(structures, RADIUS, self_images=True), MAX_NEIGHBORS
        )

    def angles(self, graph: Graph) -> tuple[
        Float[torch.Tensor, "n_edges"],
        Int[torch.Tensor, "2 n_pairs"],
        Float[torch.Tensor, "n_pairs"],
    ]:
        """Line graph of a graph of the atoms.

        A bond u→v is followed by every bond v→w, its own reverse included,
        both of at most 3.5 Å.

        Returns:
            Length of each bond in Å, shape (n_edges,), indices of the two
            bonds of each angle, shape (2, n_pairs), and cosine of the angle
            at v between u and w, shape (n_pairs,).
        """
        vectors = graph.edges_vectors()
        distance = vectors.norm(dim=-1)

        short = (distance <= THREE_BODY_RADIUS).nonzero().squeeze(1)
        pair_index = short[
            line_graph(graph.edge_index[:, short], int(graph.num_atoms.sum()))
        ]

        # the first bond arrives at v, the second one leaves it
        cosine = -triplet_cosines(vectors, pair_index)

        return distance, pair_index, cosine

    def forward(
        self, structures: Structures, graph: Graph | None = None
    ) -> Float[torch.Tensor, "n_struct"]:
        """Predict the property of each structure.

        Args:
            structures: Batched Structures object.
            graph: Graph of the atoms, the one of neighbors by default.
        """
        if graph is None:
            graph = self.neighbors(structures)

        distance, pair_index, cosine = self.angles(graph)

        atoms = self.atom_embedding(self.species_table[structures.z])
        bonds = self.edge_embedding(distance)
        angles = self.angle_embedding(cosine)

        for layer in self.alignn_layers:
            atoms, bonds, angles = layer(
                atoms, bonds, angles, graph.edge_index, pair_index
            )

        for layer in self.gcn_layers:
            atoms, bonds = layer(atoms, bonds, graph.edge_index)

        mean = scatter(atoms, structures.batch, len(structures.num_atoms), "mean")

        return self.fc(mean).squeeze(-1)
