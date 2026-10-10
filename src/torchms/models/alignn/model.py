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

Weights are the ones published with alignn, see config.yaml.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Mapping

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.models.alignn.config import BasisConfig, Config, NetworkConfig
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
)
from torchms.models.estimator.weights import download
from torchms.graph.geometry import pair_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddLineGraph,
    AddRadiusGraph,
    KeepNearestNeighbors,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.layers.basis import GaussianBasis


def rbf_expansion(basis: BasisConfig) -> GaussianBasis:
    """Gaussians of a value, centered on a regular grid.

    As alignn, their gamma is the inverse of the step of the grid, not of its
    square.
    """
    gamma = (basis.num_basis - 1) / (basis.stop - basis.start)

    return GaussianBasis(basis.start, basis.stop, basis.num_basis, gamma=gamma)


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

    The graph is the one of the atoms, or its line graph, whose nodes are the
    bonds and whose edges are the angles, see EdgeTuples.as_graph.
    """

    def __init__(self, network: NetworkConfig):
        super().__init__()
        hidden_dim = network.hidden_dim
        self.gate_epsilon = network.gate_epsilon

        self.src_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dst_gate = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)
        self.bn_edges = nn.LayerNorm(hidden_dim)
        self.src_update = nn.Linear(hidden_dim, hidden_dim)
        self.dst_update = nn.Linear(hidden_dim, hidden_dim)
        self.bn_nodes = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_nodes d"],
        edges: Float[torch.Tensor, "n_edges d"],
        graph: Graph,
    ) -> tuple[Float[torch.Tensor, "n_nodes d"], Float[torch.Tensor, "n_edges d"]]:
        messages = (
            graph.unpool_senders(self.src_gate(nodes))
            + graph.unpool_receivers(self.dst_gate(nodes))
            + self.edge_gate(edges)
        )
        gate = torch.sigmoid(messages)

        neighbors = graph.unpool_senders(self.dst_update(nodes))
        received = graph.aggregate_edges_to_nodes(neighbors * gate)
        received = received / (graph.aggregate_edges_to_nodes(gate) + self.gate_epsilon)

        new_nodes = nn.functional.silu(self.bn_nodes(self.src_update(nodes) + received))
        new_edges = nn.functional.silu(self.bn_edges(messages))

        return nodes + new_nodes, edges + new_edges


class ALIGNNConv(nn.Module):
    """Update on the graph of the atoms, then on its line graph."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.node_update = EdgeGatedGraphConv(network)
        self.edge_update = EdgeGatedGraphConv(network)

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        angles: Float[torch.Tensor, "n_pairs d"],
        graph: Graph,
        line_graph: Graph,
    ) -> tuple[
        Float[torch.Tensor, "n_atoms d"],
        Float[torch.Tensor, "n_edges d"],
        Float[torch.Tensor, "n_pairs d"],
    ]:
        atoms, bonds = self.node_update(atoms, bonds, graph)
        bonds, angles = self.edge_update(bonds, angles, line_graph)

        return atoms, bonds, angles


def element_features(file_name: str | os.PathLike) -> Float[torch.Tensor, "119 92"]:
    """Features of each element, indexed by atomic number.

    They are the features "cgcnn" of jarvis-tools, 92 values in {0, 1} which
    encode properties of the element. Elements without features are zeros.

    Args:
        file_name: File of the features, see Config.element_features_path.

    Examples:
        >>> features = element_features(Config.load().element_features_path())
        >>> features.shape
        torch.Size([119, 92])
        >>> features[6].sum()  # one value per property of carbon
        tensor(9.)
    """
    with np.load(file_name) as data:
        return torch.from_numpy(data["features"]).float()


class ALIGNN(PretrainedModel[Config]):
    """Property of crystal structures predicted by ALIGNN.

    The pretrained model predicts the formation energy in eV/atom.

    Examples:
        >>> model = ALIGNN()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     prediction = model(structures)
        >>> prediction.shape
        torch.Size([2])
    """

    CONFIG = Config
    GRAPH_FIELDS = ("graph", "line_graph")

    species_table: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model
        hidden_dim, embedding_dim = network.hidden_dim, network.embedding_dim

        self.atom_embedding = MLPLayer(network.num_element_features, hidden_dim)
        self.edge_embedding = nn.Sequential(
            rbf_expansion(network.edge_basis),
            MLPLayer(network.edge_basis.num_basis, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )
        self.angle_embedding = nn.Sequential(
            rbf_expansion(network.angle_basis),
            MLPLayer(network.angle_basis.num_basis, embedding_dim),
            MLPLayer(embedding_dim, hidden_dim),
        )

        self.alignn_layers = nn.ModuleList(
            ALIGNNConv(network) for _ in range(network.num_layers)
        )
        self.gcn_layers = nn.ModuleList(
            EdgeGatedGraphConv(network) for _ in range(network.num_layers)
        )
        self.fc = nn.Linear(hidden_dim, 1)

        self.register_buffer(
            "species_table",
            element_features(self.config.element_features_path()),
            persistent=False,
        )

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them."""
        (path,) = download("alignn", name, cache)
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))
            state_dict = torch.load(
                io.BytesIO(archive.read("best_model.pt")),
                map_location="cpu",
                weights_only=True,
            )

        self.load_pretrained(state_dict, config)

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

            upstream = self.config.upstream
            different = different_arguments(model_config, upstream.model_args)
            different.update(different_arguments(config, upstream.graph_args))
            if different:
                raise InvalidWeights(
                    f"weights of a model with another configuration: {different}"
                )

        self.load_parameters(state_dict)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the graph of the atoms and its line graph.

        An atom is connected to its nearest neighbours, 12 of them, closer than
        the radius, 5 Å, periodic images of the atom itself included. Among
        neighbours at the same distance, the ones which are kept are not the
        ones of alignn. In the line graph, a bond u→v is followed by every bond
        v→w, its own reverse included, both of at most the three-body radius,
        3.5 Å.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return Compose(
            [
                AddRadiusGraph(graph.radius, self_images=True),
                KeepNearestNeighbors(graph.max_neighbors),
                AddLineGraph(radius=graph.three_body_radius, backtrack=True),
            ]
        )

    def graphs(self, structures: Structures) -> tuple[
        Graph,
        Float[torch.Tensor, "n_edges"],
        EdgeTuples,
        Float[torch.Tensor, "n_pairs"],
    ]:
        """Bonds and angles of the structures, with their geometry.

        They are the graph and the line graph attached to the structures by
        transform.

        Returns:
            Graph of the atoms, length of each bond in Å, shape (n_edges,),
            pairs of bonds of the line graph, one per angle, and cosine of the
            angle at v between u and w, shape (n_pairs,).
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        pairs = attached_tuples(structures, "line_graph", self)

        vectors = edges_vectors(graph)

        return graph, vectors.norm(dim=-1), pairs, pair_cosines(vectors, pairs)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the property of each structure."""
        graph, distance, pairs, cosine = self.graphs(structures)

        atoms = self.atom_embedding(self.species_table[structures.z])
        bonds = self.edge_embedding(distance)
        angles = self.angle_embedding(cosine)

        line_graph = pairs.as_graph()

        for layer in self.alignn_layers:
            atoms, bonds, angles = layer(atoms, bonds, angles, graph, line_graph)

        for layer in self.gcn_layers:
            atoms, bonds = layer(atoms, bonds, graph)

        mean = graph.aggregate_nodes_to_structures(atoms, "mean")

        return self.fc(mean).squeeze(-1)
