"""Configuration of ALIGNN: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the model.

    Args:
        radius: An atom is connected to its neighbours closer than this
            radius, in Å.
        three_body_radius: Largest length of the two bonds of an angle of the
            line graph, in Å.
        max_neighbors: Largest number of neighbours of an atom, the nearest.
    """

    radius: float
    three_body_radius: float
    max_neighbors: int


@dataclass(frozen=True)
class BasisConfig:
    """Gaussians of a value, centered on a regular grid.

    Args:
        start: First center.
        stop: Last center.
        num_basis: Number of gaussians.
    """

    start: float
    stop: float
    num_basis: int


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the network.

    Args:
        hidden_dim: Features of the atoms, of the bonds and of the angles.
        embedding_dim: Hidden layer of the embeddings.
        num_layers: Layers on both graphs, then on the atoms alone.
        num_element_features: Features of an element.
        element_features: File of the features of the elements, in the
            directory of the model.
        edge_basis: Gaussians of the lengths of the bonds, in Å.
        angle_basis: Gaussians of the cosines of the angles.
        gate_epsilon: Added to the sum of the gates of a node.
    """

    hidden_dim: int
    embedding_dim: int
    num_layers: int
    num_element_features: int
    element_features: str
    edge_basis: BasisConfig
    angle_basis: BasisConfig
    gate_epsilon: float


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the configuration stored with the weights of alignn.

    Args:
        model_args: Configuration of the model the rewrite is written for.
        graph_args: Configuration of its graphs.
    """

    model_args: dict[str, object]
    graph_args: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of ALIGNN, the one of config.yaml by default.

    Args:
        graph: Radii and number of neighbours of the graph.
        model: Hyperparameters of the network.
        upstream: What is checked in the configuration of alignn.
    """

    graph: GraphConfig
    model: NetworkConfig
    upstream: UpstreamConfig

    def element_features_path(self) -> Path:
        """File of the features of the elements, next to this module."""
        return Path(__file__).parent / self.model.element_features
