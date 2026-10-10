"""Configuration of CHGNet: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the model.

    Args:
        radius: Atoms closer than this radius are bonded, in Å.
        bond_radius: The two bonds of an angle are shorter than this radius,
            in Å.
    """

    radius: float
    bond_radius: float


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the network.

    Args:
        latent_dim: Features of the atoms, of the bonds and of the angles.
        num_elements: Rows of the embedding of the elements, from hydrogen.
        num_radial: Functions of the radial basis.
        num_angular: Functions of the basis of the angles.
        cutoff_power: Power of the smooth cutoff of the bonds.
        num_layers: Number of layers.
        readout_layers: Hidden layers of the read-out.
        cosine_shrink: As chgnet, cosines are shrunk by 1 minus this value.
    """

    latent_dim: int
    num_elements: int
    num_radial: int
    num_angular: int
    cutoff_power: int
    num_layers: int
    readout_layers: int
    cosine_shrink: float


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the weights of chgnet.

    Args:
        model_args: Arguments of the model of chgnet the rewrite is written
            for, as stored with the weights.
    """

    model_args: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of CHGNet, the one of config.yaml by default.

    Args:
        graph: Radii of the graph.
        model: Hyperparameters of the network.
        upstream: What is checked in the weights of chgnet.
    """

    graph: GraphConfig
    model: NetworkConfig
    upstream: UpstreamConfig
