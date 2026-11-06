"""Configuration of MatterSim: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Bonds and angles of the model.

    Args:
        radius: Atoms closer than this, in Å, are bonded.
        three_body_radius: Angles are between two bonds not longer than this.
    """

    radius: float
    three_body_radius: float


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the network.

    Args:
        units: Features of the atoms and of the bonds.
        max_n: Radial functions of the bonds, and per degree of the basis of
            the angles.
        max_l: Degrees of the basis of the angles.
        num_blocks: Number of blocks of message passing.
        max_z: Last atomic number of the embedding, which starts at 0.
        cosine_margin: The cosines of the angles are kept this far from -1
            and 1.
        cutoff_power: Power of the smooth cutoff of the two bonds of an angle.
    """

    units: int
    max_n: int
    max_l: int
    num_blocks: int
    max_z: int
    cosine_margin: float
    cutoff_power: int

    @property
    def basis_dim(self) -> int:
        """Size of the basis of an angle.

        Examples:
            >>> Config.load().model.basis_dim
            16
        """
        return self.max_n * self.max_l


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the weights of MatterSim.

    Args:
        model_name: Architecture of the weights.
        model_args: Arguments of the model stored with them.
    """

    model_name: str
    model_args: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of MatterSim, the one of config.yaml by default.

    Args:
        graph: Bonds and angles of the model.
        model: Hyperparameters of the network.
        upstream: What is checked in the weights of MatterSim.
    """

    graph: GraphConfig
    model: NetworkConfig
    upstream: UpstreamConfig
