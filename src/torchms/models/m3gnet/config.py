"""Configuration of M3GNet: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the model.

    Args:
        radius: Atoms closer than this radius are bonded, in Å.
        three_body_radius: Largest length of the two bonds of an angle, in Å.
    """

    radius: float
    three_body_radius: float


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the network.

    Args:
        latent_dim: Features of the atoms and of the bonds.
        max_n: Radial functions per degree.
        max_l: Degrees of the basis of the angles.
        num_blocks: Number of blocks.
        cutoff_power: Power of the smooth cutoff of the angles.
        cosine_margin: The cosines of the angles are kept this far from ±1.
        elements: Atomic numbers of the elements, in the order of the
            embedding, as the first and the last of each range.
    """

    latent_dim: int
    max_n: int
    max_l: int
    num_blocks: int
    cutoff_power: int
    cosine_margin: float
    elements: tuple[tuple[int, int], ...]

    @property
    def basis_dim(self) -> int:
        """Number of functions of the bases of a bond and of an angle."""
        return self.max_n * self.max_l

    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic number of each element, in the order of the embedding.

        Examples:
            >>> network = Config.load().model
            >>> numbers = network.atomic_numbers()
            >>> len(numbers), numbers[:3], numbers[-1]
            (89, (1, 2, 3), 94)
        """
        return tuple(z for first, last in self.elements for z in range(first, last + 1))


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the weights of matgl.

    Args:
        model_args: Arguments of the model of matgl the rewrite is written
            for, as stored with the weights.
    """

    model_args: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of M3GNet, the one of config.yaml by default.

    Args:
        graph: Radii of the graph.
        model: Hyperparameters of the network.
        upstream: What is checked in the weights of matgl.
    """

    graph: GraphConfig
    model: NetworkConfig
    upstream: UpstreamConfig
