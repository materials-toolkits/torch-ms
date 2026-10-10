"""Configuration of orb-v3: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the model.

    Args:
        radius: An atom is connected to its neighbours closer than this
            radius, in Å.
        max_neighbors: Largest number of neighbours of an atom, the nearest.
    """

    radius: float
    max_neighbors: int


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the network.

    Args:
        latent_dim: Features of the atoms and of the edges.
        hidden_dim: Hidden layers of the networks of the layers.
        num_elements: Number of elements, indexed by atomic number.
        num_radial: Functions of the radial basis.
        num_layers: Message passing layers.
        cutoff_power: Power of the smooth cutoff of the edges.
    """

    latent_dim: int
    hidden_dim: int
    num_elements: int
    num_radial: int
    num_layers: int
    cutoff_power: int


@dataclass(frozen=True)
class RepulsionConfig:  # pylint: disable=too-many-instance-attributes
    """Ziegler-Biersack-Littmark repulsion between close atoms.

    Args:
        coefficients: Coefficients of the exponentials of the screening.
        exponents: Exponents of these exponentials.
        screening_length: Screening length of two atoms, in Bohr radii, over
            the sum of their atomic numbers to the screening power.
        screening_power: Power of the atomic numbers in the screening length.
        bohr_radius: Bohr radius, in Å.
        coulomb_constant: Coulomb constant, in eV Å.
        cutoff_power: Power of the smooth cutoff at the covalent radii.
        covalent_radii: Covalent radii in Å, indexed by atomic number.
    """

    coefficients: tuple[float, ...]
    exponents: tuple[float, ...]
    screening_length: float
    screening_power: float
    bohr_radius: float
    coulomb_constant: float
    cutoff_power: int
    covalent_radii: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of orb-v3 conservative, the one of config.yaml by default.

    Args:
        graph: Radius and number of neighbours of the graph.
        model: Hyperparameters of the network.
        repulsion: Constants of the repulsion between close atoms.
    """

    graph: GraphConfig
    model: NetworkConfig
    repulsion: RepulsionConfig
