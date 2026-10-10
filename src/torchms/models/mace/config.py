"""Configuration of MACE: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig
from torchms.models.layers.basis import NUM_HARMONICS


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the model.

    Args:
        radius: Atoms closer than this radius are connected, in Å.
    """

    radius: float


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the network.

    Args:
        num_channels: Features of the atoms.
        num_elements: Number of elements of the model.
        num_bessel: Functions of the radial basis.
        cutoff_power: Power of the smooth cutoff of the edges.
        num_layers: Number of layers.
        max_l: Degree of the spherical harmonics.
        radial_hidden: Hidden layers of the network of the radial functions.
        readout_dim: Hidden layer of the read-out of the last layer.
        num_couplings: Invariants of one, two and three environments.
        activation: Constant of the normalized activation of e3nn, replaced
            by the one stored with the weights.
    """

    num_channels: int
    num_elements: int
    num_bessel: int
    cutoff_power: int
    num_layers: int
    max_l: int
    radial_hidden: int
    readout_dim: int
    num_couplings: tuple[int, int, int]
    activation: float


@dataclass(frozen=True)
class UpstreamConfig:
    """Classes of the model of mace the rewrite is written for.

    Args:
        model_class: Class of the model.
        interaction_class: Class of its interactions.
    """

    model_class: str
    interaction_class: str


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of MACE-MP-0 small, the one of config.yaml by default.

    Args:
        graph: Radius of the graph.
        model: Hyperparameters of the network.
        upstream: Classes of the model of mace the rewrite is written for.
    """

    graph: GraphConfig
    model: NetworkConfig
    upstream: UpstreamConfig

    def _layer_parameters(self, layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Parameters of a layer of mace, see upstream_parameters."""
        network = self.model
        interaction, product = f"interactions.{layer}.", f"products.{layer}."
        contraction = product + "symmetric_contractions.contractions.0."
        c, e, h = network.num_channels, network.num_elements, NUM_HARMONICS
        hidden, degrees = network.radial_hidden, network.max_l + 1
        one, two, three = network.num_couplings
        radial = interaction + "conv_tp_weights."

        return {
            interaction + "linear_up.weight": (interaction + "linear_up", (c, c)),
            radial
            + "layer0.weight": (
                interaction + "radial.0",
                (network.num_bessel, hidden),
            ),
            radial + "layer1.weight": (interaction + "radial.1", (hidden, hidden)),
            radial + "layer2.weight": (interaction + "radial.2", (hidden, hidden)),
            radial
            + "layer3.weight": (
                interaction + "radial.3",
                (hidden, degrees * c),
            ),
            interaction + "linear.weight": (interaction + "linear", (degrees, c, c)),
            interaction + "skip_tp.weight": (interaction + "skip", (c, e, c)),
            contraction + "U_matrix_1": (product + "U_matrix_1", (h, one)),
            contraction + "U_matrix_2": (product + "U_matrix_2", (h, h, two)),
            contraction + "U_matrix_3": (product + "U_matrix_3", (h, h, h, three)),
            contraction + "weights.1": (product + "weights_1", (e, one, c)),
            contraction + "weights.0": (product + "weights_2", (e, two, c)),
            contraction + "weights_max": (product + "weights_3", (e, three, c)),
            product + "linear.weight": (product + "linear", (c, c)),
        }

    def upstream_parameters(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Parameters of mace and the parameter of the rewrite each one becomes.

        Returns:
            Name in the rewrite and shape of each parameter of mace, by name.

        Examples:
            >>> parameters = Config.load().upstream_parameters()
            >>> parameters["node_embedding.linear.weight"]
            ('embedding', (89, 128))
            >>> parameters["interactions.1.skip_tp.weight"]
            ('interactions.1.skip', (128, 89, 128))
        """
        network = self.model
        c, e = network.num_channels, network.num_elements

        return {
            "atomic_numbers": ("atomic_numbers", (e,)),
            "atomic_energies_fn.atomic_energies": ("atomic_energies", (e,)),
            "node_embedding.linear.weight": ("embedding", (e, c)),
            "radial_embedding.bessel_fn.bessel_weights": (
                "bessel_weights",
                (network.num_bessel,),
            ),
            "readouts.0.linear.weight": ("readout_0", (c, 1)),
            "readouts.1.linear_1.weight": ("readout_1.0", (c, network.readout_dim)),
            "readouts.1.linear_2.weight": ("readout_1.1", (network.readout_dim, 1)),
            "scale_shift.scale": ("scale", ()),
            "scale_shift.shift": ("shift", ()),
            **{
                name: target
                for layer in range(network.num_layers)
                for name, target in self._layer_parameters(layer).items()
            },
        }

    def expected_values(self) -> dict[str, float]:
        """Values the rewrite relies on, checked in the weights of mace.

        Examples:
            >>> Config.load().expected_values()["num_interactions"]
            2
        """
        return {
            "r_max": self.graph.radius,
            "num_interactions": self.model.num_layers,
            "radial_embedding.cutoff_fn.p": self.model.cutoff_power,
            "radial_embedding.cutoff_fn.r_max": self.graph.radius,
            "radial_embedding.bessel_fn.r_max": self.graph.radius,
        }
