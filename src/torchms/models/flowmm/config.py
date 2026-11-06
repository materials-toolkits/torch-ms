"""Configuration of FlowMM: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the network: every pair of atoms of a structure.

    Args:
        self_loops: If True, an atom is also connected to itself.
    """

    self_loops: bool


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the CSPNet of FlowMM.

    Args:
        hidden_dim: Features of the atoms and of the messages.
        time_dim: Size of the embedding of the time.
        num_layers: Number of message passing layers.
        num_freqs: Frequencies of the Fourier basis of the differences of the
            fractional coordinates.
        num_bits: Analog bits of an element.
        max_num_atoms: Length of the one-hot vector of the number of atoms of
            a structure: a structure has fewer atoms.
    """

    hidden_dim: int
    time_dim: int
    num_layers: int
    num_freqs: int
    num_bits: int
    max_num_atoms: int

    @property
    def edge_dim(self) -> int:
        """What a layer reads on an edge, besides the features of its two atoms.

        The 3 cosines of the edge with the lattice vectors, the embedding of
        the number of atoms, the 6 lattice parameters, and a sine and a cosine
        of each frequency for each of the 3 coordinates.

        Examples:
            >>> Config.load().model.edge_dim
            1289
        """
        return 3 + self.hidden_dim + 6 + 2 * 3 * self.num_freqs


@dataclass(frozen=True)
class FlowConfig:
    """Spaces of the flow and its base distribution.

    Args:
        analog_bits_scale: Value of a bit of an element: -scale or +scale.
        angle_min: The angles of a cell, in degrees, are larger.
        angle_max: The angles of a cell, in degrees, are smaller.
        length_log_mean: Mean of the logarithm of each length of a cell, in
            Å, in the base distribution.
        length_log_std: Deviation of the logarithm of each length.
    """

    analog_bits_scale: float
    angle_min: float
    angle_max: float
    length_log_mean: tuple[float, float, float]
    length_log_std: tuple[float, float, float]


@dataclass(frozen=True)
class SamplingConfig:
    """Integration of the flow, and the numbers of atoms to sample.

    Args:
        num_steps: Euler steps from the time 0 to 1.
        inference_anneal_slope: The velocity of the coordinates is multiplied
            by 1 + slope * max(t - offset, 0).
        inference_anneal_offset: Time from which it is.
        num_atoms_distribution: Probability of each number of atoms, from 0.
    """

    num_steps: int
    inference_anneal_slope: float
    inference_anneal_offset: float
    num_atoms_distribution: tuple[float, ...]


@dataclass(frozen=True)
class UpstreamConfig:
    """What the checkpoint of FlowMM is read with, and what is checked in it.

    Args:
        model_prefix: Prefix of the parameters and buffers of the network in
            the state dict.
        ema_prefix: Prefix of the moving average of its parameters, numbered
            in their order.
        base_distribution_from_data: Whether the flow starts from structures
            of data, which the rewrite does not.
        sections: Arguments of the configuration of the training, by section,
            see check_configuration.
    """

    model_prefix: str
    ema_prefix: str
    base_distribution_from_data: bool
    sections: dict[str, dict[str, object]]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of FlowMM, the one of config.yaml by default.

    Args:
        graph: Graph of the network.
        model: Hyperparameters of the network.
        flow: Spaces of the flow and its base distribution.
        sampling: Integration of the flow.
        upstream: How the weights of FlowMM are read and checked.
    """

    graph: GraphConfig
    model: NetworkConfig
    flow: FlowConfig
    sampling: SamplingConfig
    upstream: UpstreamConfig
