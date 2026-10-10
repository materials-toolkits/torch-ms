"""Configuration of CDVAE: the fields of config.yaml, in this directory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchms.models.estimator.config import YamlConfig
from torchms.models.layers import dimenet, gemnet


@dataclass(frozen=True)
class DecoderGraphConfig:
    """Graph of the decoder.

    Args:
        radius: An atom is connected to its neighbours closer than this
            radius, in Å.
        max_neighbors: Largest number of neighbours of an atom, the nearest.
        max_image_shift: Largest number of cells between an atom and the image
            of a neighbour, along each lattice vector.
    """

    radius: float
    max_neighbors: int
    max_image_shift: int


@dataclass(frozen=True)
class GraphConfig:
    """Graphs of the two networks.

    Args:
        encoder: Nothing: the encoder reads a graph of bonds it is given.
        decoder: Graph the decoder builds at each step.
    """

    encoder: dict[str, object]
    decoder: DecoderGraphConfig


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the autoencoder.

    Args:
        latent_dim: Latent vector of a structure.
        hidden_dim: Hidden layers of the networks of the latent vector.
        num_hidden_layers: Number of these hidden layers.
        max_atoms: Largest number of atoms of a structure.
        num_elements: Elements a decoded atom can have, from hydrogen.
        sigma_begin: Largest level of the noise of the positions, in Å.
        sigma_end: Smallest level of the noise.
        num_noise_levels: Number of levels of noise.
    """

    latent_dim: int
    hidden_dim: int
    num_hidden_layers: int
    max_atoms: int
    num_elements: int
    sigma_begin: float
    sigma_end: float
    num_noise_levels: int


@dataclass(frozen=True)
class SamplingConfig:
    """Annealed Langevin dynamics.

    Args:
        steps_per_level: Number of steps at each level of noise.
        step_size: Step at the lowest level of noise.
        min_sigma: The dynamics stops before the first level of noise below
            it, in Å.
    """

    steps_per_level: int
    step_size: float
    min_sigma: float


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked and renamed in the weights of cdvae.

    Args:
        renamed_parameters: Buffers of cdvae which have another name here.
        constant_buffers: Buffers of the rewrite which are constants, absent
            from the weights.
        model_args: Configuration of the model the rewrite is written for.
        encoder_args: Configuration of its encoder.
        decoder_args: Configuration of its decoder.
        data_args: Configuration of its data.
    """

    renamed_parameters: dict[str, str]
    constant_buffers: tuple[str, ...]
    model_args: dict[str, object]
    encoder_args: dict[str, object]
    decoder_args: dict[str, object]
    data_args: dict[str, object]

    def sections(self) -> dict[str, dict[str, object]]:
        """Arguments checked in the configuration of a training, by section."""
        return {
            "model": self.model_args,
            "model.encoder": self.encoder_args,
            "model.decoder": self.decoder_args,
            "data": self.data_args,
        }


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of CDVAE, the one of config.yaml by default.

    Args:
        graph: Graphs of the encoder and of the decoder.
        model: Hyperparameters of the autoencoder.
        encoder: Hyperparameters of the DimeNet++, see encoder_config.
        decoder: Hyperparameters of the GemNet-T, see decoder_config.
        sampling: Annealed Langevin dynamics.
        upstream: What is checked and renamed in the weights of cdvae.
    """

    graph: GraphConfig
    model: NetworkConfig
    encoder: dict[str, Any]
    decoder: dict[str, Any]
    sampling: SamplingConfig
    upstream: UpstreamConfig

    def encoder_config(self) -> dimenet.Config:
        """Hyperparameters of the encoder, which gives the latent vector.

        Examples:
            >>> encoder = Config.load().encoder_config()
            >>> encoder.out_channels, encoder.num_blocks
            (256, 4)
        """
        return dimenet.Config(out_channels=self.model.latent_dim, **self.encoder)

    def decoder_config(self) -> gemnet.Config:
        """Hyperparameters of the decoder, which reads the latent vector.

        Examples:
            >>> decoder = Config.load().decoder_config()
            >>> decoder.latent_dim, decoder.radius, decoder.num_elements
            (256, 7.0, 100)
        """
        return gemnet.Config(
            latent_dim=self.model.latent_dim,
            num_elements=self.model.num_elements,
            radius=self.graph.decoder.radius,
            **self.decoder,
        )
