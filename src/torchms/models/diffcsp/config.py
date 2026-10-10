"""Configuration of the two models of DiffCSP: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchms.models.estimator.config import YamlConfig
from torchms.models.layers import cspnet


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the models: every pair of atoms of a structure.

    Args:
        self_loops: If True, an atom is also connected to itself.
    """

    self_loops: bool


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the diffusion.

    Args:
        timesteps: Steps of the diffusion.
        time_dim: Size of the embedding of the time.
        time_max_period: Inverse of the lowest frequency of this embedding.
        beta_offset: Offset s of the cosine schedule of the noise of the
            lattice.
        beta_min: The betas of this schedule are not smaller.
        beta_max: The betas of this schedule are not larger.
        sigma_begin: Noise of the fractional coordinates at the first time,
            in cells.
        sigma_end: Noise of the fractional coordinates at the last time.
    """

    timesteps: int
    time_dim: int
    time_max_period: float
    beta_offset: float
    beta_min: float
    beta_max: float
    sigma_begin: float
    sigma_end: float


@dataclass(frozen=True)
class SamplingConfig:
    """Predictor and corrector of the sampling from a composition.

    Args:
        step_size: Step of the corrector at the lowest level of noise.
    """

    step_size: float


@dataclass(frozen=True)
class GenerationConfig:
    """What the model which also generates the elements has of its own.

    Args:
        default_weights: Name of its weights loaded by from_pretrained.
        unused_parameters: Prefixes of the parameters of its weights which it
            does not have.
        decoder: Hyperparameters of its CSPNet which are not the ones of the
            other model.
        step_size: Step of its corrector at the lowest level of noise.
        num_atoms_distribution: Probability of each number of atoms, from 0.
    """

    default_weights: str
    unused_parameters: tuple[str, ...]
    decoder: dict[str, Any]
    step_size: float
    num_atoms_distribution: tuple[float, ...]


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the configuration stored with the weights of DiffCSP.

    Args:
        model_args: Configuration of the model given a composition.
        generation_model_args: Configuration of the model of the generation.
        decoder_args: Configuration of the CSPNet of both.
        beta_scheduler_args: Configuration of the noise of the lattice.
        sigma_scheduler_args: Configuration of the noise of the coordinates.
    """

    model_args: dict[str, object]
    generation_model_args: dict[str, object]
    decoder_args: dict[str, object]
    beta_scheduler_args: dict[str, object]
    sigma_scheduler_args: dict[str, object]

    def sections(self, generation: bool = False) -> dict[str, dict[str, object]]:
        """Arguments checked in the configuration of a training, by section.

        Args:
            generation: If True, the ones of the model of the generation.
        """
        return {
            "model": self.generation_model_args if generation else self.model_args,
            "model.decoder": self.decoder_args,
            "model.beta_scheduler": self.beta_scheduler_args,
            "model.sigma_scheduler": self.sigma_scheduler_args,
        }


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of the two models of DiffCSP, the one of config.yaml by default.

    Args:
        graph: Graph of the models.
        model: Hyperparameters of the diffusion.
        decoder: Hyperparameters of the CSPNet, see decoder_config.
        sampling: Sampling from a composition.
        generation: What the model of the generation has of its own.
        upstream: What is checked in the configuration of DiffCSP.
    """

    graph: GraphConfig
    model: NetworkConfig
    decoder: dict[str, Any]
    sampling: SamplingConfig
    generation: GenerationConfig
    upstream: UpstreamConfig

    def decoder_config(self, generation: bool = False) -> cspnet.Config:
        """Hyperparameters of the CSPNet, which reads the embedding of the time.

        Args:
            generation: If True, the ones of the model of the generation.

        Examples:
            >>> config = Config.load()
            >>> config.decoder_config().predict_elements
            False
            >>> config.decoder_config(generation=True).predict_elements
            True
        """
        fields = {**self.decoder, **(self.generation.decoder if generation else {})}

        return cspnet.Config(latent_dim=self.model.time_dim, **fields)
