"""Configuration of the two models of DiffCSP++: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchms.models.estimator.config import YamlConfig
from torchms.models.layers import cspnet, schedules


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the models: every pair of atoms of a structure.

    Args:
        self_loops: If True, an atom is also connected to itself.
    """

    self_loops: bool


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the diffusion.

    Args:
        time_dim: Size of the embedding of the time.
        time_max_period: Inverse of the lowest frequency of this embedding.
        schedules: Steps of the diffusion and schedules of its noise.
    """

    time_dim: int
    time_max_period: float
    schedules: schedules.Config


@dataclass(frozen=True)
class FamilyConfig:
    """Numbers of the lattice a crystal family fixes.

    Args:
        last_spacegroup: Last space group of the family: a family starts after
            the last group of the one before it.
        fixed: Numbers of the vector k of the lattice which are fixed.
        values: Their values, in the same order.
    """

    last_spacegroup: int
    fixed: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class LatticeConfig:
    """Lattices as vectors of six numbers, constrained by the space groups.

    Args:
        basis: Symmetric matrices B_i, before their normalization: a lattice
            is exp(sum_i k_i B_i / |B_i|).
        families: Crystal families, in the order of the space groups, from
            the group 0.
    """

    basis: tuple[tuple[tuple[float, float, float], ...], ...]
    families: tuple[FamilyConfig, ...]


@dataclass(frozen=True)
class SamplingConfig:
    """Predictor and corrector of the sampling.

    Args:
        step_sizes: Step of the corrector at the lowest level of noise, for
            each weights.
    """

    step_sizes: dict[str, float]


@dataclass(frozen=True)
class GenerationConfig:
    """What the model which also generates the elements has of its own.

    Args:
        default_weights: Name of its weights loaded by from_pretrained.
        decoder: Hyperparameters of its CSPNet which are not the ones of the
            other model.
    """

    default_weights: str
    decoder: dict[str, Any]


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the configuration stored with the weights of DiffCSP++.

    Args:
        model_args: Configuration of the model given the elements.
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

        Examples:
            >>> sections = Config.load().upstream.sections(generation=True)
            >>> sections["model"]["_target_"]
            'diffcsp.pl_modules.diffusion_w_type.CSPDiffusion'
        """
        return {
            "model": self.generation_model_args if generation else self.model_args,
            "model.decoder": self.decoder_args,
            "model.beta_scheduler": self.beta_scheduler_args,
            "model.sigma_scheduler": self.sigma_scheduler_args,
        }


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of the two models of DiffCSP++, the one of config.yaml by default.

    Args:
        graph: Graph of the models.
        model: Hyperparameters of the diffusion.
        decoder: Hyperparameters of the CSPNet, see decoder_config.
        lattice: Lattices as vectors, constrained by the space groups.
        sampling: Steps of the sampling.
        generation: What the model of the generation has of its own.
        upstream: What is checked in the configuration of DiffCSP++.
    """

    graph: GraphConfig
    model: NetworkConfig
    decoder: dict[str, Any]
    lattice: LatticeConfig
    sampling: SamplingConfig
    generation: GenerationConfig
    upstream: UpstreamConfig

    def decoder_config(self, generation: bool = False) -> cspnet.Config:
        """Hyperparameters of the CSPNet, which reads the embedding of the time.

        Args:
            generation: If True, the ones of the model of the generation.

        Examples:
            >>> config = Config.load()
            >>> config.decoder_config().lattice_dim
            6
            >>> config.decoder_config(generation=True).predict_elements
            True
        """
        fields = {**self.decoder, **(self.generation.decoder if generation else {})}

        return cspnet.Config(latent_dim=self.model.time_dim, **fields)
