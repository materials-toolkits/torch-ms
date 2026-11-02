"""Configuration of MatterGen: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchms.models.estimator.config import YamlConfig
from torchms.models.layers import gemnet


@dataclass(frozen=True)
class GraphConfig:
    """Graph of GemNet-T.

    Args:
        radius: Radius of the graph, in Å.
        max_neighbors: Largest number of neighbours of an atom, before the
            graph is made symmetric.
        max_image_shift: Largest number of cells, along each lattice vector,
            between an atom and the image of its neighbour.
    """

    radius: float
    max_neighbors: int
    max_image_shift: int


@dataclass(frozen=True)
class NetworkConfig:
    """Hyperparameters of the network around GemNet-T.

    Args:
        hidden_dim: Size of the embedding of the time, and of the features of
            GemNet-T.
        time_max_period: Inverse of the lowest frequency of this embedding.
        num_elements: Number of elements, from hydrogen. One more class is the
            mask, the element of an atom which has none yet.
        allowed_elements: Atomic numbers an atom can take when it is sampled.
        masked_logit: Added to the score of the other elements.
    """

    hidden_dim: int
    time_max_period: float
    num_elements: int
    allowed_elements: tuple[int, ...]
    masked_logit: float

    @property
    def num_classes(self) -> int:
        """Number of classes of the element of an atom: the elements and the mask.

        Examples:
            >>> Config.load().model.num_classes
            101
        """
        return self.num_elements + 1

    @property
    def mask(self) -> int:
        """Atomic number of the mask: the one after the last element.

        Examples:
            >>> Config.load().model.mask
            101
        """
        return self.num_elements + 1


@dataclass(frozen=True)
class LatticeConfig:
    """Diffusion of the lattice, which preserves the variance.

    The lattice goes to a cubic cell of the mean density, with a variance per
    element which grows as the number of atoms to the power 2/3.

    Args:
        beta_min: Rate of the diffusion at the time 0.
        beta_max: Rate at the time 1. It grows linearly between them.
        limit_density: Density of the cell at the time 1, in atoms per Å³.
        limit_var_scaling: Variance of an element of the lattice at the time
            1, over the number of atoms to the power 2/3.
    """

    beta_min: float
    beta_max: float
    limit_density: float
    limit_var_scaling: float


@dataclass(frozen=True)
class CoordinatesConfig:
    """Diffusion of the fractional coordinates, wrapped around the cell.

    Args:
        sigma_min: Level of the noise at the time 0, before it is divided by
            the cube root of the number of atoms.
        sigma_max: Level at the time 1. It grows geometrically between them.
    """

    sigma_min: float
    sigma_max: float


@dataclass(frozen=True)
class ElementsConfig:
    """Diffusion of the elements, towards the mask.

    Args:
        num_steps: Steps of the diffusion. At the step t of N, an atom has
            been replaced by the mask with the probability t / N.
        log_epsilon: Added to a probability before its logarithm.
    """

    num_steps: int
    log_epsilon: float


@dataclass(frozen=True)
class SamplingConfig:
    """Predictor and corrector of the sampling.

    Args:
        num_steps: Steps of the sampling, from the time 1 to the time
            1 / num_steps.
        snr_coordinates: Ratio of the signal to the noise of the Langevin
            corrector of the coordinates.
        snr_lattice: The same for the lattice.
        max_step_size: Largest step of the Langevin corrector.
        min_alpha_ratio: Lower bound of the ratio of the means of the lattice
            at two consecutive times.
        num_atoms_distribution: Probability of each number of atoms, from 0.
    """

    num_steps: int
    snr_coordinates: float
    snr_lattice: float
    max_step_size: float
    min_alpha_ratio: float
    num_atoms_distribution: tuple[float, ...]


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the configuration stored with the weights of MatterGen.

    Args:
        prefix: Start of the names of the parameters of the denoiser in the
            checkpoint of MatterGen.
        renamed_parameters: End of the name here of the parameters which
            have another name in MatterGen, by the end of this name.
        sections: Arguments of the training, by section of its configuration,
            a section inside another one named with dots.
    """

    prefix: str
    renamed_parameters: dict[str, str]
    sections: dict[str, dict[str, object]]


@dataclass(frozen=True)
class ConditionConfig:
    """What a model fine-tuned on a property has of its own.

    Args:
        default_weights: Name of its weights loaded by from_pretrained.
        guidance_scale: Factor of the guidance of its sampling by default.
        upstream_sections: Sections of the configuration of its training
            which replace or complete the ones of the base model.
    """

    default_weights: str
    guidance_scale: float
    upstream_sections: dict[str, dict[str, object]]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of MatterGen, the one of config.yaml by default.

    Args:
        graph: Graph of GemNet-T.
        model: Hyperparameters of the network around GemNet-T.
        decoder: Hyperparameters of GemNet-T, see decoder_config.
        lattice: Diffusion of the lattice.
        coordinates: Diffusion of the fractional coordinates.
        elements: Diffusion of the elements.
        sampling: Predictor and corrector of the sampling.
        upstream: What is checked in the configuration of MatterGen.
        chemical_system: What the model conditioned on a chemical system has
            of its own.
        space_group: What the model conditioned on a space group has of its
            own.
    """

    graph: GraphConfig
    model: NetworkConfig
    decoder: dict[str, Any]
    lattice: LatticeConfig
    coordinates: CoordinatesConfig
    elements: ElementsConfig
    sampling: SamplingConfig
    upstream: UpstreamConfig
    chemical_system: ConditionConfig
    space_group: ConditionConfig

    def condition(self, name: str) -> ConditionConfig:
        """What the model fine-tuned on a property has of its own.

        Args:
            name: The property: "chemical_system" or "space_group".

        Examples:
            >>> Config.load().condition("space_group").default_weights
            'mattergen-space-group'
        """
        if name not in ("chemical_system", "space_group"):
            raise ValueError(f"no model of MatterGen is conditioned on {name}")

        condition = getattr(self, name)
        assert isinstance(condition, ConditionConfig)
        return condition

    def decoder_config(self, condition: str | None = None) -> gemnet.Config:
        """Hyperparameters of GemNet-T, which reads the embedding of the time.

        Its embedding of the elements has a row for the mask.

        Args:
            condition: The property of a fine-tuned model, which has its
                adapter. None for the base model.

        Examples:
            >>> decoder = Config.load().decoder_config()
            >>> decoder.latent_dim, decoder.num_elements, decoder.predict_lattice
            (512, 101, True)
            >>> Config.load().decoder_config("space_group").conditions
            ('space_group',)
        """
        return gemnet.Config(
            latent_dim=self.model.hidden_dim,
            num_elements=self.model.num_classes,
            radius=self.graph.radius,
            hidden_dim=self.model.hidden_dim,
            conditions=() if condition is None else (condition,),
            **self.decoder,
        )

    def upstream_sections(
        self, condition: str | None = None
    ) -> dict[str, dict[str, object]]:
        """Sections checked in the configuration of a training of MatterGen.

        Args:
            condition: The property of a fine-tuned model, None for the base
                model.

        Examples:
            >>> sections = Config.load().upstream_sections("chemical_system")
            >>> sections["lightning_module.diffusion_module.model"]["_target_"]
            'mattergen.adapter.GemNetTAdapter'
        """
        if condition is None:
            return dict(self.upstream.sections)

        return {**self.upstream.sections, **self.condition(condition).upstream_sections}
