"""Configuration of CrystalFlow: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class GraphConfig:
    """Graph of the network: every pair of atoms of a structure.

    Args:
        self_loops: If True, an atom is also connected to itself.
    """

    self_loops: bool


@dataclass(frozen=True)
class NetworkConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the CSPNet of CrystalFlow.

    Args:
        num_elements: Rows of the embedding of the elements, from hydrogen.
        hidden_dim: Features of the atoms and of the messages.
        num_layers: Number of message passing layers.
        num_freqs: Frequencies of the Fourier basis of the differences of the
            fractional coordinates.
        num_millers: Miller indices per direction of the plane waves of the
            reciprocal lattice.
        lattice_dim: Numbers of the polar decomposition of a cell.
        time_dim: Size of the embedding of the time.
        time_max_period: Inverse of the lowest frequency of this embedding.
        soft_elements: If True, the element of an atom is a vector of type_dim
            values, read by a linear layer, in place of an index read in a
            table.
        predict_elements: If True, the network also gives a vector of
            type_dim values for each atom.
        type_dim: Size of the vector of an element.
        conditions: Properties the network is conditioned on, by name.
        condition_basis: Gaussians of the expansion of a property.
        condition_start: Center of the first one, for the standardized property.
        condition_stop: Center of the last one.
    """

    num_elements: int
    hidden_dim: int
    num_layers: int
    num_freqs: int
    num_millers: int
    lattice_dim: int
    time_dim: int
    time_max_period: float
    soft_elements: bool
    predict_elements: bool
    type_dim: int
    conditions: tuple[str, ...]
    condition_basis: int
    condition_start: float
    condition_stop: float

    @property
    def condition_dim(self) -> int:
        """Size of the embedding of the conditions, the gaussians of each property.

        Examples:
            >>> config = Config.load().for_weights("crystalflow-csp-calypso60-pressure")
            >>> config.model.condition_dim
            50
        """
        return len(self.conditions) * self.condition_basis

    @property
    def edge_dim(self) -> int:
        """What a layer reads on an edge, besides the features of its two atoms.

        The plane waves of the Miller indices, a cosine and a sine each, the
        Fourier basis of the difference of the coordinates and the numbers of
        the cell.

        Examples:
            >>> Config.load().model.edge_dim
            2566
        """
        return 2 * self.num_millers**3 + 2 * 3 * self.num_freqs + self.lattice_dim


@dataclass(frozen=True)
class FlowConfig:
    """Base distribution of the flow.

    Args:
        lattice_polar_sigma: Deviation of the normal noise of the numbers of
            the cell at the time 0; 1 is added to the last one.
    """

    lattice_polar_sigma: float


@dataclass(frozen=True)
class SamplingConfig:
    """Integration of the flow.

    Args:
        num_steps: Euler steps from the time 0 to 1.
        anneal_slope: The velocity of the coordinates is multiplied by
            1 + slope * max(t - offset, 0).
        anneal_offset: Time from which it is.
        guide_factor: A model with conditions mixes guide_factor times its
            velocity with the conditions and 1 - guide_factor times the one
            without.
    """

    num_steps: int
    anneal_slope: float
    anneal_offset: float
    guide_factor: float


@dataclass(frozen=True)
class GenerationConfig:
    """What the model which also generates the elements has of its own.

    Args:
        default_weights: Name of its weights loaded by from_pretrained.
        num_rows: Rows of the table of the elements.
        num_columns: Its columns.
        table: Row and column of each element, by atomic number from 1.
        num_atoms_distribution: Probability of each number of atoms, from 0.
    """

    default_weights: str
    num_rows: int
    num_columns: int
    table: tuple[tuple[int, int], ...]
    num_atoms_distribution: tuple[float, ...]


@dataclass(frozen=True)
class MembersConfig:
    """Files of a model in the archive of the weights.

    Args:
        checkpoint: Path of its checkpoint of pytorch-lightning.
        checkpoint_sha256: SHA-256 of this file.
        hparams: Path of the configuration of its training.
        hparams_sha256: SHA-256 of this file.
        scalers: Path of the standardization of its properties, for a model
            with conditions.
        scalers_sha256: SHA-256 of this file.
    """

    checkpoint: str
    checkpoint_sha256: str
    hparams: str
    hparams_sha256: str
    scalers: str | None = None
    scalers_sha256: str | None = None


@dataclass(frozen=True)
class UpstreamConfig:
    """How the weights of CrystalFlow are read and checked.

    Args:
        members: Files of each model in the archive, by name of the weights.
        sections: Arguments of the configuration of the training, by section,
            see check_configuration.
    """

    members: dict[str, dict[str, str]]
    sections: dict[str, dict[str, object]]

    def members_of(self, name: str) -> MembersConfig:
        """Files of the model of some weights in the archive.

        Raises:
            KeyError: If the archive has no model of this name.

        Examples:
            >>> Config.load().upstream.members_of("crystalflow-csp-mp20").hparams
            'ckpt-v1.0.0/CSP-mp-20/hparams.yaml'
        """
        if name not in self.members:
            raise KeyError(f'no files of the weights "{name}" in the archive')

        return MembersConfig(**self.members[name])


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of CrystalFlow, the one of config.yaml by default.

    Args:
        graph: Graph of the network.
        model: Hyperparameters of the network.
        flow: Base distribution of the flow.
        sampling: Integration of the flow.
        generation: What the model of the generation has of its own.
        variants: What the network and the sampling of other weights change,
            by name of the weights: values of the sections model ("model") and
            sampling ("sampling"), unused_parameters, and the arguments
            checked in their configuration ("sections"). See for_weights.
        upstream: How the weights of CrystalFlow are read and checked.
    """

    graph: GraphConfig
    model: NetworkConfig
    flow: FlowConfig
    sampling: SamplingConfig
    generation: GenerationConfig
    variants: dict[str, dict[str, Any]]
    upstream: UpstreamConfig

    def for_weights(self, name: str) -> Config:
        """Configuration of the network and the sampling of some weights.

        Args:
            name: Name of weights: the default ones, or a key of variants.

        Raises:
            KeyError: If the weights are neither.

        Examples:
            >>> config = Config.load().for_weights("crystalflow-dng-mp20")
            >>> config.default_weights, config.model.predict_elements
            ('crystalflow-dng-mp20', True)
            >>> config.sampling.anneal_slope
            0.0
            >>> config.upstream.sections["model"]["type_encoding"]
            'table'
            >>> Config.load().for_weights("crystalflow-csp-mp20") == Config.load()
            True
        """
        if name == self.default_weights:
            return self
        if name not in self.variants:
            raise KeyError(f'no network for the weights "{name}"')

        variant = self.variants[name]
        sections = {
            section: {**arguments, **variant.get("sections", {}).get(section, {})}
            for section, arguments in self.upstream.sections.items()
        }
        sections.update(
            {
                section: arguments
                for section, arguments in variant.get("sections", {}).items()
                if section not in sections
            }
        )
        model = dict(variant.get("model", {}))
        if "conditions" in model:
            model["conditions"] = tuple(model["conditions"])

        return replace(
            self,
            default_weights=name,
            unused_parameters=tuple(
                variant.get("unused_parameters", self.unused_parameters)
            ),
            model=replace(self.model, **model),
            sampling=replace(self.sampling, **variant.get("sampling", {})),
            upstream=replace(self.upstream, sections=sections),
        )
