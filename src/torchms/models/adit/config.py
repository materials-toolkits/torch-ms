"""Configuration of ADiT: the fields of config.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from torchms.models.estimator.config import YamlConfig


@dataclass(frozen=True)
class AutoencoderConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the variational autoencoder.

    Args:
        num_elements: Rows of the embedding of the elements and outputs of the
            prediction of the element of an atom, indexed by atomic number:
            0 is a masked atom.
        latent_dim: Size of the latent vector of an atom.
        logvar_min: The log-variance of the posterior is not smaller.
        logvar_max: The log-variance of the posterior is not larger.
        d_model: Features of an atom in the transformers of the encoder and
            of the decoder.
        nhead: Heads of their attention.
        dim_feedforward: Hidden features of their feed-forward networks.
        num_layers: Layers of each transformer.
        activation: Activation of the feed-forward networks, "gelu" or
            "relu", as nn.TransformerEncoderLayer reads it.
        position_max_len: Longest period of the embedding of the index of an
            atom in its structure.
        position_scale: Å in a unit of the positions the decoder predicts.
    """

    num_elements: int
    latent_dim: int
    logvar_min: float
    logvar_max: float
    d_model: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    activation: str
    position_max_len: int
    position_scale: float


@dataclass(frozen=True)
class DenoiserConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the diffusion transformer.

    Args:
        latent_dim: Size of the latent vector of an atom, its input and its
            output.
        d_model: Features of an atom.
        num_layers: Number of blocks.
        nhead: Heads of the attention.
        mlp_ratio: Hidden features of a feed-forward network over d_model.
        num_datasets: Classes of the dataset a structure comes from; the class
            0, without dataset, is added for the guidance.
        crystal_class: Class of the crystals.
        molecule_class: Class of the molecules.
        num_spacegroups: Space groups the network reads; the class 0, without
            space group, is added.
        frequency_embedding_dim: Sines and cosines of the embedding of the
            time.
        time_max_period: Inverse of the lowest frequency of this embedding.
        position_max_len: Longest period of the embedding of the index of an
            atom.
        layer_norm_eps: Epsilon of the normalizations of the blocks.
    """

    latent_dim: int
    d_model: int
    num_layers: int
    nhead: int
    mlp_ratio: float
    num_datasets: int
    crystal_class: int
    molecule_class: int
    num_spacegroups: int
    frequency_embedding_dim: int
    time_max_period: float
    position_max_len: int
    layer_norm_eps: float

    @property
    def mlp_hidden_dim(self) -> int:
        """Hidden features of the feed-forward networks.

        Examples:
            >>> Config.load().denoiser.mlp_hidden_dim
            3072
        """
        return int(self.d_model * self.mlp_ratio)


@dataclass(frozen=True)
class SamplingConfig:
    """Flow matching from normal noise to latent vectors.

    Args:
        min_t: First time, where the latent vectors are noise; the last one
            is 1.
        num_timesteps: Number of times from min_t to 1, both included.
        cfg_scale: Scale of the classifier-free guidance by the dataset.
        num_atoms_crystals: Number of crystals of the training set by number
            of atoms, from 0.
        num_atoms_molecules: The same for the molecules.
    """

    min_t: float
    num_timesteps: int
    cfg_scale: float
    num_atoms_crystals: tuple[int, ...]
    num_atoms_molecules: tuple[int, ...]


@dataclass(frozen=True)
class UpstreamConfig:
    """What is checked in the modules pickled with the weights of ADiT.

    Args:
        denoiser: Attributes of its DiT.
        interpolant: Attributes of its FlowMatchingInterpolant.
        conditioning: Its conditioning: by the dataset, not by the space group.
    """

    denoiser: dict[str, object]
    interpolant: dict[str, object]
    conditioning: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(YamlConfig):
    """Configuration of ADiT, the one of config.yaml by default.

    Args:
        autoencoder: Hyperparameters of the variational autoencoder.
        denoiser: Hyperparameters of the diffusion transformer.
        sampling: Flow matching and the numbers of atoms to sample.
        upstream: What is checked in the weights of ADiT.
    """

    autoencoder: AutoencoderConfig
    denoiser: DenoiserConfig
    sampling: SamplingConfig
    upstream: UpstreamConfig
