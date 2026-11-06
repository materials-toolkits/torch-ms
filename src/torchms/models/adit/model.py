# Derived from All-atom Diffusion Transformers
# (https://github.com/facebookresearch/all-atom-diffusion-transformer),
# Copyright (c) Meta Platforms, Inc. and affiliates, CC BY-NC 4.0 (see LICENSE
# in this directory).
#
# Changes: this file is a rewrite of the autoencoder and of the latent
# diffusion of ADiT (src/models/encoders/transformer.py,
# src/models/decoders/transformer.py, src/models/vae_module.py,
# src/models/ldm_module.py and src/models/interpolants/flow_matching.py), for
# inference only: encoding structures, decoding latent vectors, the denoiser
# with its classifier-free guidance and the sampling by flow matching. The
# losses, the augmentations, the training, the exponential moving average of
# the weights, the datasets, the metrics, the visualization and the Equiformer
# encoder are removed, and so is the conditioning by the space group, which
# the published weights were not trained with. Batches are torchms
# Structures; the padded layout of the transformers comes from
# torchms.graph.sparse; the noise of the sampling is an argument of a step;
# the decoded crystals get their lattice from torchms.graph.geometry. Names of
# modules and parameters follow ADiT to load its weights.
"""ADiT, the All-atom Diffusion Transformer.

ADiT generates crystals and molecules with one model. A variational
autoencoder maps the atoms of a structure to latent vectors, one per atom: a
transformer reads the element, the position in Å and the fractional
coordinates of each atom (zero for a molecule), and the index of the atom in
its structure. A second transformer decodes the latent vectors into an element
for each atom, its fractional coordinates and its position, and the lengths
and the angles of a cell. A crystal is read from the fractional coordinates
and the cell, a molecule from the positions.

A diffusion transformer, DiT, generates the latent vectors by flow matching:
from centered normal noise at the time min_t to latent vectors at the time 1,
by Euler steps along the velocity (prediction - x) / (1 - t). It is guided by
the dataset, crystals (MP-20) or molecules (QM9): its prediction is the one
without dataset plus cfg_scale times the difference with the one of the
dataset.

Weights are listed in config.yaml: the ones of the first author of ADiT,
trained on QM9 and MP-20 together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from jaxtyping import Bool, Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.graph.geometry import lattice_from_parameters
from torchms.graph.sparse import padding_mask, to_dense
from torchms.models.adit.config import AutoencoderConfig, Config
from torchms.models.adit.dit import DiT, index_embedding
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
)
from torchms.models.estimator.reader import Placeholder, read_pickle
from torchms.models.estimator.weights import download


@dataclass(frozen=True)
class Posterior:
    """Normal distribution of the latent vector of each atom, from the encoder.

    Args:
        mean: Mean, shape (n_atoms, latent_dim): the latent vector the
            diffusion was trained to generate is drawn around it.
        logvar: Logarithm of the variance, clamped, shape (n_atoms,
            latent_dim).
    """

    mean: Float[torch.Tensor, "n_atoms latent_dim"]
    logvar: Float[torch.Tensor, "n_atoms latent_dim"]

    def sample(
        self, noise: Float[torch.Tensor, "n_atoms latent_dim"]
    ) -> Float[torch.Tensor, "n_atoms latent_dim"]:
        """Latent vectors drawn from standard normal noise."""
        return self.mean + torch.exp(0.5 * self.logvar) * noise


@dataclass(frozen=True)
class Decoded:
    """What the decoder predicts, in the units of ADiT.

    Args:
        atom_types: Score of each atomic number for each atom, from 0, a
            masked atom, shape (n_atoms, num_elements).
        lattices: Lengths of the cell over the cubic root of the number of
            atoms, in Å, then its angles in radians, shape (n_struct, 6).
        frac_coords: Fractional coordinates of each atom, shape (n_atoms, 3).
        pos: Position of each atom in nm, shape (n_atoms, 3).
    """

    atom_types: Float[torch.Tensor, "n_atoms num_elements"]
    lattices: Float[torch.Tensor, "n_struct 6"]
    frac_coords: Float[torch.Tensor, "n_atoms 3"]
    pos: Float[torch.Tensor, "n_atoms 3"]


def transformer(config: AutoencoderConfig) -> nn.TransformerEncoder:
    """Transformer of the encoder and of the decoder, with pre-normalization."""
    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        activation=config.activation,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        bias=True,
    )

    return nn.TransformerEncoder(
        layer,
        num_layers=config.num_layers,
        norm=nn.LayerNorm(config.d_model),
        enable_nested_tensor=False,
    )


def run_transformer(
    network: nn.TransformerEncoder,
    x: Float[torch.Tensor, "n_atoms d"],
    num_atoms: Int[torch.Tensor, "n_struct"],
    max_len: int,
) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_struct d"]]:
    """Features of the atoms after a transformer, the atoms of a structure in a row.

    The embedding of the index of an atom in its structure is added first.

    Returns:
        Features of each atom, and their mean over each structure.
    """
    dense, mask = to_dense(x, num_atoms)
    index = torch.arange(mask.shape[1], device=x.device)
    dense = dense + index_embedding(index, x.shape[-1], max_len)

    dense = network(dense, src_key_padding_mask=~mask)
    dense = dense * mask.unsqueeze(-1)
    mean = dense.sum(dim=1) / num_atoms.unsqueeze(1)

    return dense[mask], mean


class TransformerEncoder(nn.Module):
    """Encoder of ADiT: features of the atoms of structures."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.atom_type_embedder = nn.Embedding(config.num_elements, d_model)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.transformer = transformer(config)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_atoms d"]:
        """Features of each atom of crystals or molecules.

        The fractional coordinates of the atoms of a molecule are zero.
        """
        periodic = structures.pbc.all(dim=1).repeat_interleave(structures.num_atoms)
        frac_coords = torch.where(periodic.unsqueeze(1), structures.x, 0.0)

        x = self.atom_type_embedder(structures.z)
        x = x + self.pos_embedder(structures.positions())
        x = x + self.frac_coords_embedder(frac_coords)

        x, _ = run_transformer(
            self.transformer, x, structures.num_atoms, self.config.position_max_len
        )

        return x


class TransformerDecoder(nn.Module):
    """Decoder of ADiT: elements, coordinates and cell from features of atoms."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.transformer = transformer(config)
        self.atom_types_head = nn.Linear(d_model, config.num_elements)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(
        self,
        x: Float[torch.Tensor, "n_atoms d"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Decoded:
        """Predictions from the features of the atoms of each structure.

        Args:
            x: Features of each atom, the atoms of a structure together.
            num_atoms: Number of atoms of each structure.
        """
        x, x_global = run_transformer(
            self.transformer, x, num_atoms, self.config.position_max_len
        )

        return Decoded(
            atom_types=self.atom_types_head(x),
            lattices=self.lattice_head(x_global),
            frac_coords=self.frac_coords_head(x),
            pos=self.pos_head(x),
        )


class Autoencoder(nn.Module):  # pylint: disable=abstract-method
    """Variational autoencoder of ADiT: encoder, decoder and their projections."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.quant_conv = nn.Linear(config.d_model, 2 * config.latent_dim, bias=False)
        self.post_quant_conv = nn.Linear(config.latent_dim, config.d_model, bias=False)

    def encode(self, structures: Structures) -> Posterior:
        """Distribution of the latent vector of each atom of structures."""
        mean, logvar = self.quant_conv(self.encoder(structures)).chunk(2, dim=-1)
        logvar = logvar.clamp(self.config.logvar_min, self.config.logvar_max)

        return Posterior(mean=mean, logvar=logvar)

    def decode(
        self,
        latent: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Decoded:
        """Predictions from the latent vector of each atom."""
        return self.decoder(self.post_quant_conv(latent), num_atoms)


def _values(obj: object) -> dict[str, object]:
    """Attributes of a module pickled with the weights, or values of a DictConfig."""
    if isinstance(obj, Mapping):
        return dict(obj)
    if not isinstance(obj, Placeholder):
        return {}

    attributes = obj.attributes()
    if "_content" in attributes:
        return {
            name: (
                node.attributes().get("_val") if isinstance(node, Placeholder) else node
            )
            for name, node in attributes["_content"].items()
        }

    return {
        name: value
        for name, value in attributes.items()
        if not name.startswith("_") and name != "training"
    }


class ADiT(PretrainedModel[Config]):
    """ADiT: latent diffusion of crystals and molecules.

    The model reads no graph: its transformers read every atom of a
    structure. A structure is a crystal if it is periodic along its three
    lattice vectors, a molecule if it is not periodic.

    Args:
        config: Configuration of the model, the one of config.yaml by default.

    Examples:
        A crystal and a molecule, encoded and decoded:

        >>> model = ADiT().eval()
        >>> structures = Structures.collate([Structures.diamond(), Structures.water()])
        >>> with torch.no_grad():
        ...     posterior = model.encode(structures)
        ...     decoded = model.decode(
        ...         posterior.mean, structures.num_atoms, model.is_crystal(structures)
        ...     )
        >>> posterior.mean.shape, decoded.num_atoms, decoded.pbc[:, 0]
        (torch.Size([11, 8]), tensor([8, 3]), tensor([ True, False]))

        With pretrained weights,
        `model.sample(model.sample_num_atoms(8, crystals=True))` generates 8
        crystals.
    """

    CONFIG = Config
    GRAPH_FIELDS = ()

    num_atoms_crystals: torch.Tensor
    num_atoms_molecules: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.autoencoder = Autoencoder(self.config.autoencoder)
        self.denoiser = DiT(self.config.denoiser)

        sampling = self.config.sampling
        for name in ("num_atoms_crystals", "num_atoms_molecules"):
            counts = torch.tensor(getattr(sampling, name), dtype=torch.float32)
            self.register_buffer(name, counts, persistent=False)

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint of pytorch-lightning pickles the modules of ADiT with
        their hyperparameters: it is read without importing them. Its state
        dict is loaded, not the exponential moving average of the weights it
        also has, as src/eval_diffusion.py of ADiT does.
        """
        (path,) = download("adit", name, cache)

        checkpoint = read_pickle(path)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{path} is not a checkpoint of ADiT")

        self.load_pretrained(checkpoint["state_dict"], checkpoint["hyper_parameters"])

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        hparams: Mapping[str, Any] | None = None,
    ) -> None:
        """Load the weights of a LatentDiffusionLitModule of ADiT.

        Args:
            state_dict: Parameters of the model of ADiT, with its autoencoder.
            hparams: Hyperparameters stored with the weights, checked if given:
                the denoiser, the interpolant and the conditioning, as modules
                or mappings of their attributes.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with other hyperparameters.
        """
        if hparams is not None:
            different = {}
            for section, expected in vars(self.config.upstream).items():
                found = _values(hparams.get(section))
                if not found:
                    raise InvalidWeights(f"the hyperparameters have no {section}")
                different.update(
                    {
                        f"{section}.{key}": value
                        for key, value in different_arguments(found, expected).items()
                    }
                )
            if different:
                raise InvalidWeights(
                    f"weights of a model with other hyperparameters: {different}"
                )

        self.load_parameters(state_dict)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """No transform: the model reads no graph."""
        del config
        return Compose([])

    # autoencoder

    @staticmethod
    def is_crystal(structures: Structures) -> Bool[torch.Tensor, "n_struct"]:
        """Whether each structure is a crystal, periodic along its three vectors.

        Raises:
            ValueError: If a structure is periodic along some of its vectors
                only: ADiT has crystals and molecules.
        """
        periodic = structures.pbc.all(dim=1)
        if (periodic != structures.pbc.any(dim=1)).any():
            raise ValueError("ADiT reads crystals and molecules, not slabs or wires")

        return periodic

    def check_elements(self, z: Int[torch.Tensor, "n_atoms"]) -> None:
        """Refuse atoms the embedding of the encoder does not have.

        Raises:
            ValueError: If an atomic number is not between 1 and the number of
                rows of the embedding minus one.
        """
        outside = (z < 1) | (z >= self.config.autoencoder.num_elements)
        if outside.any():
            unknown = z[outside].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

    def encode(self, structures: Structures) -> Posterior:
        """Distribution of the latent vector of each atom of crystals or molecules.

        Raises:
            ValueError: If an element is unknown, or a structure is periodic
                along some of its vectors only.
        """
        self.check_graphs(structures)
        self.is_crystal(structures)
        self.check_elements(structures.z)

        return self.autoencoder.encode(structures)

    def decode_outputs(
        self,
        latent: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Decoded:
        """What the decoder predicts from the latent vectors of the atoms, a Decoded."""
        return self.autoencoder.decode(latent, num_atoms)

    def to_structures(
        self,
        decoded: Decoded,
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"],
    ) -> Structures:
        """Structures read from what the decoder predicts.

        The element of an atom is the atomic number of largest score, 1 for
        the masked atom. A crystal has the predicted cell, its lengths
        multiplied by the cubic root of the number of atoms, and the
        fractional coordinates, wrapped in the cell. A molecule has the
        predicted positions, in a frame which is the identity. A cell which
        is not one (a negative length, angles of no cell) is not refused.

        Args:
            decoded: Predictions of the decoder.
            num_atoms: Number of atoms of each structure.
            crystals: Whether each structure is a crystal, else a molecule.
        """
        z = decoded.atom_types.argmax(dim=-1)
        z = torch.where(z == 0, 1, z)

        lengths = decoded.lattices[:, :3] * num_atoms.unsqueeze(1) ** (1 / 3)
        angles = torch.rad2deg(decoded.lattices[:, 3:])
        frame = torch.eye(3, device=lengths.device).expand(len(num_atoms), 3, 3)
        lattice = torch.where(
            crystals.view(-1, 1, 1), lattice_from_parameters(lengths, angles), frame
        )

        periodic = crystals.repeat_interleave(num_atoms).unsqueeze(1)
        positions = decoded.pos * self.config.autoencoder.position_scale
        x = torch.where(periodic, decoded.frac_coords % 1.0, positions)

        return Structures(
            lattice=lattice,
            x=x,
            z=z,
            num_atoms=num_atoms,
            pbc=crystals.unsqueeze(1).expand(-1, 3).clone(),
        )

    def decode(
        self,
        latent: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"],
    ) -> Structures:
        """Structures decoded from the latent vector of each atom, see to_structures."""
        return self.to_structures(
            self.decode_outputs(latent, num_atoms), num_atoms, crystals
        )

    def forward(self, structures: Structures) -> Structures:
        """Structures encoded and decoded: the mean of each latent vector is decoded."""
        crystals = self.is_crystal(structures)

        return self.decode(self.encode(structures).mean, structures.num_atoms, crystals)

    # diffusion

    def times(self, num_timesteps: int | None = None) -> Float[torch.Tensor, "n_times"]:
        """Times of the sampling, from min_t to 1 in float32, as ADiT.

        Args:
            num_timesteps: Number of times, the one of the configuration by
                default.
        """
        sampling = self.config.sampling
        n = sampling.num_timesteps if num_timesteps is None else num_timesteps

        return torch.linspace(
            sampling.min_t, 1.0, n, device=self.num_atoms_crystals.device
        )

    def denoise(  # pylint: disable=too-many-positional-arguments
        self,
        latent: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"],
        time: Float[torch.Tensor, ""],
        previous: Float[torch.Tensor, "n_atoms latent_dim"] | None = None,
        cfg_scale: float | None = None,
    ) -> Float[torch.Tensor, "n_atoms latent_dim"]:
        """Latent vectors without noise, guided by the dataset of each structure.

        The denoiser is run with the dataset of each structure, crystals or
        molecules, and without dataset; the prediction is the second one plus
        cfg_scale times their difference.

        Args:
            latent: Noisy latent vector of each atom, the atoms of a structure
                together.
            num_atoms: Number of atoms of each structure.
            crystals: Whether each structure is a crystal, else a molecule.
            time: Time of the structures, from min_t (noise) to 1.
            previous: Prediction of the step before, None at the first step.
            cfg_scale: Scale of the guidance, the one of the configuration by
                default.

        Returns:
            Latent vector of each atom without noise.
        """
        if cfg_scale is None:
            cfg_scale = self.config.sampling.cfg_scale

        n_struct = len(num_atoms)
        x, mask = to_dense(latent, num_atoms)
        x_sc = None if previous is None else to_dense(previous, num_atoms)[0]

        denoiser = self.config.denoiser
        dataset = torch.where(crystals, denoiser.crystal_class, denoiser.molecule_class)
        dataset = torch.cat([dataset, torch.zeros_like(dataset)])
        spacegroup = torch.zeros_like(dataset)
        t = time.expand(2 * n_struct)

        def twice(values: torch.Tensor) -> torch.Tensor:
            return torch.cat([values, values])

        out = self.denoiser(
            twice(x),
            t,
            dataset,
            spacegroup,
            twice(mask),
            None if x_sc is None else twice(x_sc),
        )
        conditional, unconditional = out.chunk(2)
        guided = unconditional + cfg_scale * (conditional - unconditional)

        return guided[mask]

    def sample_step(  # pylint: disable=too-many-positional-arguments
        self,
        latent: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"],
        time: Float[torch.Tensor, ""],
        next_time: Float[torch.Tensor, ""],
        previous: Float[torch.Tensor, "n_atoms latent_dim"] | None = None,
        cfg_scale: float | None = None,
    ) -> tuple[
        Float[torch.Tensor, "n_atoms latent_dim"],
        Float[torch.Tensor, "n_atoms latent_dim"],
    ]:
        """One Euler step of the flow, from a time to the next one.

        Args:
            latent: Latent vector of each atom at the time.
            num_atoms: Number of atoms of each structure.
            crystals: Whether each structure is a crystal, else a molecule.
            time: Time of the latent vectors.
            next_time: Time after it, at most 1.
            previous: Prediction of the step before, None at the first step.
            cfg_scale: Scale of the guidance, the one of the configuration by
                default.

        Returns:
            Latent vectors at the next time, and the prediction of the
            denoiser, given to the next step.
        """
        prediction = self.denoise(
            latent, num_atoms, crystals, time, previous, cfg_scale
        )
        velocity = (prediction - latent) / (1 - time)

        return latent + velocity * (next_time - time), prediction

    @staticmethod
    def center_noise(
        noise: Float[torch.Tensor, "n_struct max_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_atoms latent_dim"]:
        """Noise the sampling starts from, from standard normal noise.

        The noise is drawn in the padded layout of the batch and centered over
        each row, padding included, as ADiT does: the noise of a structure
        depends on the largest one of the batch.

        Args:
            noise: Standard normal noise, padded to the largest structure.
            num_atoms: Number of atoms of each structure.

        Returns:
            Centered noise of the atoms, without the padding.

        Examples:
            >>> noise = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]])
            >>> ADiT.center_noise(noise, torch.tensor([1, 2]))
            tensor([[-1.],
                    [-1.],
                    [ 1.]])
        """
        return (noise - noise.mean(dim=1, keepdim=True))[padding_mask(num_atoms)]

    def initial_noise(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        generator: torch.Generator | None = None,
    ) -> Float[torch.Tensor, "n_atoms latent_dim"]:
        """Noise the sampling starts from, drawn as ADiT draws it, see center_noise."""
        shape = (len(num_atoms), int(num_atoms.max()), self.config.denoiser.latent_dim)
        noise = torch.randn(shape, generator=generator, device=num_atoms.device)

        return self.center_noise(noise, num_atoms)

    def sample_latents(  # pylint: disable=too-many-positional-arguments
        self,
        noise: Float[torch.Tensor, "n_atoms latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"],
        num_timesteps: int | None = None,
        cfg_scale: float | None = None,
    ) -> Float[torch.Tensor, "n_atoms latent_dim"]:
        """Latent vectors from noise, by the Euler steps of the flow.

        A step at each time but the last one, then the prediction of the
        denoiser at the time 1: num_timesteps calls of the denoiser, each one
        on the batch twice, for the guidance.

        Args:
            noise: Latent vectors at the time min_t, see initial_noise.
            num_atoms: Number of atoms of each structure.
            crystals: Whether each structure is a crystal, else a molecule.
            num_timesteps: Number of times, the one of the configuration by
                default.
            cfg_scale: Scale of the guidance, the one of the configuration by
                default.
        """
        times = self.times(num_timesteps)
        latent, previous = noise, None

        for time, next_time in zip(times[:-1], times[1:]):
            latent, previous = self.sample_step(
                latent, num_atoms, crystals, time, next_time, previous, cfg_scale
            )

        return self.denoise(latent, num_atoms, crystals, times[-1], previous, cfg_scale)

    def sample_num_atoms(
        self,
        num_samples: int,
        crystals: bool,
        generator: torch.Generator | None = None,
    ) -> Int[torch.Tensor, "num_samples"]:
        """Draw numbers of atoms as the ones of the training set of MP-20 or QM9.

        Args:
            num_samples: Number of structures.
            crystals: If True, as the crystals of MP-20, else as the molecules
                of QM9.
            generator: Generator of the random draws, on the device of the
                model.
        """
        counts = self.num_atoms_crystals if crystals else self.num_atoms_molecules

        return torch.multinomial(
            counts, num_samples, replacement=True, generator=generator
        )

    def sample(  # pylint: disable=too-many-positional-arguments
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        crystals: Bool[torch.Tensor, "n_struct"] | bool = True,
        num_timesteps: int | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Generate crystals or molecules of given numbers of atoms.

        Args:
            num_atoms: Number of atoms of each structure, on the device of the
                model, see sample_num_atoms.
            crystals: Whether each structure is a crystal, else a molecule:
                one value for all of them, or one per structure.
            num_timesteps: Number of times of the flow, the one of the
                configuration by default.
            cfg_scale: Scale of the guidance, the one of the configuration by
                default.
            generator: Generator of the random draws, on the device of
                num_atoms.

        Returns:
            Structures generated, see to_structures.
        """
        if isinstance(crystals, bool):
            crystals = torch.full(
                (len(num_atoms),), crystals, dtype=torch.bool, device=num_atoms.device
            )

        noise = self.initial_noise(num_atoms, generator)
        latent = self.sample_latents(
            noise, num_atoms, crystals, num_timesteps, cfg_scale
        )

        return self.decode(latent, num_atoms, crystals)
