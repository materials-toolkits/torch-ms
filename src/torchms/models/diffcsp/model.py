# Derived from DiffCSP (https://github.com/jiaor17/DiffCSP), Copyright (c) 2023
# Rui Jiao, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the two diffusions of DiffCSP
# (diffcsp/pl_modules/diffusion.py, diffusion_w_type.py and diff_utils.py, and
# the numbers of atoms of scripts/generation.py), for inference only: the
# network on noisy structures, the sampling of a lattice and of fractional
# coordinates for a given composition, and the sampling of the elements with
# them. The losses, the training, the model of the energy and the optimization
# of a property, the datasets and the metrics are removed. The complete graph
# is built by torchms.graph, batches are torchms Structures, the noise of a
# step is an argument, and the normalization of the scores of the coordinates,
# which DiffCSP estimates from random draws, is read from the weights. Names
# of modules and parameters follow DiffCSP to load its weights.
"""DiffCSP, a diffusion model of the structure of a crystal.

A structure is a lattice, a 3 x 3 matrix, and the fractional coordinates of its
atoms. Noise is added to both: normal noise to the lattice, with a cosine
schedule, and noise wrapped around the cell to the coordinates, with a level
which grows geometrically. A network, CSPNet, reads a noisy structure and the
time, and gives the noise of the lattice and the score of the coordinates.

A structure is sampled from a random lattice and random coordinates by 1000
steps, from the last time to the first. A step corrects the coordinates by a
Langevin move, then predicts the lattice and the coordinates of the time
before.

DiffCspModel samples the structure of atoms of given elements. In
DiffCspGenerationModel the element of an atom is a vector with one value per
element, which gets the noise of the lattice and is sampled with the
structure: only the numbers of atoms are given.

Weights are listed in config.yaml: the ones of the authors, trained on MP-20.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Transform
from torchms.data.batching import Structures
from torchms.graph.transforms import AddCompleteGraph, attached_graph
from torchms.models.estimator.config import load_config
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.models.layers.basis import sinusoidal_embedding
from torchms.models.layers.cspnet import CSPNet, CSPNetConfig

# every value of the models is in config.yaml, next to this module
CONFIG = load_config(__file__)
GENERATION = CONFIG["generation"]

SELF_LOOPS: bool = CONFIG["graph"]["self_loops"]

TIMESTEPS: int = CONFIG["model"]["timesteps"]
TIME_DIM: int = CONFIG["model"]["time_dim"]
TIME_MAX_PERIOD: float = CONFIG["model"]["time_max_period"]
BETA_OFFSET: float = CONFIG["model"]["beta_offset"]
BETA_MIN: float = CONFIG["model"]["beta_min"]
BETA_MAX: float = CONFIG["model"]["beta_max"]
SIGMA_BEGIN: float = CONFIG["model"]["sigma_begin"]
SIGMA_END: float = CONFIG["model"]["sigma_end"]

# hyperparameters of the CSPNet of each model, which reads the embedding of
# the time
DECODER = CSPNetConfig(latent_dim=TIME_DIM, **CONFIG["decoder"])
GENERATION_DECODER = CSPNetConfig(
    latent_dim=TIME_DIM, **{**CONFIG["decoder"], **GENERATION["decoder"]}
)
NUM_ELEMENTS: int = DECODER.num_elements

STEP_SIZE: float = CONFIG["sampling"]["step_size"]
GENERATION_STEP_SIZE: float = GENERATION["step_size"]
NUM_ATOMS_DISTRIBUTION: tuple[float, ...] = tuple(GENERATION["num_atoms_distribution"])

# configuration of the training each rewrite is written for, by section
UPSTREAM_ARGS: dict[str, dict[str, object]] = {
    "model": CONFIG["upstream"]["model_args"],
    "model.decoder": CONFIG["upstream"]["decoder_args"],
    "model.beta_scheduler": CONFIG["upstream"]["beta_scheduler_args"],
    "model.sigma_scheduler": CONFIG["upstream"]["sigma_scheduler_args"],
}
GENERATION_UPSTREAM_ARGS: dict[str, dict[str, object]] = {
    **UPSTREAM_ARGS,
    "model": CONFIG["upstream"]["generation_model_args"],
}


class BetaScheduler(nn.Module):  # pylint: disable=abstract-method
    """Noise of the lattice at each time, with a cosine schedule.

    The time 0 is the structure without noise: its values stand for no noise.

    Attributes:
        betas: Variance of the noise added at each time, shape (timesteps + 1,).
        alphas: 1 - betas.
        alphas_cumprod: Product of the alphas up to each time: the part of the
            variance of a noisy lattice which is not noise.
        sigmas: Deviation of the noise a step of the sampling adds at each time.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sigmas: torch.Tensor

    def __init__(self):
        super().__init__()
        steps = torch.linspace(0, TIMESTEPS, TIMESTEPS + 1)
        cumprod = (
            torch.cos(
                (steps / TIMESTEPS + BETA_OFFSET) / (1 + BETA_OFFSET) * math.pi * 0.5
            )
            ** 2
        )
        cumprod = cumprod / cumprod[0]
        betas = torch.clip(1 - cumprod[1:] / cumprod[:-1], BETA_MIN, BETA_MAX)

        betas = F.pad(betas, (1, 0), value=0.0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        variances = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas = F.pad(torch.sqrt(variances), (1, 0), value=0.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)


class SigmaScheduler(nn.Module):  # pylint: disable=abstract-method
    """Noise of the fractional coordinates at each time.

    Attributes:
        sigmas: Level of the noise at each time, in cells, growing
            geometrically, with 0 at the time 0. Shape (timesteps + 1,).
        sigmas_norm: Mean of the squared score of the noise wrapped around the
            cell, at each level. The network gives the score divided by its
            square root. DiffCSP estimates it from random draws when a model is
            created: it is read from the weights, and is 1 before.
    """

    sigmas: torch.Tensor
    sigmas_norm: torch.Tensor

    def __init__(self):
        super().__init__()
        sigmas = torch.exp(
            torch.linspace(
                math.log(SIGMA_BEGIN),
                math.log(SIGMA_END),
                TIMESTEPS,
                dtype=torch.float64,
            )
        ).float()

        self.register_buffer("sigmas", F.pad(sigmas, (1, 0), value=0.0))
        self.register_buffer("sigmas_norm", torch.ones(TIMESTEPS + 1))


class Diffusion(PretrainedModel):  # pylint: disable=abstract-method
    """What the two models of DiffCSP share: network, schedules and steps.

    Args:
        decoder: Hyperparameters of the CSPNet of the model.

    Attributes:
        UPSTREAM_ARGS: Configuration of the training of DiffCSP the model is
            written for, by section.
    """

    UPSTREAM_ARGS: dict[str, dict[str, object]]
    GRAPH_FIELDS = ("graph",)

    def __init__(self, decoder: CSPNetConfig):
        super().__init__()
        self.decoder = CSPNet(decoder)
        self.beta_scheduler = BetaScheduler()
        self.sigma_scheduler = SigmaScheduler()

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint of pytorch-lightning only holds tensors and numbers.
        """
        checkpoint_path, hparams_path = download("diffcsp", name, cache)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{checkpoint_path} is not a checkpoint of DiffCSP")

        with open(hparams_path, "r", encoding="utf-8") as fp:
            hparams = yaml.safe_load(fp)

        self.load_pretrained(checkpoint["state_dict"], hparams)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        hparams: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a CSPDiffusion of DiffCSP.

        Args:
            state_dict: Parameters of the model of DiffCSP, with the buffers of
                its two schedules of noise.
            hparams: Configuration of the training of DiffCSP, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with another configuration: the weights of
                one of the two models are refused by the other one.
        """
        if hparams is not None:
            check_configuration(hparams, self.UPSTREAM_ARGS)

        self.load_parameters(state_dict)

    # graph

    @classmethod
    def transform(cls) -> Transform:
        """Transform attaching the complete graph of each structure.

        An edge goes from each atom to every atom of its structure, itself
        included. The graph only depends on the numbers of atoms.
        """
        return AddCompleteGraph(self_loops=SELF_LOOPS)

    # what a step of the sampling is made of

    def embed_time(
        self, times: Int[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct time_dim"]:
        """Embedding of the time of each structure, read by the network."""
        return sinusoidal_embedding(times, TIME_DIM, TIME_MAX_PERIOD)

    def correct_coordinates(
        self,
        x: Float[torch.Tensor, "n_atoms 3"],
        score: Float[torch.Tensor, "n_atoms 3"],
        time: int,
        noise: Float[torch.Tensor, "n_atoms 3"],
        step_size: float,
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Langevin move of the coordinates at the level of noise of a time.

        Args:
            x: Fractional coordinates at this time.
            score: Score the network gives for them.
            time: Time of the structures.
            noise: Standard normal noise.
            step_size: Step at the lowest level of noise: the step grows as
                the square of the level.

        Returns:
            Corrected coordinates, not wrapped in the cell.
        """
        sigma = self.sigma_scheduler.sigmas[time]
        score = score * torch.sqrt(self.sigma_scheduler.sigmas_norm[time])
        step = step_size * (sigma / SIGMA_BEGIN) ** 2

        return x - step * score + torch.sqrt(2 * step) * noise

    def predict_coordinates(
        self,
        x: Float[torch.Tensor, "n_atoms 3"],
        score: Float[torch.Tensor, "n_atoms 3"],
        time: int,
        noise: Float[torch.Tensor, "n_atoms 3"],
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Coordinates of the time before, from the ones of a time and their score.

        Returns:
            Coordinates at the time before, not wrapped in the cell.
        """
        sigma = self.sigma_scheduler.sigmas[time]
        previous_sigma = self.sigma_scheduler.sigmas[time - 1]
        score = score * torch.sqrt(self.sigma_scheduler.sigmas_norm[time])

        step = sigma**2 - previous_sigma**2
        deviation = torch.sqrt(previous_sigma**2 * step / sigma**2)

        return x - step * score + deviation * noise

    def predict_normal(
        self,
        values: Float[torch.Tensor, "..."],
        found: Float[torch.Tensor, "..."],
        time: int,
        noise: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Values of the time before, for what gets normal noise: the lattice.

        Args:
            values: Noisy values at a time.
            found: Noise the network finds in them.
            time: Time of the structures.
            noise: Standard normal noise.
        """
        alpha = self.beta_scheduler.alphas[time]
        alpha_cumprod = self.beta_scheduler.alphas_cumprod[time]

        return (1.0 / torch.sqrt(alpha)) * (
            values - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * found
        ) + self.beta_scheduler.sigmas[time] * noise


class DiffCspModel(Diffusion):
    """DiffCSP: diffusion of the lattice and of the coordinates of a crystal.

    The model does not build the complete graph its network reads, see
    transform. A step of the sampling builds it itself, see sample_step.

    Examples:
        The network on two structures, at the times 1 and 1000:

        >>> model = DiffCspModel().eval()
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende()]
        ... )
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     lattice_noise, score = model(structures, torch.tensor([1, 1000]))
        >>> lattice_noise.shape, score.shape
        (torch.Size([2, 3, 3]), torch.Size([16, 3]))

        With pretrained weights, `model.sample(structures.z, structures.num_atoms)`
        gives a lattice and coordinates to the atoms of these structures.
    """

    DEFAULT_WEIGHTS = CONFIG["default_weights"]
    UNUSED_PARAMETERS = tuple(CONFIG["unused_parameters"])
    UPSTREAM_ARGS = UPSTREAM_ARGS

    elements: torch.Tensor

    def __init__(self):
        super().__init__(DECODER)

        # row of each atomic number in the embedding of the network, from hydrogen
        self.register_buffer(
            "elements",
            element_table(torch.arange(1, NUM_ELEMENTS + 1)),
            persistent=False,
        )

    # network

    def denoise(
        self, structures: Structures, times: Int[torch.Tensor, "n_struct"]
    ) -> tuple[Float[torch.Tensor, "n_struct 3 3"], Float[torch.Tensor, "n_atoms 3"]]:
        """Noise of the lattice and score of the coordinates of noisy structures.

        Args:
            structures: Noisy structures with their complete graph, see
                transform. A lattice is any 3 x 3 matrix.
            times: Time of each structure, from 1 to the number of steps of
                the diffusion: how much noise it has.

        Returns:
            Normal noise of each lattice, shape (n_struct, 3, 3), and score of
            the fractional coordinates of each atom, over the square root of
            sigma_scheduler.sigmas_norm at its time, shape (n_atoms, 3).

        Raises:
            ValueError: If a structure has an element after fermium, or if
                the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        element = element_index(structures.z, self.elements)

        lattice_noise, score, _ = self.decoder(
            self.embed_time(times), element, graph, structures.x, structures.lattice
        )

        return lattice_noise, score

    def forward(
        self, structures: Structures, times: Int[torch.Tensor, "n_struct"]
    ) -> tuple[Float[torch.Tensor, "n_struct 3 3"], Float[torch.Tensor, "n_atoms 3"]]:
        """Noise of the lattice and score of the coordinates, see denoise."""
        return self.denoise(structures, times)

    # sampling

    def sample_step(
        self,
        structures: Structures,
        time: int,
        corrector_noise: Float[torch.Tensor, "n_atoms 3"],
        lattice_noise: Float[torch.Tensor, "n_struct 3 3"],
        predictor_noise: Float[torch.Tensor, "n_atoms 3"],
        step_size: float = STEP_SIZE,
    ) -> Structures:
        """One step of the sampling, from a time to the time before.

        The coordinates are first corrected by a Langevin move at the level of
        noise of the time. Then the lattice and the coordinates of the time
        before are predicted from the corrected structure. The complete graph
        is built here, on the structures given.

        Args:
            structures: Noisy structures at this time, without graph.
            time: Time of the structures, from the number of steps of the
                diffusion down to 1.
            corrector_noise: Standard normal noise of the correction of the
                coordinates, shape (n_atoms, 3). Zero at the time 1.
            lattice_noise: Standard normal noise of the prediction of the
                lattice, shape (n_struct, 3, 3). Zero at the time 1.
            predictor_noise: Standard normal noise of the prediction of the
                coordinates, shape (n_atoms, 3). Zero at the time 1.
            step_size: Step of the correction at the lowest level of noise:
                the step grows as the square of the level.

        Returns:
            Structures at the time before, without graph, their coordinates
            in the cell.
        """
        graph = attached_graph(self.transform()(structures), self)
        element = element_index(structures.z, self.elements)

        times = torch.full(
            (len(structures.num_atoms),), time, device=structures.device()
        )
        latent = self.embed_time(times)
        lattice, x = structures.lattice, structures.x

        # corrector: the lattice is kept
        _, score, _ = self.decoder(latent, element, graph, x, lattice)
        x = self.correct_coordinates(x, score, time, corrector_noise, step_size)

        # predictor
        found, score, _ = self.decoder(latent, element, graph, x, lattice)
        x = self.predict_coordinates(x, score, time, predictor_noise)
        lattice = self.predict_normal(lattice, found, time, lattice_noise)

        return Structures(
            lattice=lattice,
            x=x % 1.0,
            z=structures.z,
            num_atoms=structures.num_atoms,
        )

    def sample(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        step_size: float = STEP_SIZE,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Sample the lattice and the coordinates of structures of given atoms.

        The lattices start from normal noise and the coordinates from uniform
        ones, and follow every step of the diffusion, from the last time: two
        calls of the network per step.

        Args:
            z: Atomic number of each atom, the atoms of a structure together,
                shape (n_atoms,), on the device of the model.
            num_atoms: Number of atoms of each structure, shape (n_struct,).
            step_size: Step of the corrector at the lowest level of noise.
            generator: Generator of the random draws, on the device of z.

        Returns:
            Structures sampled, without graph. Their lattice is not reduced.

        Raises:
            ValueError: If an atom has an element after fermium.
        """
        device = z.device
        element_index(z, self.elements)

        lattice = torch.randn(
            (len(num_atoms), 3, 3), generator=generator, device=device
        )
        x = torch.rand((len(z), 3), generator=generator, device=device)
        structures = Structures(lattice=lattice, x=x, z=z, num_atoms=num_atoms)

        for time in range(TIMESTEPS, 0, -1):
            # the last step adds no noise
            scale = 1.0 if time > 1 else 0.0
            corrector_noise, lattice_noise, predictor_noise = (
                scale * torch.randn(shape, generator=generator, device=device)
                for shape in (x.shape, lattice.shape, x.shape)
            )

            structures = self.sample_step(
                structures,
                time,
                corrector_noise,
                lattice_noise,
                predictor_noise,
                step_size,
            )

        return structures


@dataclass(frozen=True)
class GenerationNoise:
    """Standard normal noise of one step of the sampling with the elements.

    Every field is zero at the time 1.

    Args:
        corrector: Noise of the correction of the coordinates, shape
            (n_atoms, 3).
        lattice: Noise of the prediction of the lattice, shape (n_struct, 3, 3).
        elements: Noise of the prediction of the elements, shape
            (n_atoms, num_elements).
        predictor: Noise of the prediction of the coordinates, shape
            (n_atoms, 3).
    """

    corrector: Float[torch.Tensor, "n_atoms 3"]
    lattice: Float[torch.Tensor, "n_struct 3 3"]
    elements: Float[torch.Tensor, "n_atoms num_elements"]
    predictor: Float[torch.Tensor, "n_atoms 3"]


class DiffCspGenerationModel(Diffusion):
    """DiffCSP which also generates the elements of the atoms.

    The element of an atom is a vector of one value per element, from hydrogen,
    which is 1 for its element and 0 elsewhere without noise. It gets the
    normal noise of the lattice, and the network gives this noise for each
    atom. The element of an atom is the largest value of its vector. The
    atomic numbers of the structures given to the model are not read.

    The model does not build the complete graph its network reads, see
    transform. A step of the sampling builds it itself, see sample_step.

    Examples:
        The network on two structures without noise, at the times 1 and 1000:

        >>> model = DiffCspGenerationModel().eval()
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende()]
        ... )
        >>> structures = model.transform()(structures)
        >>> elements = model.element_vectors(structures.z)
        >>> with torch.no_grad():
        ...     lattice_noise, score, element_noise = model(
        ...         structures, elements, torch.tensor([1, 1000])
        ...     )
        >>> lattice_noise.shape, score.shape, element_noise.shape
        (torch.Size([2, 3, 3]), torch.Size([16, 3]), torch.Size([16, 100]))

        With pretrained weights, `model.sample(model.sample_num_atoms(8))`
        generates 8 structures.
    """

    DEFAULT_WEIGHTS = GENERATION["default_weights"]
    UNUSED_PARAMETERS = tuple(GENERATION["unused_parameters"])
    UPSTREAM_ARGS = GENERATION_UPSTREAM_ARGS

    num_atoms_distribution: torch.Tensor

    def __init__(self):
        super().__init__(GENERATION_DECODER)

        # probability of each number of atoms, from 0
        self.register_buffer(
            "num_atoms_distribution",
            torch.tensor(NUM_ATOMS_DISTRIBUTION),
            persistent=False,
        )

    # elements

    def element_vectors(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms num_elements"]:
        """Vector of the element of each atom, without noise.

        Args:
            z: Atomic number of each atom.

        Returns:
            1 for the element of the atom and 0 for the other ones, from
            hydrogen, shape (n_atoms, num_elements).

        Raises:
            ValueError: If an atom has an element after fermium.
        """
        if ((z < 1) | (z > NUM_ELEMENTS)).any():
            unknown = z[(z < 1) | (z > NUM_ELEMENTS)].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

        return F.one_hot(z - 1, NUM_ELEMENTS).float()  # pylint: disable=not-callable

    def atomic_numbers(
        self, elements: Float[torch.Tensor, "n_atoms num_elements"]
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Atomic number of each atom: the largest value of its vector."""
        return elements.argmax(dim=-1) + 1

    # network

    def denoise(
        self,
        structures: Structures,
        elements: Float[torch.Tensor, "n_atoms num_elements"],
        times: Int[torch.Tensor, "n_struct"],
    ) -> tuple[
        Float[torch.Tensor, "n_struct 3 3"],
        Float[torch.Tensor, "n_atoms 3"],
        Float[torch.Tensor, "n_atoms num_elements"],
    ]:
        """Noise of the lattice and of the elements, and score of the coordinates.

        Args:
            structures: Noisy structures with their complete graph, see
                transform. A lattice is any 3 x 3 matrix. Their atomic numbers
                are not read.
            elements: Noisy vector of the element of each atom, see
                element_vectors.
            times: Time of each structure, from 1 to the number of steps of
                the diffusion: how much noise it has.

        Returns:
            Normal noise of each lattice, shape (n_struct, 3, 3), score of the
            fractional coordinates of each atom, over the square root of
            sigma_scheduler.sigmas_norm at its time, shape (n_atoms, 3), and
            normal noise of the vector of each atom, shape
            (n_atoms, num_elements).

        Raises:
            ValueError: If the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)

        lattice_noise, score, element_noise = self.decoder(
            self.embed_time(times), elements, graph, structures.x, structures.lattice
        )
        assert element_noise is not None

        return lattice_noise, score, element_noise

    def forward(
        self,
        structures: Structures,
        elements: Float[torch.Tensor, "n_atoms num_elements"],
        times: Int[torch.Tensor, "n_struct"],
    ) -> tuple[
        Float[torch.Tensor, "n_struct 3 3"],
        Float[torch.Tensor, "n_atoms 3"],
        Float[torch.Tensor, "n_atoms num_elements"],
    ]:
        """Noise of the lattice and of the elements and score, see denoise."""
        return self.denoise(structures, elements, times)

    # sampling

    def sample_step(
        self,
        structures: Structures,
        elements: Float[torch.Tensor, "n_atoms num_elements"],
        time: int,
        noise: GenerationNoise,
        step_size: float = GENERATION_STEP_SIZE,
    ) -> tuple[Structures, Float[torch.Tensor, "n_atoms num_elements"]]:
        """One step of the sampling, from a time to the time before.

        The coordinates are first corrected by a Langevin move at the level of
        noise of the time. Then the lattice, the elements and the coordinates
        of the time before are predicted from the corrected structure. The
        complete graph is built here, on the structures given.

        Args:
            structures: Noisy structures at this time, without graph. Their
                atomic numbers are not read.
            elements: Noisy vector of the element of each atom at this time.
            time: Time of the structures, from the number of steps of the
                diffusion down to 1.
            noise: Standard normal noise of the step.
            step_size: Step of the correction at the lowest level of noise:
                the step grows as the square of the level.

        Returns:
            Structures at the time before, without graph, their coordinates
            in the cell and their atomic numbers read from the vectors, and
            the vector of the element of each atom at the time before.
        """
        graph = attached_graph(self.transform()(structures), self)

        times = torch.full(
            (len(structures.num_atoms),), time, device=structures.device()
        )
        latent = self.embed_time(times)
        lattice, x = structures.lattice, structures.x

        # corrector: the lattice and the elements are kept
        _, score, _ = self.decoder(latent, elements, graph, x, lattice)
        x = self.correct_coordinates(x, score, time, noise.corrector, step_size)

        # predictor
        found, score, found_elements = self.decoder(latent, elements, graph, x, lattice)
        assert found_elements is not None
        x = self.predict_coordinates(x, score, time, noise.predictor)
        lattice = self.predict_normal(lattice, found, time, noise.lattice)
        elements = self.predict_normal(elements, found_elements, time, noise.elements)

        moved = Structures(
            lattice=lattice,
            x=x % 1.0,
            z=self.atomic_numbers(elements),
            num_atoms=structures.num_atoms,
        )

        return moved, elements

    def sample_num_atoms(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> Int[torch.Tensor, "num_samples"]:
        """Draw numbers of atoms as the ones of the training set of MP-20.

        Args:
            num_samples: Number of structures.
            generator: Generator of the random draws, on the device of the
                model.

        Returns:
            Number of atoms of each structure, from 1 to 20, shape
            (num_samples,).
        """
        return torch.multinomial(
            self.num_atoms_distribution,
            num_samples,
            replacement=True,
            generator=generator,
        )

    def sample(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        step_size: float = GENERATION_STEP_SIZE,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Sample structures of given numbers of atoms, with their elements.

        The lattices and the vectors of the elements start from normal noise
        and the coordinates from uniform ones, and follow every step of the
        diffusion, from the last time: two calls of the network per step.

        Args:
            num_atoms: Number of atoms of each structure, shape (n_struct,),
                on the device of the model. See sample_num_atoms.
            step_size: Step of the corrector at the lowest level of noise.
            generator: Generator of the random draws, on the device of
                num_atoms.

        Returns:
            Structures sampled, without graph. Their lattice is not reduced.
        """
        device = num_atoms.device
        n_atoms = int(num_atoms.sum())

        lattice = torch.randn(
            (len(num_atoms), 3, 3), generator=generator, device=device
        )
        x = torch.rand((n_atoms, 3), generator=generator, device=device)
        elements = torch.randn(
            (n_atoms, NUM_ELEMENTS), generator=generator, device=device
        )
        structures = Structures(
            lattice=lattice,
            x=x,
            z=self.atomic_numbers(elements),
            num_atoms=num_atoms,
        )

        for time in range(TIMESTEPS, 0, -1):
            # the last step adds no noise
            scale = 1.0 if time > 1 else 0.0
            noise = GenerationNoise(
                *(
                    scale * torch.randn(shape, generator=generator, device=device)
                    for shape in (x.shape, lattice.shape, elements.shape, x.shape)
                )
            )

            structures, elements = self.sample_step(
                structures, elements, time, noise, step_size
            )

        return structures
