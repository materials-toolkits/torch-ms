# Derived from cdvae (https://github.com/txie-93/cdvae), Copyright (c) 2021
# Tian Xie, Xiang Fu, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the model of cdvae
# (cdvae/pl_modules/model.py and decoder.py), for inference only: encoding,
# decoding of the number of atoms, of the cell and of the composition, and
# generation by annealed Langevin dynamics. The losses, the training, the
# prediction of a property from the latent vector, the optimization of a
# property, the datasets and the conversion from pymatgen are removed. Graphs are built by
# torchms.graph, batches are torchms Structures, and the composition of a
# batch is drawn without a loop over its structures. Names of modules and
# parameters follow cdvae to load its weights.
"""Crystal Diffusion Variational Autoencoder (CDVAE).

A structure is encoded by a DimeNet++ into the mean and the variance of a
latent vector. From a latent vector, three small networks give the number of
atoms, the cell and the composition of a structure. Its atoms, placed at
random, are then moved by an annealed Langevin dynamics: at each step a
GemNet-T reads the noisy structure and the latent vector and gives the
direction each atom should move in and the element it should have.

Weights are listed in config.yaml. The authors of cdvae publish none for this
model: the ones listed were trained by a third party.
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.models.estimator.config import load_config
from torchms.models.estimator.reader import read_pickle
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.graph.geometry import cartesian, lattice_from_parameters, triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddBonds,
    AddLineGraph,
    AddRadiusGraph,
    AddTriplets,
    KeepNearestNeighbors,
    KeepNearImages,
    MirrorEdges,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.layers.dimenet import DimeNetConfig, DimeNetPlusPlus
from torchms.models.layers.gemnet import GemNetConfig, GemNetT

# every value of the model is in config.yaml, next to this module
CONFIG = load_config(__file__)

UNUSED_PARAMETERS: tuple[str, ...] = tuple(CONFIG["unused_parameters"])

RADIUS: float = CONFIG["graph"]["decoder"]["radius"]
MAX_NEIGHBORS: int = CONFIG["graph"]["decoder"]["max_neighbors"]
MAX_IMAGE_SHIFT: int = CONFIG["graph"]["decoder"]["max_image_shift"]

LATENT_DIM: int = CONFIG["model"]["latent_dim"]
HIDDEN_DIM: int = CONFIG["model"]["hidden_dim"]
NUM_HIDDEN_LAYERS: int = CONFIG["model"]["num_hidden_layers"]
MAX_ATOMS: int = CONFIG["model"]["max_atoms"]
NUM_ELEMENTS: int = CONFIG["model"]["num_elements"]
SIGMA_BEGIN: float = CONFIG["model"]["sigma_begin"]
SIGMA_END: float = CONFIG["model"]["sigma_end"]
NUM_NOISE_LEVELS: int = CONFIG["model"]["num_noise_levels"]

# hyperparameters of the DimeNet++ encoder and of the GemNet-T decoder
ENCODER = DimeNetConfig(out_channels=LATENT_DIM, **CONFIG["encoder"])
DECODER = GemNetConfig(
    latent_dim=LATENT_DIM,
    num_elements=NUM_ELEMENTS,
    radius=RADIUS,
    **CONFIG["decoder"],
)

STEPS_PER_LEVEL: int = CONFIG["sampling"]["steps_per_level"]
STEP_SIZE: float = CONFIG["sampling"]["step_size"]
MIN_SIGMA: float = CONFIG["sampling"]["min_sigma"]

# buffers of cdvae with another name here, and constants absent from its weights
RENAMED_PARAMETERS: dict[str, str] = CONFIG["upstream"]["renamed_parameters"]
CONSTANT_BUFFERS: tuple[str, ...] = tuple(CONFIG["upstream"]["constant_buffers"])

# configuration of the training the rewrite is written for, by section
UPSTREAM_ARGS: dict[str, dict[str, object]] = {
    "model": CONFIG["upstream"]["model_args"],
    "model.encoder": CONFIG["upstream"]["encoder_args"],
    "model.decoder": CONFIG["upstream"]["decoder_args"],
    "data": CONFIG["upstream"]["data_args"],
}

# what the encoder and the decoder read on the structures
ENCODER_FIELDS = ("graph", "line_graph")
DECODER_FIELDS = ("graph", "triplets", "bonds")


def build_mlp(output_size: int) -> nn.Sequential:
    """Network from a latent vector to one of the quantities decoded from it."""
    layers: list[nn.Module] = [nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU()]
    for _ in range(NUM_HIDDEN_LAYERS - 1):
        layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()]
    layers.append(nn.Linear(HIDDEN_DIM, output_size))

    return nn.Sequential(*layers)


class GemNetTDecoder(nn.Module):
    """Move and element of each atom of noisy structures, from their latent vector."""

    def __init__(self):
        super().__init__()
        self.gemnet = GemNetT(DECODER)
        self.fc_atom = nn.Linear(DECODER.hidden_dim, NUM_ELEMENTS)

    def forward(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        element: Int[torch.Tensor, "n_atoms"],
        graph: Graph,
        angles: EdgeTuples,
        bonds: EdgeTuples,
        vectors: Float[torch.Tensor, "n_edges 3"],
        cosine: Float[torch.Tensor, "n_triplets"],
    ) -> tuple[
        Float[torch.Tensor, "n_atoms 3"], Float[torch.Tensor, "n_atoms num_elements"]
    ]:
        """Vector in Å and scores of the elements of each atom, see GemNetT."""
        h, move = self.gemnet(latent, element, graph, angles, bonds, vectors, cosine)

        return move, self.fc_atom(h)


def sample_composition(
    probability: Float[torch.Tensor, "n_struct num_elements"],
    num_atoms: Int[torch.Tensor, "n_struct"],
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, "n_atoms"]:
    """Draw the elements of the atoms of structures from their compositions.

    An element with a probability p has round(p n) of the n atoms of a
    structure. If atoms are missing, they are drawn from what the rounding
    left out; if there are too many, some are removed at random. The atoms of
    a structure are in a random order.

    Args:
        probability: Probability of each element in each structure, from
            hydrogen, shape (n_struct, num_elements).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        generator: Generator of the random draws.

    Returns:
        Atomic number of each atom, the atoms of a structure together, shape
        (n_atoms,).

    Examples:
        >>> probability = torch.tensor([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        >>> z = sample_composition(probability, torch.tensor([4, 2]))
        >>> z[:4].sort().values, z[4:]
        (tensor([1, 1, 3, 3]), tensor([2, 2]))
    """
    device = probability.device
    n_struct, n_elements = probability.shape
    wanted = num_atoms.unsqueeze(1).clamp(min=1)

    count = torch.round(probability * wanted).long()

    # atoms missing after the rounding, drawn from what it left out
    missing = (num_atoms - count.sum(dim=1)).clamp(min=0)
    most = int(missing.max()) if n_struct > 0 else 0
    if most > 0:
        left = (probability - count / wanted).clamp(min=0.0)
        # a structure which misses no atom keeps none of its draws
        left = torch.where((missing > 0).unsqueeze(1), left, torch.ones_like(left))
        drawn = torch.multinomial(left, most, replacement=True, generator=generator)
        kept = torch.arange(most, device=device) < missing.unsqueeze(1)
        count = count + (F.one_hot(drawn, n_elements) * kept.unsqueeze(-1)).sum(dim=1)

    # one candidate atom per unit of count, by structure then by element
    pair = torch.arange(n_struct * n_elements, device=device)
    owner = (pair // n_elements).repeat_interleave(count.flatten())
    z = (pair % n_elements + 1).repeat_interleave(count.flatten())

    # candidates in a random order in each structure: the first ones are kept
    key = torch.rand(len(z), generator=generator, device=device)
    by_key = torch.argsort(key)
    order = by_key[torch.argsort(owner[by_key], stable=True)]

    candidate_ptr = F.pad(torch.cumsum(count.sum(dim=1), dim=0), (1, 0), value=0)
    rank = torch.arange(len(z), device=device) - candidate_ptr[owner[order]]

    return z[order][rank < num_atoms[owner[order]]]


class CDVAE(PretrainedModel):
    """Crystal Diffusion Variational Autoencoder.

    The model does not build the graph of the bonds its encoder reads, see
    transform. Its decoder reads another graph, built at each step of the
    dynamics by decoder_transform.

    Examples:
        Latent vectors of two structures, with a graph of their atoms closer
        than 2 Å in place of the graph of CrystalNN:

        >>> model = CDVAE().eval()
        >>> structures = Structures.collate([Structures.diamond(), Structures.water()])
        >>> structures = model.transform()(AddRadiusGraph(2.0)(structures))
        >>> with torch.no_grad():
        ...     mean, log_variance = model(structures)
        >>> mean.shape
        torch.Size([2, 256])

        With pretrained weights, `model.sample(8)` generates 8 structures.
    """

    DEFAULT_WEIGHTS = CONFIG["default_weights"]
    UNUSED_PARAMETERS = UNUSED_PARAMETERS
    GRAPH_FIELDS = ENCODER_FIELDS

    sigmas: torch.Tensor
    lattice_means: torch.Tensor
    lattice_stds: torch.Tensor
    encoder_elements: torch.Tensor
    decoder_elements: torch.Tensor

    def __init__(self):
        super().__init__()
        self.encoder = DimeNetPlusPlus(ENCODER)
        self.decoder = GemNetTDecoder()

        self.fc_mu = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.fc_var = nn.Linear(LATENT_DIM, LATENT_DIM)

        self.fc_num_atoms = build_mlp(MAX_ATOMS + 1)
        self.fc_lattice = build_mlp(6)
        self.fc_composition = build_mlp(NUM_ELEMENTS)

        # noise levels of the positions in Å, from the largest
        self.register_buffer(
            "sigmas",
            torch.exp(
                torch.linspace(
                    math.log(SIGMA_BEGIN), math.log(SIGMA_END), NUM_NOISE_LEVELS
                )
            ),
        )

        # mean and deviation of the lengths, over the cube root of the number
        # of atoms, and of the angles of the cells of the training set
        self.register_buffer("lattice_means", torch.zeros(6))
        self.register_buffer("lattice_stds", torch.ones(6))

        # row of each atomic number in the embedding of the encoder, indexed
        # by the atomic number, and of the decoder, from hydrogen
        self.register_buffer(
            "encoder_elements",
            element_table(torch.arange(ENCODER.num_elements)),
            persistent=False,
        )
        self.register_buffer(
            "decoder_elements",
            element_table(torch.arange(1, NUM_ELEMENTS + 1)),
            persistent=False,
        )

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint and the scaler of the cells are pickles of objects of
        pytorch-lightning, omegaconf and cdvae: they are read without
        importing or running anything, see torchms.models.estimator.reader.
        """
        checkpoint_path, scaler_path, hparams_path = download("cdvae", name, cache)

        checkpoint = read_pickle(checkpoint_path)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{checkpoint_path} is not a checkpoint of cdvae")

        scaler = getattr(read_pickle(scaler_path), "state", None)
        if not isinstance(scaler, dict) or not {"means", "stds"} <= set(scaler):
            raise InvalidWeights(f"{scaler_path} is not a scaler of cdvae")

        with open(hparams_path, "r", encoding="utf-8") as fp:
            hparams = yaml.safe_load(fp)

        self.load_pretrained(
            checkpoint["state_dict"], scaler["means"], scaler["stds"], hparams
        )

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        lattice_means: Float[torch.Tensor, "6"],
        lattice_stds: Float[torch.Tensor, "6"],
        hparams: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a CDVAE of cdvae.

        Args:
            state_dict: Parameters of the model of cdvae.
            lattice_means: Means of its scaler of the cells.
            lattice_stds: Deviations of its scaler of the cells.
            hparams: Configuration of the training of cdvae, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with another configuration.
        """
        if hparams is not None:
            check_configuration(hparams, UPSTREAM_ARGS)

        parameters: dict[str, torch.Tensor] = {}
        for name, values in state_dict.items():
            for old, new in RENAMED_PARAMETERS.items():
                if name.endswith(old):
                    name = name[: -len(old)] + new
            parameters[name] = values

        own = self.state_dict()
        parameters.update({name: own[name] for name in CONSTANT_BUFFERS})
        parameters["lattice_means"] = lattice_means
        parameters["lattice_stds"] = lattice_stds

        self.load_parameters(parameters)

    # graphs

    @classmethod
    def transform(cls) -> Transform:
        """Transform of the encoder, for structures with a graph of their bonds.

        The encoder was trained on the bonds found by CrystalNN of pymatgen,
        which torchms does not build: attach this graph to the structures,
        or another graph of the bonds, on which the weights were not trained.
        The transform attaches its line graph: the pairs of bonds i → j, j → k
        where k is another atom than i.
        """
        return Compose([AddLineGraph(backtrack=True, same_atom=False)])

    @classmethod
    def decoder_transform(cls) -> Transform:
        """Transform attaching the graph of the decoder, its triplets and its bonds.

        An atom is connected to its 20 nearest neighbours closer than 7 Å,
        among the atoms of its cell and of the 26 cells around it, its own
        images included. The graph is then made symmetric from the edges
        towards an atom of lower index. Among neighbours at the same distance,
        the ones which are kept are not the ones of cdvae: this does not
        happen for atoms moved by the dynamics.
        """
        return Compose(
            [
                AddRadiusGraph(RADIUS, self_images=True),
                KeepNearImages(MAX_IMAGE_SHIFT),
                KeepNearestNeighbors(MAX_NEIGHBORS),
                MirrorEdges(),
                AddTriplets(),
                AddBonds(),
            ]
        )

    def check_decoder_graphs(self, structures: Structures) -> None:
        """Check that structures have what the decoder reads on them.

        Raises:
            ValueError: If one of them is not attached.
        """
        missing = [name for name in DECODER_FIELDS if getattr(structures, name) is None]
        if missing:
            name = type(self).__name__
            raise ValueError(
                f"the decoder of {name} needs structures with their "
                f"{', '.join(missing)}: apply {name}.decoder_transform() to the "
                "structures"
            )

    # encoder

    def encode(self, structures: Structures) -> tuple[
        Float[torch.Tensor, "n_struct latent_dim"],
        Float[torch.Tensor, "n_struct latent_dim"],
    ]:
        """Mean and logarithm of the variance of the latent vector of each structure.

        Args:
            structures: Structures with a graph of their bonds and its line
                graph, see transform.

        Raises:
            ValueError: If a structure has an element after plutonium, or if
                the structures have no graph or no line graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        pairs = attached_tuples(structures, "line_graph", self)
        z = element_index(structures.z, self.encoder_elements)

        vectors = edges_vectors(graph)
        # cosine of the angle between the two bonds of a pair, as vectors
        cosine = triplet_cosines(vectors, pairs)

        hidden = self.encoder(z, graph, pairs, vectors.norm(dim=-1), cosine)

        return self.fc_mu(hidden), self.fc_var(hidden)

    def forward(self, structures: Structures) -> tuple[
        Float[torch.Tensor, "n_struct latent_dim"],
        Float[torch.Tensor, "n_struct latent_dim"],
    ]:
        """Mean and logarithm of the variance of the latent vectors, see encode."""
        return self.encode(structures)

    # quantities decoded from a latent vector

    def predict_num_atoms(
        self, latent: Float[torch.Tensor, "n_struct latent_dim"]
    ) -> Float[torch.Tensor, "n_struct max_atoms+1"]:
        """Scores of each number of atoms, from 0 to the largest one."""
        return self.fc_num_atoms(latent)

    def predict_lattice(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> tuple[Float[torch.Tensor, "n_struct 3"], Float[torch.Tensor, "n_struct 3"]]:
        """Lengths in Å and angles in degrees of the cell of each structure.

        Args:
            latent: Latent vector of each structure.
            num_atoms: Number of atoms of each structure: the lengths grow as
                its cube root.
        """
        scaled = self.fc_lattice(latent) * self.lattice_stds + self.lattice_means
        lengths = scaled[:, :3] * num_atoms.unsqueeze(1).to(scaled.dtype) ** (1 / 3)

        return lengths, scaled[:, 3:]

    def predict_composition(
        self, latent: Float[torch.Tensor, "n_struct latent_dim"]
    ) -> Float[torch.Tensor, "n_struct num_elements"]:
        """Scores of each element in the composition of a structure, from hydrogen."""
        return self.fc_composition(latent)

    # decoder

    def denoise(
        self, latent: Float[torch.Tensor, "n_struct latent_dim"], structures: Structures
    ) -> tuple[
        Float[torch.Tensor, "n_atoms 3"], Float[torch.Tensor, "n_atoms num_elements"]
    ]:
        """Move and element of each atom of noisy structures.

        Args:
            latent: Latent vector of each structure.
            structures: Noisy structures with the graph of the decoder, see
                decoder_transform.

        Returns:
            Vector in Å towards the position without noise, times the level of
            the noise, shape (n_atoms, 3), and scores of each element, from
            hydrogen, shape (n_atoms, num_elements).

        Raises:
            ValueError: If a structure has an element after fermium, or if
                the structures do not have the graph of the decoder.
        """
        self.check_decoder_graphs(structures)
        graph = attached_graph(structures, self)
        angles = attached_tuples(structures, "triplets", self)
        bonds = attached_tuples(structures, "bonds", self)
        element = element_index(structures.z, self.decoder_elements)

        vectors = edges_vectors(graph)
        cosine = triplet_cosines(vectors, angles)

        return self.decoder(latent, element, graph, angles, bonds, vectors, cosine)

    def langevin_step(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        structures: Structures,
        sigma: Float[torch.Tensor, ""] | float,
        noise: Float[torch.Tensor, "n_atoms 3"],
        step_size: float = STEP_SIZE,
        update_elements: bool = True,
    ) -> Structures:
        """One step of the annealed Langevin dynamics.

        The graph of the decoder is built here, on the structures given.

        Args:
            latent: Latent vector of each structure.
            structures: Noisy structures, without graph.
            sigma: Level of the noise, in Å.
            noise: Standard normal noise of the step, shape (n_atoms, 3).
            step_size: Step at the lowest level of noise: the step grows as
                the square of the level.
            update_elements: If True, each atom takes the element with the
                best score of the decoder.

        Returns:
            Structures with the atoms moved, without graph.
        """
        move, scores = self.denoise(latent, self.decoder_transform()(structures))

        step = step_size * (sigma / self.sigmas[-1]) ** 2
        positions = (
            structures.positions() + step * move / sigma + noise * (2 * step) ** 0.5
        )

        # fractional coordinates, with the pseudo-inverse as cdvae: the
        # decoded cell can be flat
        inverse = torch.linalg.pinv(structures.lattice)
        x = cartesian(positions, inverse, structures.batch) % 1.0

        return Structures(
            lattice=structures.lattice,
            x=x,
            z=scores.argmax(dim=1) + 1 if update_elements else structures.z,
            num_atoms=structures.num_atoms,
        )

    def generate(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        num_atoms: Int[torch.Tensor, "n_struct"] | None = None,
        z: Int[torch.Tensor, "n_atoms"] | None = None,
        steps_per_level: int = STEPS_PER_LEVEL,
        step_size: float = STEP_SIZE,
        min_sigma: float = MIN_SIGMA,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Decode structures from latent vectors by annealed Langevin dynamics.

        The number of atoms, the cell and the composition are decoded from the
        latent vector. The atoms start at random positions and are moved, for
        each level of noise from the largest, by steps_per_level steps.

        Args:
            latent: Latent vector of each structure.
            num_atoms: Number of atoms of each structure. Decoded from the
                latent vector by default.
            z: Atomic number of each atom, kept during the dynamics. By
                default they are drawn from the decoded composition, then
                updated at each step.
            steps_per_level: Number of steps at each level of noise.
            step_size: Step at the lowest level of noise.
            min_sigma: The dynamics stops before the first level of noise
                below it, in Å.
            generator: Generator of the random draws, on the device of latent.

        Returns:
            Structures generated, without graph.
        """
        device = latent.device

        if num_atoms is None:
            num_atoms = self.predict_num_atoms(latent).argmax(dim=-1)
        lattice = lattice_from_parameters(*self.predict_lattice(latent, num_atoms))

        update_elements = z is None
        if z is None:
            composition = F.softmax(self.predict_composition(latent), dim=-1)
            z = sample_composition(composition, num_atoms, generator)

        x = torch.rand((len(z), 3), generator=generator, device=device)
        structures = Structures(lattice=lattice, x=x, z=z, num_atoms=num_atoms)

        for sigma in self.sigmas:
            if sigma < min_sigma:
                break

            for _ in range(steps_per_level):
                noise = torch.randn(x.shape, generator=generator, device=device)
                structures = self.langevin_step(
                    latent, structures, sigma, noise, step_size, update_elements
                )

        return structures

    def sample(
        self,
        num_samples: int,
        steps_per_level: int = STEPS_PER_LEVEL,
        step_size: float = STEP_SIZE,
        min_sigma: float = MIN_SIGMA,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Generate structures from latent vectors drawn from a standard normal.

        Args:
            num_samples: Number of structures.
            steps_per_level: Number of steps at each level of noise.
            step_size: Step at the lowest level of noise.
            min_sigma: The dynamics stops before the first level of noise
                below it, in Å.
            generator: Generator of the random draws, on the device of the
                model.
        """
        latent = torch.randn(
            (num_samples, LATENT_DIM), generator=generator, device=self.sigmas.device
        )

        return self.generate(
            latent,
            steps_per_level=steps_per_level,
            step_size=step_size,
            min_sigma=min_sigma,
            generator=generator,
        )
