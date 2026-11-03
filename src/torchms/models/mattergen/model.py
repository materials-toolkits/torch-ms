# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license. Its diffusion of the elements is adapted
# from D3PM of Google Research and its predictor and corrector from
# score_sde_pytorch, both under the Apache License 2.0 (see LICENSE in this
# directory).
#
# Changes: this file is a rewrite of the base model of MatterGen, which
# generates crystals without condition, and of its model conditioned on a
# chemical system, for inference only: the denoiser (mattergen/denoiser.py,
# mattergen/adapter.py, the embedding of a chemical system of
# mattergen/property_embeddings.py), the three diffusions of the lattice, of the
# fractional coordinates and of the elements
# (mattergen/common/diffusion/corruption.py, mattergen/diffusion/corruption,
# mattergen/diffusion/d3pm/d3pm.py), their predictors and correctors
# (mattergen/common/diffusion/predictors_correctors.py,
# mattergen/diffusion/sampling, mattergen/diffusion/wrapped) and the numbers
# of atoms of mattergen/common/data/num_atoms_distribution.py, with the
# guidance of mattergen/diffusion/sampling/classifier_free_guidance.py. The
# training, the losses, the embeddings of the other properties, the
# fine-tuning, the prediction of a structure from its composition, the
# datasets and the evaluation are removed. The graph is built
# by torchms.graph, batches are torchms Structures, the noise of a step is an
# argument, and the elements are sampled with Gumbel noise. Names of modules
# and parameters follow MatterGen to load its weights.
"""MatterGen, a diffusion model which generates crystals.

A crystal is a lattice, the fractional coordinates of its atoms and their
elements, which get noise together from the time 0 to the time 1:

* the lattice goes to a cubic cell of the mean density by a diffusion which
  preserves the variance, with a symmetric noise;
* the fractional coordinates get a noise which grows geometrically, smaller
  with more atoms, wrapped around the cell;
* the element of an atom becomes a mask, the element of an atom which has
  none, with a probability which grows linearly (an absorbing D3PM).

A network, GemNet-T, reads a noisy crystal and its time, and gives the scores
of its lattice and of its coordinates times their deviations, and the scores
of the element each atom had at the time 0.

A crystal is sampled from a random lattice and random coordinates, every atom
masked, in 1000 steps from the time 1 to the time 1 / 1000. A step corrects the
lattice and the coordinates by a Langevin move, then predicts the three of
them at the time of the step before.

The weights are the base model of the authors, trained on Alex-MP-20: it
generates crystals of up to 20 atoms without condition. MatterGenChemicalSystem
is the same model fine-tuned on a chemical system, the elements a crystal can
have: its network also reads this system, and its sampling mixes the scores
with and without it (classifier-free guidance).
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Bool, Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.graph.geometry import symmetric_lattice, triplet_cosines
from torchms.graph.transforms import (
    AddBonds,
    AddRadiusGraph,
    AddTriplets,
    KeepNearestNeighbors,
    MirrorEdges,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
)
from torchms.models.estimator.weights import download
from torchms.models.layers.basis import sinusoidal_frequencies, sinusoids
from torchms.models.layers.gemnet import Condition, GemNetT
from torchms.models.mattergen.config import Config
from torchms.models.mattergen.properties import PropertyEmbedding, property_embedding
from torchms.models.mattergen.diffusion import (
    Denoised,
    Guidance,
    PropertyValues,
    StepNoise,
    ancestral_coefficients,
    guided_scores,
    symmetric_noise,
)


# the steps of the sampling are separate methods, to compare them with MatterGen
# pylint: disable-next=abstract-method,too-many-public-methods
class MatterGen(PretrainedModel[Config]):
    """MatterGen: diffusion of the lattice, coordinates and elements of crystals.

    The model does not build the graph its network reads, see transform. A
    step of the sampling builds it itself, see sample_step.

    Args:
        config: Configuration of the model, the one of config.yaml by default.

    Examples:
        The network on two crystals, at the times 0.1 and 0.9:

        >>> model = MatterGen().eval()
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende()]
        ... )
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     denoised = model(structures, torch.tensor([0.1, 0.9]))
        >>> denoised.x.shape, denoised.lattice.shape, denoised.logits.shape
        (torch.Size([16, 3]), torch.Size([2, 3, 3]), torch.Size([16, 101]))

        With pretrained weights, `model.sample(model.sample_num_atoms(8))`
        generates 8 crystals.
    """

    CONFIG = Config
    GRAPH_FIELDS = ("graph", "triplets", "bonds")
    # properties the network was fine-tuned on, in the order of MatterGen:
    # ("chemical_system",) for MatterGenChemicalSystem, none for the base model
    CONDITIONS: tuple[str, ...] = ()

    time_frequencies: torch.Tensor
    logit_offset: torch.Tensor
    mask_kept: torch.Tensor
    num_atoms_distribution: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model

        # frequencies of the embedding of the time, from 1 to 1 / time_max_period:
        # the weights have their own, computed on another device
        self.register_buffer(
            "time_frequencies",
            sinusoidal_frequencies(
                network.hidden_dim, network.time_max_period, endpoint=False
            ),
        )
        self.gemnet = GemNetT(self.config.decoder_config(self.CONDITIONS))
        self.fc_atom = nn.Linear(network.hidden_dim, network.num_classes)
        # embeddings of the properties, named as in MatterGen
        tuning = self.fine_tuning()
        self.property_embeddings_adapt = nn.ModuleDict(
            {c: property_embedding(c, self.config, tuning) for c in self.CONDITIONS}
        )

        # added to the scores of the elements: 0 for an allowed one
        allowed = torch.zeros(network.num_classes)
        allowed[torch.tensor(network.allowed_elements) - 1] = 1.0
        self.register_buffer(
            "logit_offset", (1 - allowed) * network.masked_logit, persistent=False
        )

        # probability that an atom is not masked yet at each step of the
        # diffusion of the elements, from 0: 1 - t / N, computed as MatterGen
        steps = self.config.elements.num_steps
        betas = torch.cat([torch.zeros(1), 1 / (1.0 * steps - torch.arange(steps))]).to(
            torch.float64
        )
        kept = torch.cumprod(1 - betas, dim=0)
        kept[-1] = 0.0
        self.register_buffer("mask_kept", kept.float(), persistent=False)

        # probability of each number of atoms, from 0
        self.register_buffer(
            "num_atoms_distribution",
            torch.tensor(self.config.sampling.num_atoms_distribution),
            persistent=False,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes of the element of an atom: the elements, then the mask."""
        return self.config.model.num_classes

    @property
    def mask(self) -> int:
        """Atomic number of an atom which has no element yet."""
        return self.config.model.mask

    @classmethod
    def fine_tuning(cls) -> str | None:
        """Its properties joined by "_", its section of config.yaml; None if base."""
        return "_".join(cls.CONDITIONS) or None

    def default_weights(self) -> str:
        fine_tuning = self.fine_tuning()
        if fine_tuning is not None:
            return self.config.condition(fine_tuning).default_weights

        return self.config.default_weights

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint of pytorch-lightning only holds tensors and numbers.
        """
        checkpoint_path, config_path = download("mattergen", name, cache)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{checkpoint_path} is not a checkpoint of MatterGen")

        with open(config_path, "r", encoding="utf-8") as fp:
            configuration = yaml.safe_load(fp)

        self.load_pretrained(checkpoint["state_dict"], configuration)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        configuration: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a DiffusionLightningModule of MatterGen.

        Args:
            state_dict: Parameters of the module of MatterGen, its denoiser
                under diffusion_module.model.
            configuration: Configuration of the training of MatterGen,
                checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with another configuration: the base model
                and the models fine-tuned on a property refuse the weights of
                each other.
        """
        if configuration is not None:
            check_configuration(
                configuration, self.config.upstream_sections(self.fine_tuning())
            )

        prefix = self.config.upstream.prefix
        outside = [name for name in state_dict if not name.startswith(prefix)]
        if outside:
            raise InvalidWeights(f"parameters outside of {prefix}: {outside}")
        parameters = {}
        for name, values in state_dict.items():
            name = name[len(prefix) :]
            for old, new in self.config.upstream.renamed_parameters.items():
                if name.endswith(old):
                    name = name[: -len(old)] + new
            parameters[name] = values

        self.load_parameters(parameters)

    # graph

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the graph of GemNet-T, its triplets and its bonds.

        An atom is connected to its 50 nearest neighbours closer than 7 Å,
        its own images included, among the cells up to 5 cells away along
        each lattice vector: their number is bounded before they are listed,
        as MatterGen does, since the sampling meets flat cells. The graph is
        then made symmetric from the edges towards an atom of lower index.
        Among neighbours at the same distance, the ones which are kept are not
        always the ones of MatterGen: in a crystal, the two images of an atom
        on each side of it are always at the same distance.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return Compose(
            [
                AddRadiusGraph(
                    graph.radius,
                    self_images=True,
                    max_image_shift=graph.max_image_shift,
                ),
                KeepNearestNeighbors(graph.max_neighbors),
                MirrorEdges(),
                AddTriplets(),
                AddBonds(),
            ]
        )

    # network

    def embed_time(
        self, times: Float[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct hidden_dim"]:
        """Embedding of the time of each crystal, read by GemNet-T.

        The sine and the cosine of each frequency follow each other.
        """
        return sinusoids(times.unsqueeze(1) * self.time_frequencies, interleave=True)

    def element_rows(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Row of each atom in the embedding of the elements: z - 1, the mask last.

        Raises:
            ValueError: If an atom has an element after fermium, which is not
                the mask.
        """
        outside = (z < 1) | (z > self.mask)
        if outside.any():
            unknown = z[outside].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

        return z - 1

    def denoise(
        self,
        structures: Structures,
        times: Float[torch.Tensor, "n_struct"],
        values: PropertyValues | None = None,
        conditioned: Bool[torch.Tensor, "n_struct"] | None = None,
    ) -> Denoised:
        """Scores of noisy crystals, from the network.

        Args:
            structures: Noisy crystals with their graph, see transform. An atom
                without element has the atomic number of the mask, see mask.
            times: Time of each crystal, from 0 to 1: how much noise it has.
            values: Value of each property of CONDITIONS for each crystal, as
                Guidance.values: None for a property without condition. None
                for a network without condition.
            conditioned: Which crystals are conditioned on their properties,
                all of them by default.

        Raises:
            ValueError: If an atom has an element after fermium, if the
                structures do not have their graph, or if the values are not
                one per property of the model.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        angles = attached_tuples(structures, "triplets", self)
        bonds = attached_tuples(structures, "bonds", self)

        conditions, keep = self.property_conditions(times, values, conditioned)

        vectors = edges_vectors(graph)
        output = self.gemnet(
            self.embed_time(times),
            self.element_rows(structures.z),
            graph,
            angles,
            bonds,
            vectors,
            triplet_cosines(vectors, angles),
            structures.lattice,
            conditions,
        )
        assert output.lattice is not None

        # the move of an atom in Å, to fractional coordinates
        inverse = torch.linalg.inv(  # pylint: disable=not-callable
            structures.lattice
        ).transpose(1, 2)
        inverse = graph.unpool_structures(inverse.flatten(1)).view(-1, 3, 3)
        x = (inverse @ output.move.unsqueeze(-1)).squeeze(-1)

        logits = self.fc_atom(output.features) + self.logit_offset
        if keep is not None:
            # as MatterGen: the elements outside of a chemical system are
            # lowered after the ones which are not allowed
            keep = graph.unpool_structures(keep)
            logits = logits + (1 - keep) * self.config.model.masked_logit

        return Denoised(x=x, lattice=output.lattice, logits=logits)

    def property_conditions(
        self,
        times: Float[torch.Tensor, "n_struct"],
        values: PropertyValues | None,
        conditioned: Bool[torch.Tensor, "n_struct"] | None,
    ) -> tuple[
        tuple[Condition, ...], Float[torch.Tensor, "n_struct num_classes"] | None
    ]:
        """Condition of GemNet-T per property, and the classes of the elements kept.

        A property without value has its condition inactive, as the zero
        embedding of the adapters of MatterGen. Only a chemical system keeps
        fewer classes.
        """
        if values is None:
            values = (None,) * len(self.CONDITIONS)
        if values and not self.CONDITIONS:
            raise ValueError(f"{type(self).__name__} is not conditioned")
        if len(values) != len(self.CONDITIONS):
            raise ValueError(
                f"{type(self).__name__} is conditioned on {list(self.CONDITIONS)}: "
                f"give one value each, not {len(values)}"
            )

        n_struct, device = len(times), times.device
        if conditioned is None:
            conditioned = torch.ones(n_struct, dtype=torch.bool, device=device)
        inactive = torch.zeros(n_struct, dtype=torch.bool, device=device)
        zeros = times.new_zeros((n_struct, self.config.model.hidden_dim))

        conditions, keep = [], None
        for name, value in zip(self.CONDITIONS, values):
            if value is None:
                conditions.append(Condition(zeros, inactive))
                continue

            embedding = self.property_embeddings_adapt[name]
            assert isinstance(embedding, PropertyEmbedding)
            conditions.append(Condition(embedding(value), conditioned))
            kept = embedding.kept_classes(value, conditioned)
            if kept is not None:
                keep = kept

        return tuple(conditions), keep

    def guidance(
        self,
        *values: Sequence[object] | torch.Tensor | None,
        scale: float | None = None,
    ) -> Guidance:
        """Guidance of the sampling of a fine-tuned model towards its properties.

        Args:
            values: One value per crystal for each property of CONDITIONS, in
                their order, or None for a property the sampling is not
                conditioned on: chemical systems as "Li-O", space groups from
                1 to 230, numbers in the unit of the training set of MatterGen.
            scale: Factor of the guidance, the one of the configuration by
                default, 2 as in the README of MatterGen.

        Raises:
            ValueError: If the model is not fine-tuned, if the values are not
                one per property or all None, or if a value is not one of its
                property.
        """
        fine_tuning = self.fine_tuning()
        if fine_tuning is None:
            raise ValueError(f"{type(self).__name__} is not conditioned")
        if len(values) != len(self.CONDITIONS) or all(v is None for v in values):
            raise ValueError(
                f"{type(self).__name__} is conditioned on {list(self.CONDITIONS)}: "
                "give values for each, None for one without condition"
            )

        device = self.time_frequencies.device
        parsed = []
        for name, value in zip(self.CONDITIONS, values):
            embedding = self.property_embeddings_adapt[name]
            assert isinstance(embedding, PropertyEmbedding)
            parsed.append(None if value is None else embedding.parse(value, device))

        if scale is None:
            scale = self.config.condition(fine_tuning).guidance_scale

        return Guidance(tuple(parsed), scale)

    def forward(
        self,
        structures: Structures,
        times: Float[torch.Tensor, "n_struct"],
        values: PropertyValues | None = None,
        conditioned: Bool[torch.Tensor, "n_struct"] | None = None,
    ) -> Denoised:
        """Scores of noisy crystals, see denoise."""
        return self.denoise(structures, times, values, conditioned)

    # the three diffusions

    def coordinate_deviation(
        self, times: Float[torch.Tensor, "n"], num_atoms: Int[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n"]:
        """Deviation of the noise of the fractional coordinates, at each time.

        It grows geometrically with the time, and is divided by the cube root
        of the number of atoms.
        """
        noise = self.config.coordinates
        sigma = noise.sigma_min * (noise.sigma_max / noise.sigma_min) ** times

        return sigma * num_atoms ** (-1 / 3)

    def lattice_mean_coefficient(
        self, times: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n"]:
        """Part of a noisy lattice which is the lattice without noise, at each time."""
        noise = self.config.lattice
        beta_0, beta_1 = noise.beta_min, noise.beta_max

        return torch.exp(-0.25 * times**2 * (beta_1 - beta_0) - 0.5 * times * beta_0)

    def lattice_limit(
        self, num_atoms: Int[torch.Tensor, "n_struct"]
    ) -> tuple[
        Float[torch.Tensor, "n_struct 3 3"], Float[torch.Tensor, "n_struct 3 3"]
    ]:
        """Mean and variance of each element of a lattice at the time 1.

        The mean is a cubic cell of the mean density. The variance of each
        element grows as the number of atoms to the power 2/3.
        """
        noise = self.config.lattice
        eye = torch.eye(3, device=num_atoms.device).expand(len(num_atoms), 3, 3)
        counts = num_atoms[:, None, None]

        mean = torch.pow(eye * counts / noise.limit_density, 1.0 / 3)
        variance = torch.pow(counts.expand(-1, 3, 3), 2.0 / 3) * noise.limit_var_scaling

        return mean, variance

    def lattice_deviation(
        self,
        times: Float[torch.Tensor, "n_struct"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct 3 3"]:
        """Deviation of the noise of each element of a lattice, at each time."""
        coefficient = self.lattice_mean_coefficient(times)[:, None, None]
        _, variance = self.lattice_limit(num_atoms)

        return torch.sqrt((1.0 - coefficient**2) * variance)

    # sampling

    def times(self) -> Float[torch.Tensor, "num_steps"]:
        """Time of each step of the sampling, from 1 down to 1 / num_steps."""
        steps = self.config.sampling.num_steps

        return torch.linspace(1.0, 1 / steps, steps, device=self.logit_offset.device)

    def time_step(self) -> Float[torch.Tensor, ""]:
        """Difference between the time of a step and the time of the step after."""
        steps = self.config.sampling.num_steps

        return -torch.tensor((1.0 - 1 / steps) / (steps - 1)).to(
            self.logit_offset.device
        )

    def step_times(
        self, structures: Structures, step: int
    ) -> tuple[Float[torch.Tensor, "n_struct"], Float[torch.Tensor, ""]]:
        """Time of each crystal at a step of the sampling, and the time step."""
        times = torch.full(
            (len(structures.num_atoms),),
            float(self.times()[step]),
            device=structures.device(),
        )

        return times, self.time_step()

    def element_steps(
        self, times: Float[torch.Tensor, "n_struct"]
    ) -> Int[torch.Tensor, "n_struct"]:
        """Step of the diffusion of the elements at each time, from 0 to N - 1."""
        return (times * (self.config.elements.num_steps - 1)).long()

    def scores(
        self,
        structures: Structures,
        times: Float[torch.Tensor, "n_struct"],
        guidance: Guidance | None = None,
    ) -> Denoised:
        """Scores of noisy crystals: the ones of the network over their deviations.

        The scores of the elements are the ones of the network. With a
        guidance, the scores without and with the condition are mixed, see
        Guidance: two calls of the network, one at a scale of 0 or 1.

        Args:
            structures: Noisy crystals with their graph, see transform.
            times: Time of each crystal.
            guidance: Condition of a conditioned model, None without.
        """
        if guidance is None or abs(guidance.scale) < 1e-15:
            return self.unguided_scores(structures, times, None)
        if abs(guidance.scale - 1) < 1e-15:
            return self.unguided_scores(structures, times, guidance.values)

        unconditional = self.unguided_scores(structures, times, None)
        conditional = self.unguided_scores(structures, times, guidance.values)

        return guided_scores(unconditional, conditional, guidance.scale)

    def unguided_scores(
        self,
        structures: Structures,
        times: Float[torch.Tensor, "n_struct"],
        values: PropertyValues | None,
    ) -> Denoised:
        """Scores of noisy crystals, with or without their properties, see scores."""
        graph = attached_graph(structures, self)
        denoised = self.denoise(structures, times, values)

        num_atoms = structures.num_atoms
        deviation = graph.unpool_structures(self.coordinate_deviation(times, num_atoms))

        return Denoised(
            x=denoised.x / deviation.unsqueeze(-1),
            lattice=denoised.lattice / self.lattice_deviation(times, num_atoms),
            logits=denoised.logits,
        )

    def langevin_step_size(
        self,
        score_norm: Float[torch.Tensor, "n_struct"],
        noise_norm: Float[torch.Tensor, "n_struct"],
        snr: float,
        alpha: Float[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct"]:
        """Step of the Langevin corrector of each crystal.

        It is computed from the means of the norms of the scores and of the
        noises over the batch: the step of a crystal depends on the other
        crystals of the batch, as in MatterGen.
        """
        grad_norm, noise_norm = score_norm.mean(), noise_norm.mean()
        max_step = torch.tensor(self.config.sampling.max_step_size, device=alpha.device)

        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, max_step)

        return torch.where(grad_norm == 0, max_step, step_size)

    def correct(
        self,
        structures: Structures,
        step: int,
        noise: StepNoise,
        guidance: Guidance | None = None,
    ) -> Structures:
        """Langevin move of the coordinates and of the lattice at a step.

        The elements are kept. The graph is built here, on the crystals given,
        see correct_given_scores.

        Args:
            structures: Noisy crystals at the time of the step, without graph.
            step: Step of the sampling, from 0, at the time 1.
            noise: Random draws of the step: its corrector_x and its
                corrector_lattice are read.
            guidance: Condition of a conditioned model, see scores.
        """
        structures = self.transform(self.config)(structures)
        times, _ = self.step_times(structures, step)

        return self.correct_given_scores(
            structures, step, self.scores(structures, times, guidance), noise
        )

    def correct_given_scores(
        self, structures: Structures, step: int, scores: Denoised, noise: StepNoise
    ) -> Structures:
        """Langevin move of the coordinates and of the lattice, given their scores.

        Args:
            structures: Noisy crystals at the time of the step, with their
                graph, which only tells which atoms are in each crystal.
            step: Step of the sampling, from 0, at the time 1.
            scores: Scores of the crystals at the time of the step, see scores.
            noise: Random draws of the step: its corrector_x and its
                corrector_lattice are read.

        Returns:
            Crystals moved, without graph, their coordinates in the cell and
            their lattices symmetric.
        """
        sampling = self.config.sampling
        graph = attached_graph(structures, self)
        times, time_step = self.step_times(structures, step)

        # coordinates
        score_norm = graph.aggregate_nodes_to_structures(scores.x.square().sum(dim=1))
        noise_norm = graph.aggregate_nodes_to_structures(
            noise.corrector_x.square().sum(dim=1)
        )
        step_size = self.langevin_step_size(
            score_norm.sqrt(),
            noise_norm.sqrt(),
            sampling.snr_coordinates,
            torch.ones_like(times),
        )
        step_size = graph.unpool_structures(step_size).unsqueeze(-1)
        x = structures.x + step_size * scores.x
        x = x + torch.sqrt(step_size * 2) * noise.corrector_x

        # lattice
        coefficient = self.lattice_mean_coefficient(times)
        before = self.lattice_mean_coefficient(times + time_step)
        lattice_noise = symmetric_noise(noise.corrector_lattice)
        step_size = self.langevin_step_size(
            scores.lattice.square().flatten(1).sum(dim=1).sqrt(),
            lattice_noise.square().flatten(1).sum(dim=1).sqrt(),
            sampling.snr_lattice,
            coefficient**2 / before**2,
        )[:, None, None]
        lattice = structures.lattice + step_size * scores.lattice
        lattice = lattice + torch.sqrt(step_size * 2) * lattice_noise

        return Structures(
            lattice=symmetric_lattice(lattice),
            x=x % 1.0,
            z=structures.z,
            num_atoms=structures.num_atoms,
        )

    def element_posterior(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        logits: Float[torch.Tensor, "n_atoms num_classes"],
        steps: Int[torch.Tensor, "n_atoms"],
    ) -> Float[torch.Tensor, "n_atoms num_classes"]:
        """Scores of the element of each atom at the step before, in the diffusion.

        The probabilities of the element at the time 0 given by the network
        are moved to the step before, then multiplied by the probability to
        go from each element to the one of the atom.

        Args:
            z: Atomic number of each atom at its step, the mask included.
            logits: Scores of the element of each atom at the time 0, from the
                network.
            steps: Step of the diffusion of the elements of each atom, from 0
                to num_steps - 1.

        Returns:
            Logarithms of the probabilities, not normalized, of each element,
            then of the mask, shape (n_atoms, num_classes).
        """
        epsilon = self.config.elements.log_epsilon
        steps_total = self.config.elements.num_steps
        probabilities = torch.softmax(logits, dim=-1)

        # the element at the step, from the one at the time 0
        kept = self.mask_kept[steps]
        unmasked = kept[:, None] * probabilities[:, :-1]
        at_step = torch.cat([unmasked, (1 - unmasked.sum(-1))[:, None]], dim=-1)
        at_step = torch.where(steps[:, None] == 0, probabilities, at_step)

        # probability of going from each element to the one of the atom: the
        # element is kept or masked, the mask is kept
        beta = 1 / (1.0 * steps_total - steps)
        # pylint: disable-next=not-callable
        current = F.one_hot(z - 1, self.num_classes).float()
        transition = torch.cat(
            [
                (1 - beta)[:, None] * current[:, :-1] + beta[:, None] * current[:, -1:],
                current[:, -1:],
            ],
            dim=-1,
        )

        return torch.log(at_step + epsilon) + torch.log(transition + epsilon)

    def predict(
        self,
        structures: Structures,
        step: int,
        noise: StepNoise,
        guidance: Guidance | None = None,
    ) -> Structures:
        """Crystals at the step after, from the ones of a step and their scores.

        The graph is built here, on the crystals given, see
        predict_given_scores.

        Args:
            structures: Noisy crystals at the time of the step, without graph.
            step: Step of the sampling, from 0, at the time 1.
            noise: Random draws of the step: its predictor_x, its
                predictor_lattice and its elements are read.
            guidance: Condition of a conditioned model, see scores.
        """
        structures = self.transform(self.config)(structures)
        times, _ = self.step_times(structures, step)

        return self.predict_given_scores(
            structures, step, self.scores(structures, times, guidance), noise
        )

    def predict_given_scores(  # pylint: disable=too-many-locals
        self, structures: Structures, step: int, scores: Denoised, noise: StepNoise
    ) -> Structures:
        """Crystals at the step after, from the ones of a step and their scores.

        Args:
            structures: Noisy crystals at the time of the step, with their
                graph, which only tells which atoms are in each crystal.
            step: Step of the sampling, from 0, at the time 1.
            scores: Scores of the crystals at the time of the step, see scores.
            noise: Random draws of the step: its predictor_x, its
                predictor_lattice and its elements are read.

        Returns:
            Crystals at the time of the step after, without graph, their
            coordinates in the cell. After the last step, none of them is
            noisy.
        """
        min_ratio = self.config.sampling.min_alpha_ratio
        num_atoms = structures.num_atoms
        graph = attached_graph(structures, self)
        times, time_step = self.step_times(structures, step)
        before = times + time_step
        at_zero = before <= 0

        # coordinates: no shrinking towards a mean
        ones = torch.ones_like(structures.x)
        x_coeff, score_coeff, std = ancestral_coefficients(
            ones,
            graph.unpool_structures(self.coordinate_deviation(times, num_atoms))[
                :, None
            ],
            ones,
            graph.unpool_structures(self.coordinate_deviation(before, num_atoms))[
                :, None
            ],
            graph.unpool_structures(at_zero)[:, None],
            min_ratio,
        )
        x = x_coeff * structures.x + score_coeff * scores.x
        x = x + std * noise.predictor_x

        # lattice, towards the cubic cell of the time 1
        limit_mean, _ = self.lattice_limit(num_atoms)
        x_coeff, score_coeff, std = ancestral_coefficients(
            self.lattice_mean_coefficient(times)[:, None, None],
            self.lattice_deviation(times, num_atoms),
            self.lattice_mean_coefficient(before)[:, None, None],
            self.lattice_deviation(before, num_atoms),
            at_zero[:, None, None],
            min_ratio,
        )
        lattice = (
            x_coeff * structures.lattice
            + score_coeff * scores.lattice
            + (1 - x_coeff) * limit_mean
        )
        lattice = lattice + std * symmetric_noise(
            noise.predictor_lattice[:, None, None]
        )

        # elements
        steps = graph.unpool_structures(self.element_steps(times))
        posterior = self.element_posterior(structures.z, scores.logits, steps)
        if noise.elements is None:
            classes = torch.softmax(posterior, dim=-1).argmax(dim=-1)
        else:
            classes = (posterior + noise.elements).argmax(dim=-1)

        return Structures(
            lattice=lattice, x=x % 1.0, z=classes + 1, num_atoms=num_atoms
        )

    def sample_step(
        self,
        structures: Structures,
        step: int,
        noise: StepNoise,
        guidance: Guidance | None = None,
    ) -> Structures:
        """One step of the sampling: a correction, then a prediction.

        Two calls of the network, each on a graph built on the crystals it is
        given, see correct and predict; four with a guidance.
        """
        corrected = self.correct(structures, step, noise, guidance)

        return self.predict(corrected, step, noise, guidance)

    def sample_prior(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Crystals at the time 1: random lattices and coordinates, every atom masked.

        Args:
            num_atoms: Number of atoms of each crystal, on the device of the
                model.
            generator: Generator of the random draws, on the device of
                num_atoms.
        """
        device = num_atoms.device
        n_struct, n_atoms = len(num_atoms), int(num_atoms.sum())
        batch = torch.arange(n_struct, device=device).repeat_interleave(num_atoms)

        x = torch.randn((n_atoms, 3), generator=generator, device=device)
        x = x * self.config.coordinates.sigma_max
        x = x * (num_atoms ** (-1 / 3))[batch].unsqueeze(-1)

        mean, variance = self.lattice_limit(num_atoms)
        noise = torch.randn((n_struct, 3, 3), generator=generator, device=device)
        lattice = symmetric_noise(noise) * variance.sqrt() + mean

        z = torch.full((n_atoms,), self.mask, device=device)

        return Structures(lattice=lattice, x=x % 1.0, z=z, num_atoms=num_atoms)

    def sample_num_atoms(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> Int[torch.Tensor, "num_samples"]:
        """Draw numbers of atoms as the ones of the training set, Alex-MP-20.

        Args:
            num_samples: Number of crystals.
            generator: Generator of the random draws, on the device of the
                model.

        Returns:
            Number of atoms of each crystal, from 1 to 20, shape (num_samples,).
        """
        return torch.multinomial(
            self.num_atoms_distribution,
            num_samples,
            replacement=True,
            generator=generator,
        )

    def step_noise(
        self,
        structures: Structures,
        last: bool,
        generator: torch.Generator | None = None,
    ) -> StepNoise:
        """Random draws of a step of the sampling, for crystals of a batch.

        Args:
            structures: Crystals of the step.
            last: If True, the elements take their most probable values.
            generator: Generator of the random draws, on the device of the
                structures.
        """
        device = structures.device()
        n_struct, n_atoms = len(structures.num_atoms), len(structures.z)

        def normal(*shape: int) -> torch.Tensor:
            return torch.randn(shape, generator=generator, device=device)

        elements = None
        if not last:
            uniform = torch.rand(
                (n_atoms, self.num_classes), generator=generator, device=device
            )
            # Gumbel noise, from a uniform one which is never 0 or 1
            uniform = uniform.clamp(torch.finfo(uniform.dtype).tiny, 1 - 1e-7)
            elements = -torch.log(-torch.log(uniform))

        return StepNoise(
            corrector_x=normal(n_atoms, 3),
            corrector_lattice=normal(n_struct, 3, 3),
            predictor_x=normal(n_atoms, 3),
            predictor_lattice=normal(n_struct),
            elements=elements,
        )

    def sample(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        generator: torch.Generator | None = None,
        guidance: Guidance | None = None,
    ) -> Structures:
        """Generate crystals of given numbers of atoms.

        From crystals at the time 1, see sample_prior, every step of the
        sampling is followed: two calls of the network per step. The crystals
        of a batch are not independent: the steps of the Langevin corrector
        depend on all of them, as in MatterGen.

        Args:
            num_atoms: Number of atoms of each crystal, shape (n_struct,), on
                the device of the model. See sample_num_atoms.
            generator: Generator of the random draws, on the device of
                num_atoms.
            guidance: Chemical system of each crystal and factor of the
                guidance, for a conditioned model: see
                MatterGenChemicalSystem.guidance. Without, a conditioned model
                samples without condition.

        Returns:
            Crystals generated, without graph. Their lattices are symmetric
            matrices, not reduced.
        """
        structures = self.sample_prior(num_atoms, generator)
        steps = self.config.sampling.num_steps

        for step in range(steps):
            noise = self.step_noise(structures, step == steps - 1, generator)
            structures = self.sample_step(structures, step, noise, guidance)

        return structures
