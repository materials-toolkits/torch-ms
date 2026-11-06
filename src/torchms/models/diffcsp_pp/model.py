# Derived from DiffCSP++ (https://github.com/jiaor17/DiffCSP-PP), Copyright (c)
# 2024 Rui Jiao, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the two diffusions of DiffCSP++
# (diffcsp/pl_modules/diffusion.py, diffusion_w_type.py and
# lattice/crystal_family.py, at the commit e82e7ff), for inference only: the
# network on noisy structures and the sampling of a lattice and of fractional
# coordinates on given Wyckoff positions, for given elements or with the
# elements. The losses, the training, the sampling from a noisy structure
# (diff_ratio), the search of templates (CSPML), the datasets and the metrics
# are removed. The Wyckoff positions are read from the table of torchms, built
# from pyxtal (torchms.data.symmetry.WyckoffSites); the complete graph is built
# by torchms.graph, once per sampling; batches are torchms Structures; the
# noise of a step is an argument. The network, CSPNet, and the schedules of the
# noise are the ones of DiffCSP in torchms.models.layers. Names of modules and
# parameters follow DiffCSP++ to load its weights.
"""DiffCSP++, a diffusion model of the structure of a crystal of a space group.

The diffusion of DiffCSP (torchms.models.diffcsp), constrained by a space group
and Wyckoff positions. The atoms of a structure are grouped in sites, each one
an orbit of a Wyckoff position: an atom is the image of the free point of its
site by an operation of the group, and only the free points move. The lattice
is exp(sum_i k_i B_i) for six numbers k and a basis B of symmetric matrices;
the crystal family of the group fixes some of the numbers (a cubic lattice
only has its volume).

A structure is sampled from random numbers k and random coordinates by 999
steps, from the time 999 to 1, as DiffCSP++ does. A step corrects the
coordinates by a Langevin move, then predicts k and the coordinates of the
time before. The noise and the scores of the coordinates are brought back to
the free point of each site, averaged over the site and sent back to its
atoms, and the atoms are placed again from their free points.

DiffCSPPlusPlus samples the structure of atoms of given elements on given
Wyckoff positions. In DiffCSPPlusPlusGeneration the element of a site is a
vector with one value per element, sampled with the structure.

Weights are listed in config.yaml: the ones of the authors, trained on MP-20,
MPTS-52, Perov-5 and Carbon-24.
"""

from __future__ import annotations

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
from torchms.data.symmetry import WyckoffSites
from torchms.graph.geometry import polar_logarithm
from torchms.graph.transforms import AddCompleteGraph, attached_graph
from torchms.models.diffcsp_pp.config import Config, LatticeConfig
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.models.layers.basis import sinusoidal_embedding
from torchms.models.layers.cspnet import CSPNet
from torchms.models.layers.schedules import BetaScheduler, SigmaScheduler

# space groups, from 0: DiffCSP++ has a row for the group 0
NUM_SPACEGROUPS = 231


class CrystalFamily(nn.Module):  # pylint: disable=abstract-method
    """Lattices as six numbers, with the constraints of the space groups.

    Attributes:
        basis: Symmetric matrices B_i, of norm 1, shape (6, 3, 3).
        masks: 0 for the numbers each space group fixes, 1 for the other ones,
            shape (231, 6), from the group 0.
        biass: Values of the fixed numbers, 0 for the other ones.
        family: Crystal family of each space group, from 1 (triclinic) to 6
            (cubic).

    Examples:
        >>> family = CrystalFamily(Config.load().lattice)
        >>> k = family.project(torch.ones(2, 6), torch.tensor([225, 194]))
        >>> lattice = family.lattice(k)
        >>> torch.allclose(lattice[0], lattice[0, 0, 0] * torch.eye(3))
        True
        >>> lengths = lattice[1].norm(dim=1)
        >>> cosine = (lattice[1, 0] @ lattice[1, 1]) / (lengths[0] * lengths[1])
        >>> round(float(lengths[0] / lengths[1]), 5), round(float(cosine), 5)
        (1.0, -0.5)
    """

    basis: torch.Tensor
    masks: torch.Tensor
    biass: torch.Tensor
    family: torch.Tensor

    def __init__(self, config: LatticeConfig):
        super().__init__()
        basis = torch.tensor(config.basis)
        basis = basis / basis.norm(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        masks = torch.ones(NUM_SPACEGROUPS, 6)
        biass = torch.zeros(NUM_SPACEGROUPS, 6)
        family = torch.zeros(NUM_SPACEGROUPS, dtype=torch.int64)
        first = 0
        for index, crystal in enumerate(config.families, start=1):
            groups = slice(first, crystal.last_spacegroup + 1)
            fixed = list(crystal.fixed)
            masks[groups, fixed] = 0.0
            biass[groups, fixed] = torch.tensor(crystal.values)
            family[groups] = index
            first = crystal.last_spacegroup + 1

        self.register_buffer("basis", basis)
        self.register_buffer("masks", masks)
        self.register_buffer("biass", biass)
        self.register_buffer("family", family)

    def project(
        self,
        k: Float[torch.Tensor, "n_struct 6"],
        spacegroup: Int[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct 6"]:
        """The numbers k of each structure, with the ones its space group fixes set."""
        return k * self.masks[spacegroup] + self.biass[spacegroup]

    def lattice(
        self, k: Float[torch.Tensor, "n_struct 6"]
    ) -> Float[torch.Tensor, "n_struct 3 3"]:
        """Symmetric lattice exp(sum_i k_i B_i) of each structure."""
        return torch.matrix_exp(torch.einsum("bk, kij -> bij", k, self.basis))

    def parameters_of(
        self, lattice: Float[torch.Tensor, "n_struct 3 3"]
    ) -> Float[torch.Tensor, "n_struct 6"]:
        """Numbers k of cells: the logarithm of their symmetric part on the basis.

        The inverse of lattice for a symmetric lattice; a cell and its
        rotations have the same numbers. The constraints of a space group are
        not applied, see project.
        """
        return torch.einsum("bij, kij -> bk", polar_logarithm(lattice), self.basis)


@dataclass(frozen=True)
class State:
    """Noisy structures of a time of the diffusion, or what the network finds in them.

    Args:
        lattice: The six numbers k of each lattice, shape (n_struct, 6).
        x: Fractional coordinates of each atom, shape (n_atoms, 3).
        elements: Vector of the element of each atom, for the generation,
            shape (n_atoms, num_elements); None when the elements are given.
    """

    lattice: Float[torch.Tensor, "n_struct 6"]
    x: Float[torch.Tensor, "n_atoms 3"]
    elements: Float[torch.Tensor, "n_atoms num_elements"] | None = None


@dataclass(frozen=True)
class Noise:
    """Standard normal noise of one step of the sampling, as DiffCSP++ draws it.

    The noise is drawn for every atom; only the one of the first atom of each
    site is used. Every field but the lattice is zero at the time 1, where the
    noise of the lattice has no weight.

    Args:
        corrector: Noise of the correction of the coordinates, shape
            (n_atoms, 3).
        lattice: Noise of the prediction of the numbers of the lattice, shape
            (n_struct, 6).
        predictor: Noise of the prediction of the coordinates, shape
            (n_atoms, 3).
        elements: Noise of the prediction of the elements, for the generation,
            shape (n_atoms, num_elements); None otherwise.
    """

    corrector: Float[torch.Tensor, "n_atoms 3"]
    lattice: Float[torch.Tensor, "n_struct 6"]
    predictor: Float[torch.Tensor, "n_atoms 3"]
    elements: Float[torch.Tensor, "n_atoms num_elements"] | None = None


class Diffusion(PretrainedModel[Config]):  # pylint: disable=abstract-method
    """What the two models of DiffCSP++ share: network, schedules and steps.

    Args:
        config: Configuration of the models, the one of config.yaml by default.

    Attributes:
        GENERATION: Whether the model also generates the elements: it then
            reads what the configuration has for the generation.
        weights: Name of the weights loaded by load_checkpoint, None before:
            the step of the corrector is the one of these weights.
    """

    CONFIG = Config
    GENERATION = False
    GRAPH_FIELDS = ("graph",)

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.decoder = CSPNet(self.config.decoder_config(self.GENERATION))
        self.beta_scheduler = BetaScheduler(self.config.model.schedules)
        self.sigma_scheduler = SigmaScheduler(self.config.model.schedules)
        self.crystal_family = CrystalFamily(self.config.lattice)
        self.weights: str | None = None

    @property
    def num_elements(self) -> int:
        """Number of elements of the network, from hydrogen."""
        return self.decoder.config.num_elements

    @property
    def timesteps(self) -> int:
        """Number of steps of the diffusion."""
        return self.config.model.schedules.timesteps

    def default_weights(self) -> str:
        if self.GENERATION:
            return self.config.generation.default_weights

        return self.config.default_weights

    def step_size(self, step_size: float | None = None) -> float:
        """Step of the corrector at the lowest level of noise.

        The one given, or the one of the configuration for the weights loaded
        (for the default ones before).
        """
        if step_size is not None:
            return step_size

        return self.config.sampling.step_sizes[self.weights or self.default_weights()]

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint of pytorch-lightning only holds tensors and numbers.
        """
        checkpoint_path, hparams_path = download("diffcsp_pp", name, cache)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{checkpoint_path} is not a checkpoint of DiffCSP++")

        with open(hparams_path, "r", encoding="utf-8") as fp:
            hparams = yaml.safe_load(fp)

        self.load_pretrained(checkpoint["state_dict"], hparams)
        self.weights = name

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        hparams: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a CSPDiffusion of DiffCSP++.

        Args:
            state_dict: Parameters of the model of DiffCSP++, with the buffers
                of its two schedules of noise and of its crystal families.
            hparams: Configuration of the training of DiffCSP++, checked if
                given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was trained with another configuration: the weights of
                one of the two models are refused by the other one.
        """
        if hparams is not None:
            check_configuration(hparams, self.config.upstream.sections(self.GENERATION))

        self.load_parameters(state_dict)

    # graph

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the complete graph of each structure.

        An edge goes from each atom to every atom of its structure, itself
        included. The graph only depends on the numbers of atoms.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return AddCompleteGraph(self_loops=graph.self_loops)

    # network

    def embed_time(
        self, times: Int[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct time_dim"]:
        """Embedding of the time of each structure, read by the network."""
        network = self.config.model

        return sinusoidal_embedding(times, network.time_dim, network.time_max_period)

    def network_input(
        self, structures: Structures, state: State
    ) -> torch.Tensor | Float[torch.Tensor, "n_atoms num_elements"]:
        """What the network reads of the element of each atom."""
        raise NotImplementedError

    def denoise(
        self,
        structures: Structures,
        state: State,
        times: Int[torch.Tensor, "n_struct"],
    ) -> State:
        """What the network finds in noisy structures.

        Args:
            structures: Structures with their complete graph, see transform:
                their graph and, given the elements, their atomic numbers are
                read; the noisy structures are the state.
            state: Numbers of the lattices, coordinates and, for the
                generation, vectors of the elements, at the times.
            times: Time of each structure, from 1 to the number of steps of
                the diffusion: how much noise it has.

        Returns:
            Normal noise of the numbers of each lattice, score of the
            coordinates of each atom over the square root of
            sigma_scheduler.sigmas_norm at its time, and for the generation
            normal noise of the vector of each atom. No constraint of a space
            group is applied.

        Raises:
            ValueError: If an atom has an element after fermium, or if the
                structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)

        lattice, score, elements = self.decoder(
            self.embed_time(times),
            self.network_input(structures, state),
            graph,
            state.x,
            state.lattice,
        )

        return State(lattice=lattice, x=score, elements=elements)

    # sampling

    def sample_step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        structures: Structures,
        sites: WyckoffSites,
        state: State,
        time: int,
        noise: Noise,
        step_size: float | None = None,
    ) -> State:
        """One step of the sampling, from a time to the time before.

        The coordinates are first corrected by a Langevin move at the level of
        noise of the time. Then the lattice, the coordinates and the elements
        of the time before are predicted from the corrected structure. The
        moves keep the constraints of the space groups and of the sites.

        Args:
            structures: Structures with their complete graph, see denoise.
            sites: Sites of the atoms on their Wyckoff positions.
            state: Noisy structures at the time, which keep the constraints.
            time: Time of the structures, from 999 down to 1.
            noise: Standard normal noise of the step.
            step_size: Step of the correction at the lowest level of noise,
                see step_size.

        Returns:
            Structures at the time before, their coordinates in the cell.
        """
        schedule = self.sigma_scheduler
        times = torch.full((len(sites.num_atoms),), time, device=state.x.device)

        # corrector: the lattice and the elements are kept
        found = self.denoise(structures, state, times)
        score = sites.symmetrize(schedule.score(found.x, time))
        step = schedule.normalized_langevin_step(time, self.step_size(step_size))
        x = schedule.correct(state.x, score, step, sites.spread(noise.corrector))
        corrected = State(state.lattice, sites.place(x), state.elements)

        # predictor
        found = self.denoise(structures, corrected, times)
        lattice_noise = self.crystal_family.project(noise.lattice, sites.spacegroup)
        lattice = self.beta_scheduler.predict(
            corrected.lattice, found.lattice, time, lattice_noise
        )
        lattice = self.crystal_family.project(lattice, sites.spacegroup)

        score = sites.symmetrize(schedule.score(found.x, time))
        x = schedule.predict(corrected.x, score, time, sites.spread(noise.predictor))

        elements = None
        if corrected.elements is not None:
            assert found.elements is not None and noise.elements is not None
            elements = self.beta_scheduler.predict(
                corrected.elements,
                sites.unpool_sites(sites.mean_over_sites(found.elements)),
                time,
                sites.unpool_sites(sites.anchors(noise.elements)),
            )
            elements = sites.unpool_sites(sites.anchors(elements))

        return State(lattice, sites.place(x), elements)

    def initial_state(
        self, sites: WyckoffSites, generator: torch.Generator | None = None
    ) -> State:
        """Random structures to start a sampling from, in the order of DiffCSP++.

        Uniform coordinates, numbers of the lattices and, for the generation,
        vectors of the elements of standard normal noise, with the constraints
        of the space groups and of the sites.
        """
        device = sites.operations.device
        n_atoms = len(sites.site)

        x = torch.rand((n_atoms, 3), generator=generator, device=device)
        lattice = torch.randn(
            (len(sites.num_atoms), 6), generator=generator, device=device
        )
        lattice = self.crystal_family.project(lattice, sites.spacegroup)
        elements = None
        if self.GENERATION:
            elements = torch.randn(
                (n_atoms, self.num_elements), generator=generator, device=device
            )
            elements = sites.unpool_sites(sites.anchors(elements))

        return State(lattice, sites.place(x), elements)

    def step_noise(
        self,
        sites: WyckoffSites,
        time: int,
        generator: torch.Generator | None = None,
    ) -> Noise:
        """Standard normal noise of a step, zero at the time 1."""
        device = sites.operations.device
        n_atoms = len(sites.site)
        scale = 1.0 if time > 1 else 0.0

        def draw(*shape: int) -> torch.Tensor:
            return scale * torch.randn(shape, generator=generator, device=device)

        corrector = draw(n_atoms, 3)
        lattice = draw(len(sites.num_atoms), 6)
        predictor = draw(n_atoms, 3)
        elements = draw(n_atoms, self.num_elements) if self.GENERATION else None

        return Noise(corrector, lattice, predictor, elements)

    def integrate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        structures: Structures,
        sites: WyckoffSites,
        state: State,
        step_size: float | None = None,
        generator: torch.Generator | None = None,
    ) -> State:
        """Every step of the sampling from the time 999 to 0, from a state.

        Args:
            structures: Structures with their complete graph, see denoise.
            sites: Sites of the atoms.
            state: Structures at the time 999, see initial_state.
            step_size: See sample_step.
            generator: Generator of the noise, on the device of the sites.
        """
        # DiffCSP++ starts one step after the last time of its schedules
        for time in range(self.timesteps - 1, 0, -1):
            noise = self.step_noise(sites, time, generator)
            state = self.sample_step(structures, sites, state, time, noise, step_size)

        return state

    def structures_of(
        self,
        sites: WyckoffSites,
        state: State,
        z: Int[torch.Tensor, "n_atoms"],
    ) -> Structures:
        """Structures of a state: its lattices are the symmetric matrices of k."""
        return Structures(
            lattice=self.crystal_family.lattice(state.lattice),
            x=state.x,
            z=z,
            num_atoms=sites.num_atoms,
        )


class DiffCSPPlusPlus(Diffusion):
    """DiffCSP++: diffusion of a crystal of given elements on Wyckoff positions.

    The model does not build the complete graph its network reads, see
    transform; sample builds it.

    Examples:
        The network on two structures of the rock salt (Fm-3m, the positions
        4a and 4b), at the times 1 and 1000:

        >>> model = DiffCSPPlusPlus().eval()
        >>> sites = WyckoffSites.from_letters([225, 225], ["ab", "ab"])
        >>> z = sites.unpool_sites(torch.tensor([11, 17, 19, 35]))
        >>> structures = model.transform()(
        ...     Structures(
        ...         lattice=torch.eye(3).expand(2, 3, 3),
        ...         x=sites.place(torch.zeros(16, 3)),
        ...         z=z,
        ...         num_atoms=sites.num_atoms,
        ...     )
        ... )
        >>> k = model.crystal_family.project(torch.zeros(2, 6), sites.spacegroup)
        >>> times = torch.tensor([1, 1000])
        >>> with torch.no_grad():
        ...     found = model(structures, State(k, structures.x), times)
        >>> found.lattice.shape, found.x.shape, found.elements
        (torch.Size([2, 6]), torch.Size([16, 3]), None)

        With pretrained weights, `model.sample(sites, torch.tensor([11, 17, 19, 35]))`
        gives a lattice and coordinates to the atoms of these sites.
    """

    elements: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)

        # row of each atomic number in the embedding of the network, from hydrogen
        self.register_buffer(
            "elements",
            element_table(torch.arange(1, self.num_elements + 1)),
            persistent=False,
        )

    def network_input(
        self, structures: Structures, state: State
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Row of the element of each atom in the embedding of the network."""
        return element_index(structures.z, self.elements)

    def forward(
        self,
        structures: Structures,
        state: State,
        times: Int[torch.Tensor, "n_struct"],
    ) -> State:
        """What the network finds in noisy structures, see denoise."""
        return self.denoise(structures, state, times)

    def sample(
        self,
        sites: WyckoffSites,
        z: Int[torch.Tensor, "n_sites"],
        step_size: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Sample the lattice and the coordinates of atoms on Wyckoff positions.

        The numbers of the lattices start from normal noise and the
        coordinates of the free points from uniform ones, and follow every
        step of the diffusion, from the time 999: two calls of the network
        per step.

        Args:
            sites: Space groups of the structures and sites of their atoms,
                on the device of the model, see WyckoffSites.from_letters.
            z: Atomic number of each site, given to its atoms, shape
                (n_sites,).
            step_size: See sample_step.
            generator: Generator of the random draws, on the device of the
                sites.

        Returns:
            Structures sampled, without graph. Their lattices are symmetric
            matrices.

        Raises:
            ValueError: If an atom has an element after fermium.
        """
        z = sites.unpool_sites(z)
        element_index(z, self.elements)
        state = self.initial_state(sites, generator)
        structures = self.transform(self.config)(self.structures_of(sites, state, z))

        state = self.integrate(structures, sites, state, step_size, generator)

        return self.structures_of(sites, state, z)


class DiffCSPPlusPlusGeneration(Diffusion):
    """DiffCSP++ which also generates the element of each site.

    The element of an atom is a vector of one value per element, from
    hydrogen, which is 1 for its element and 0 elsewhere without noise. It
    gets the normal noise of the lattice, and the network gives this noise for
    each atom; the atoms of a site share their vector. The element of an atom
    is the largest value of its vector. The atomic numbers of the structures
    given to the model are not read.

    The model does not build the complete graph its network reads, see
    transform; sample builds it.

    Examples:
        The network on structures of the rock salt without noise:

        >>> model = DiffCSPPlusPlusGeneration().eval()
        >>> sites = WyckoffSites.from_letters([225], ["ab"])
        >>> z = sites.unpool_sites(torch.tensor([11, 17]))
        >>> structures = model.transform()(
        ...     Structures(
        ...         lattice=torch.eye(3).unsqueeze(0),
        ...         x=sites.place(torch.zeros(8, 3)),
        ...         z=z,
        ...         num_atoms=sites.num_atoms,
        ...     )
        ... )
        >>> k = model.crystal_family.project(torch.zeros(1, 6), sites.spacegroup)
        >>> state = State(k, structures.x, model.element_vectors(z))
        >>> with torch.no_grad():
        ...     found = model(structures, state, torch.tensor([1]))
        >>> found.elements.shape
        torch.Size([8, 100])

        With pretrained weights, `model.sample(sites)` generates structures of
        these Wyckoff positions with their elements.
    """

    GENERATION = True

    def element_vectors(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms num_elements"]:
        """Vector of the element of each atom, without noise.

        Raises:
            ValueError: If an atom has an element after fermium.
        """
        outside = (z < 1) | (z > self.num_elements)
        if outside.any():
            unknown = z[outside].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

        # pylint: disable-next=not-callable
        return F.one_hot(z - 1, self.num_elements).float()

    def atomic_numbers(
        self, elements: Float[torch.Tensor, "n_atoms num_elements"]
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Atomic number of each atom: the largest value of its vector."""
        return elements.argmax(dim=-1) + 1

    def network_input(
        self, structures: Structures, state: State
    ) -> Float[torch.Tensor, "n_atoms num_elements"]:
        """The vector of the element of each atom, from the state."""
        if state.elements is None:
            raise ValueError("the generation needs the vectors of the elements")

        return state.elements

    def forward(
        self,
        structures: Structures,
        state: State,
        times: Int[torch.Tensor, "n_struct"],
    ) -> State:
        """What the network finds in noisy structures, see denoise."""
        return self.denoise(structures, state, times)

    def sample(
        self,
        sites: WyckoffSites,
        step_size: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Sample structures on Wyckoff positions, with the element of each site.

        Args:
            sites: Space groups of the structures and sites of their atoms,
                on the device of the model, see WyckoffSites.from_letters.
            step_size: See sample_step.
            generator: Generator of the random draws, on the device of the
                sites.

        Returns:
            Structures sampled, without graph. Their lattices are symmetric
            matrices.
        """
        state = self.initial_state(sites, generator)
        assert state.elements is not None
        z = self.atomic_numbers(state.elements)
        structures = self.transform(self.config)(self.structures_of(sites, state, z))

        state = self.integrate(structures, sites, state, step_size, generator)
        assert state.elements is not None

        return self.structures_of(sites, state, self.atomic_numbers(state.elements))
