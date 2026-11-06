# Derived from CrystalFlow (https://github.com/ixsluo/CrystalFlow), Copyright (c)
# 2024 Xiaoshan Luo, MIT license, itself derived from DiffCSP, Copyright (c) 2023
# Rui Jiao, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of CrystalFlow at the tag v1.0.0-alpha.1 of
# its weights (diffcsp/pl_modules/flow.py, cspnet.py, type_module.py and
# conditioning.py, and the polar decomposition of diffcsp/common/data_utils.py),
# for inference only: the network on a state of the flow, its sampling by
# Euler steps with the annealing of the coordinates, the generation of the
# elements in the table of type_module.py, and the guidance by a property
# expanded in gaussians. The losses, the training, the optimal transport, the
# symmetrization by space groups, the other solvers, the other encodings of
# the elements and of the properties, the nearest neighbours, the datasets and
# the metrics are removed. The complete graph is built by torchms.graph, the
# layers are the ones of torchms.models.layers.cspnet, the polar decomposition
# is torchms.graph.geometry, batches are torchms Structures, and the noise of
# the sampling is drawn by torch with a generator. Names of modules and
# parameters follow CrystalFlow to load its weights.
"""CrystalFlow, flow matching of the structure of crystals.

A structure is its lattice and the fractional coordinates of its atoms. The
lattice is given to the flow as six numbers k of its polar decomposition, the
symmetric matrix exp(S(k)) with its lengths and its angles, which do not
change with a rotation. A CSPNet reads the atoms, their coordinates, k and the
time, and gives the velocity of k and of the coordinates. The flow goes from
noise at the time 0 (normal k around a cube of volume e^3, uniform
coordinates) to a crystal at the time 1, by Euler steps.

CrystalFlow predicts the structure of atoms of given elements, as DiffCSP. In
CrystalFlowGeneration the element of an atom is also generated: two one-hot
vectors of its row and of its column in a table of the elements, which start
from normal noise and get their velocity from the network. A model can be
conditioned on a property of the structures (the pressure, the formation
energy): the property is standardized, expanded in gaussians, and added to
the features of the atoms before each layer.

Weights are listed in config.yaml: the ones of the authors.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import yaml
from jaxtyping import Float, Int

from torchms.data.augmentation.transforms import Transform
from torchms.data.batching import Structures
from torchms.graph.geometry import (
    lattice_from_polar_parameters,
    polar_lattice_parameters,
)
from torchms.graph.transforms import AddCompleteGraph, attached_graph
from torchms.models.crystalflow.config import Config
from torchms.models.crystalflow.network import CSPNet, MultiEmbedding, TypeTableModule
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
    element_index,
    element_table,
)
from torchms.models.estimator.reader import Placeholder, read_pickle
from torchms.models.estimator.weights import download
from torchms.models.layers.basis import sinusoidal_embedding


@dataclass(frozen=True)
class State:
    """A point of the flow, or a velocity.

    Args:
        lattice: The six numbers of the polar decomposition of each cell,
            shape (n_struct, 6).
        x: Fractional coordinates of each atom, shape (n_atoms, 3).
        elements: Vector of the element of each atom, for the generation,
            shape (n_atoms, type_dim); None when the elements are given.
    """

    lattice: Float[torch.Tensor, "n_struct 6"]
    x: Float[torch.Tensor, "n_atoms 3"]
    elements: Float[torch.Tensor, "n_atoms type_dim"] | None = None


class Flow(PretrainedModel[Config]):  # pylint: disable=abstract-method
    """What the two models of CrystalFlow share: network, conditions and steps.

    Args:
        config: Configuration of the model, the one of config.yaml for its
            default weights by default.

    Attributes:
        GENERATION: Whether the model also generates the elements.
        condition_means: Mean of each property in the training set, read with
            the weights.
        condition_stds: Its deviation.
    """

    CONFIG = Config
    GENERATION = False
    GRAPH_FIELDS = ("graph",)

    condition_means: torch.Tensor
    condition_stds: torch.Tensor
    # row of each atomic number in the embedding, for the prediction of a structure
    elements: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(self.default_config() if config is None else config)
        network = self.config.model
        self.decoder = CSPNet(network)
        if network.conditions:
            self.cond_emb = MultiEmbedding(network)

        # what standardizes the properties, read with the weights
        num_conditions = len(network.conditions)
        self.register_buffer(
            "condition_means", torch.zeros(num_conditions), persistent=False
        )
        self.register_buffer(
            "condition_stds", torch.ones(num_conditions), persistent=False
        )

    @classmethod
    def default_config(cls) -> Config:
        """Configuration of the default weights of the model."""
        config = Config.load()
        if cls.GENERATION:
            return config.for_weights(config.generation.default_weights)

        return config

    @classmethod
    def for_weights(cls, name: str | None = None) -> Any:
        """Model of the network of some weights, before they are loaded.

        Args:
            name: Name of the weights, see Config.for_weights. The default
                ones of the model if None.
        """
        if name is None:
            return cls()

        return cls(Config.load().for_weights(name))

    @property
    def conditions(self) -> tuple[str, ...]:
        """Properties the model is conditioned on, by name."""
        return self.config.model.conditions

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download the archive of the weights and load the model of a name.

        The checkpoint, the configuration of the model and, with conditions,
        the standardization of its properties are read from the archive and
        checked against their SHA-256. These files hold objects of omegaconf
        and of CrystalFlow: they are read without importing them.
        """
        (archive_path,) = download("crystalflow", name, cache)
        members = self.config.upstream.members_of(name)

        wanted = {
            members.checkpoint: members.checkpoint_sha256,
            members.hparams: members.hparams_sha256,
        }
        if members.scalers is not None and members.scalers_sha256 is not None:
            wanted[members.scalers] = members.scalers_sha256
        files = {}
        with zipfile.ZipFile(archive_path) as archive:
            for member, expected in wanted.items():
                content = archive.read(member)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise InvalidWeights(
                        f"{member} of {archive_path} has another SHA-256"
                    )
                files[member] = content

        checkpoint = read_pickle(io.BytesIO(files[members.checkpoint]))
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(
                f"{members.checkpoint} is not a checkpoint of CrystalFlow"
            )
        hparams = yaml.safe_load(files[members.hparams])

        scalers = None
        if members.scalers is not None:
            scalers = scaler_values(read_pickle(io.BytesIO(files[members.scalers])))

        self.load_pretrained(checkpoint["state_dict"], hparams, scalers)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        hparams: Mapping[str, Any] | None = None,
        scalers: list[tuple[float, float]] | None = None,
    ) -> None:
        """Load the weights of a CSPFlow of CrystalFlow.

        Args:
            state_dict: Parameters of the model of CrystalFlow.
            hparams: Configuration of its training, checked if given.
            scalers: Mean and deviation of each property of its training set,
                in the order of the properties of the data of hparams, for a
                model with conditions.

        Raises:
            InvalidWeights: If parameters are missing or unknown, if the model
                was trained with another configuration, or if the
                standardization of a condition is missing.
        """
        if hparams is not None:
            check_configuration(hparams, self.config.upstream.sections)

        self.load_parameters(state_dict)

        if self.conditions:
            if hparams is None or scalers is None:
                raise InvalidWeights(
                    "a model with conditions needs their standardization"
                )
            properties = list(hparams["data"]["properties"])
            values = [scalers[properties.index(name)] for name in self.conditions]
            device = self.condition_means.device
            self.condition_means = torch.tensor(
                [mean for mean, _ in values], device=device
            )
            self.condition_stds = torch.tensor(
                [std for _, std in values], device=device
            )

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

    # the flow

    def embed_time(
        self, times: Float[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct time_dim"]:
        """Embedding of the time of each structure, read by the network."""
        network = self.config.model

        return sinusoidal_embedding(times, network.time_dim, network.time_max_period)

    def embed_conditions(
        self, conditions: Float[torch.Tensor, "n_struct n_conditions"]
    ) -> Float[torch.Tensor, "n_struct condition_dim"]:
        """Embedding of the properties of each structure, in the units of the data.

        Args:
            conditions: Value of each property of the model for each
                structure, in the order of conditions.

        Raises:
            ValueError: If the model has no conditions, or another number.
        """
        if not self.conditions or conditions.shape[-1] != len(self.conditions):
            raise ValueError(
                f"the model is conditioned on {list(self.conditions)}, "
                f"not on {conditions.shape[-1]} properties"
            )
        standardized = (conditions - self.condition_means) / self.condition_stds

        return self.cond_emb(standardized)

    def network_velocity(
        self,
        structures: Structures,
        state: State,
        time: float,
        condition: Float[torch.Tensor, "n_struct condition_dim"] | None = None,
    ) -> State:
        """Velocity the network gives at a point and a time, see velocity."""
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        if self.GENERATION:
            assert state.elements is not None
            element = state.elements
        else:
            element = element_index(structures.z, self.elements)

        n_struct = len(structures.num_atoms)
        times = torch.full((n_struct,), time, device=structures.device())
        lattice, x, elements = self.decoder(
            self.embed_time(times),
            element,
            graph,
            state.x,
            state.lattice,
            lattice_from_polar_parameters(state.lattice),
            condition,
        )

        return State(lattice=lattice, x=x, elements=elements)

    def velocity(
        self,
        structures: Structures,
        state: State,
        time: float,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
    ) -> State:
        """Velocity the network gives at a point of the flow and a time, unguided.

        Args:
            structures: Structures with their complete graph, see transform:
                their graph and, for the prediction of a structure, their
                atomic numbers are read; the point is the state.
            state: Point of the flow.
            time: Time of the point, from 0 to 1.
            conditions: Properties of each structure, for a model with
                conditions; None for its velocity without them.

        Raises:
            ValueError: If an atom has an element after fermium, if the
                structures have no graph, or if the conditions are not the
                ones of the model.
        """
        condition = None if conditions is None else self.embed_conditions(conditions)

        return self.network_velocity(structures, state, time, condition)

    def sample_step(  # pylint: disable=too-many-positional-arguments,too-many-arguments,too-many-locals
        self,
        structures: Structures,
        state: State,
        step: int,
        num_steps: int | None = None,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
        guide_factor: float | None = None,
        anneal_slope: float | None = None,
        anneal_offset: float | None = None,
        anneal_elements: bool = False,
    ) -> State:
        """One Euler step of the flow, to the time step / num_steps.

        The network is read at the time of the end of the step, as CrystalFlow
        does: the first step reads the time 1 / num_steps, the last one 1. The
        velocity of the coordinates is annealed, and they are wrapped in the
        cell after the step. With conditions, the velocity is guide_factor
        times the one with them plus 1 - guide_factor times the one without;
        with a factor of 1 only the first one is computed.

        Args:
            structures: Structures with their complete graph, see velocity.
            state: Point of the flow before the step.
            step: Number of the step, from 1 to num_steps.
            num_steps: Number of steps of the flow, the one of the
                configuration by default.
            conditions: Properties of each structure, for a model with
                conditions, in the units of the data.
            guide_factor: Factor of the guidance, the one of the configuration
                by default.
            anneal_slope: The velocity of the coordinates is multiplied by
                1 + slope * max(t - offset, 0). The one of the configuration
                by default.
            anneal_offset: The offset of this factor.
            anneal_elements: If True, the velocity of the elements without
                conditions is annealed too (not the one with them, as in
                CrystalFlow).

        Returns:
            Point of the flow after the step.

        Raises:
            ValueError: If a model with conditions is not given them, or a
                model without is.
        """
        sampling = self.config.sampling
        num_steps = sampling.num_steps if num_steps is None else num_steps
        slope = sampling.anneal_slope if anneal_slope is None else anneal_slope
        offset = sampling.anneal_offset if anneal_offset is None else anneal_offset
        factor_guide = sampling.guide_factor if guide_factor is None else guide_factor
        if bool(self.conditions) != (conditions is not None):
            raise ValueError(
                f"the model is conditioned on {list(self.conditions)}: "
                "give them, and only them"
            )

        time = step / num_steps
        # as CrystalFlow: the time in float32, then the offset
        factor = 1 + slope * torch.relu(
            torch.tensor(time, device=state.x.device) - offset
        )

        def annealed(velocity: State, elements: bool) -> State:
            moved = velocity.elements
            if elements and moved is not None:
                moved = moved * factor
            return replace(velocity, x=velocity.x * factor, elements=moved)

        if conditions is None:
            velocity = annealed(
                self.network_velocity(structures, state, time), anneal_elements
            )
        else:
            guided = annealed(
                self.velocity(structures, state, time, conditions), elements=False
            )
            if abs(factor_guide - 1) < 1e-4:
                velocity = guided
            else:
                free = annealed(
                    self.network_velocity(structures, state, time), anneal_elements
                )
                velocity = mix(guided, free, factor_guide)

        elements = None
        if state.elements is not None and velocity.elements is not None:
            elements = state.elements + velocity.elements / num_steps

        return State(
            lattice=state.lattice + velocity.lattice / num_steps,
            x=(state.x + velocity.x / num_steps) % 1.0,
            elements=elements,
        )

    def state_from_noise(
        self,
        lattice: Float[torch.Tensor, "n_struct 6"],
        x: Float[torch.Tensor, "n_atoms 3"],
        elements: Float[torch.Tensor, "n_atoms type_dim"] | None = None,
    ) -> State:
        """Point of the base distribution from standard normal and uniform noise.

        The numbers of the cell are the normal noise times lattice_polar_sigma,
        1 added to the last one; the coordinates are the uniform noise, and
        the vectors of the elements the normal noise as they are.
        """
        lattice = lattice * self.config.flow.lattice_polar_sigma
        lattice = torch.cat([lattice[:, :5], lattice[:, 5:] + 1], dim=1)

        return State(lattice=lattice, x=x, elements=elements)

    def initial_state(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        generator: torch.Generator | None = None,
    ) -> State:
        """A point of the base distribution, drawn in the order of CrystalFlow.

        Args:
            num_atoms: Number of atoms of each structure.
            generator: Generator of the random draws, on the device of
                num_atoms.
        """
        device = num_atoms.device
        n_atoms = int(num_atoms.sum())
        lattice = torch.randn((len(num_atoms), 6), generator=generator, device=device)
        x = torch.rand((n_atoms, 3), generator=generator, device=device)
        elements = None
        if self.GENERATION:
            elements = torch.randn(
                (n_atoms, self.config.model.type_dim),
                generator=generator,
                device=device,
            )

        return self.state_from_noise(lattice, x, elements)


def mix(guided: State, free: State, factor: float) -> State:
    """factor times a velocity with conditions plus 1 - factor times the one without."""

    def combined(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
        if a is None or b is None:
            return None
        return factor * a + (1 - factor) * b

    lattice, x = combined(guided.lattice, free.lattice), combined(guided.x, free.x)
    assert lattice is not None and x is not None

    return State(
        lattice=lattice, x=x, elements=combined(guided.elements, free.elements)
    )


def scaler_values(scalers: Any) -> list[tuple[float, float]]:
    """Mean and deviation of each StandardScalerTorch of a file of CrystalFlow.

    Raises:
        InvalidWeights: If the file does not hold a list of them.
    """
    if not isinstance(scalers, list) or not all(
        isinstance(scaler, Placeholder)
        and {"means", "stds"} <= set(scaler.attributes())
        for scaler in scalers
    ):
        raise InvalidWeights(
            "the standardization of the properties is not a list of scalers"
        )

    return [
        (float(scaler.attributes()["means"]), float(scaler.attributes()["stds"]))
        for scaler in scalers
    ]


class CrystalFlow(Flow):  # pylint: disable=abstract-method
    """CrystalFlow: the lattice and the coordinates of a crystal from its composition.

    The model does not build the complete graph its network reads, see
    transform. The sampling builds it once: it only depends on the numbers of
    atoms. The weights "crystalflow-csp-calypso60-pressure" are conditioned
    on the pressure, in GPa.

    Args:
        config: Configuration of the model, the one of config.yaml by default.

    Examples:
        The velocity of two crystals at the time 0.5:

        >>> model = CrystalFlow().eval()
        >>> structures = model.transform()(
        ...     Structures.collate([Structures.diamond(), Structures.zincblende()])
        ... )
        >>> with torch.no_grad():
        ...     velocity = model.velocity(structures, model.state_of(structures), 0.5)
        >>> velocity.lattice.shape, velocity.x.shape
        (torch.Size([2, 6]), torch.Size([16, 3]))

        With pretrained weights, `model.sample(structures.z, structures.num_atoms)`
        gives a lattice and coordinates to the atoms of these structures.
    """

    def __init__(self, config: Config | None = None):
        super().__init__(config)

        # row of each atomic number in the embedding, from hydrogen
        num_elements = self.config.model.num_elements
        self.register_buffer(
            "elements",
            element_table(torch.arange(1, num_elements + 1)),
            persistent=False,
        )

    def state_of(self, structures: Structures) -> State:
        """The point of the flow of crystals: their cells and their atoms."""
        return State(
            lattice=polar_lattice_parameters(structures.lattice), x=structures.x
        )

    def to_structures(
        self,
        state: State,
        z: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Structures:
        """Crystals of a point of the flow, with the symmetric lattices of k."""
        return Structures(
            lattice=lattice_from_polar_parameters(state.lattice),
            x=state.x % 1.0,
            z=z,
            num_atoms=num_atoms,
        )

    def integrate(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        state: State,
        z: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
        guide_factor: float | None = None,
        anneal_slope: float | None = None,
        anneal_offset: float | None = None,
    ) -> State:
        """The flow from a point at the time 0 to the time 1, see sample_step.

        Args:
            state: Point of the flow at the time 0, see initial_state.
            z: Atomic number of each atom.
            num_atoms: Number of atoms of each structure.
            num_steps: Number of steps, the one of the configuration by default.
            conditions: See sample_step.
            guide_factor: See sample_step.
            anneal_slope: See sample_step.
            anneal_offset: See sample_step.
        """
        num_steps = self.config.sampling.num_steps if num_steps is None else num_steps
        structures = self.transform(self.config)(
            self.to_structures(state, z, num_atoms)
        )

        for step in range(1, num_steps + 1):
            state = self.sample_step(
                structures,
                state,
                step,
                num_steps,
                conditions,
                guide_factor,
                anneal_slope,
                anneal_offset,
            )

        return state

    def sample(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        z: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
        guide_factor: float | None = None,
        anneal_slope: float | None = None,
        anneal_offset: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Sample the lattice and the coordinates of structures of given atoms.

        Args:
            z: Atomic number of each atom, the atoms of a structure together,
                on the device of the model.
            num_atoms: Number of atoms of each structure.
            num_steps: Number of steps of the flow, the one of the
                configuration by default: one call of the network each, two
                with conditions and a guide_factor other than 1.
            conditions: Properties of each structure, for a model with
                conditions, shape (n_struct, n_conditions).
            guide_factor: See sample_step.
            anneal_slope: See sample_step.
            anneal_offset: See sample_step.
            generator: Generator of the random draws, on the device of z.

        Returns:
            Structures sampled, without graph, with symmetric lattices.

        Raises:
            ValueError: If an atom has an element after fermium.
        """
        element_index(z, self.elements)
        state = self.integrate(
            self.initial_state(num_atoms, generator),
            z,
            num_atoms,
            num_steps,
            conditions,
            guide_factor,
            anneal_slope,
            anneal_offset,
        )

        return self.to_structures(state, z, num_atoms)


class CrystalFlowGeneration(Flow):  # pylint: disable=abstract-method
    """CrystalFlow which also generates the elements of the atoms.

    The element of an atom is two one-hot vectors of its row and of its
    column in the table of the elements of CrystalFlow; it starts from normal
    noise and gets its velocity from the network. The element of a vector is
    its largest row, then its largest column of an element in this row. The
    weights "crystalflow-dng-mp20-formation-energy" are conditioned on the
    formation energy per atom, in eV.

    Examples:
        The velocity of two crystals at the time 0.5:

        >>> model = CrystalFlowGeneration().eval()
        >>> structures = model.transform()(
        ...     Structures.collate([Structures.diamond(), Structures.zincblende()])
        ... )
        >>> with torch.no_grad():
        ...     velocity = model.velocity(structures, model.state_of(structures), 0.5)
        >>> velocity.elements.shape
        torch.Size([16, 28])

        With pretrained weights, `model.sample(model.sample_num_atoms(8))`
        generates 8 crystals.
    """

    GENERATION = True

    num_atoms_distribution: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.type_encoding = TypeTableModule(self.config.generation)
        # the network of CrystalFlow holds the same module
        self.decoder.type_encoding = self.type_encoding
        self.register_buffer(
            "num_atoms_distribution",
            torch.tensor(self.config.generation.num_atoms_distribution),
            persistent=False,
        )

    def state_of(self, structures: Structures) -> State:
        """The point of the flow of crystals, with the vectors of their elements."""
        return State(
            lattice=polar_lattice_parameters(structures.lattice),
            x=structures.x,
            elements=self.type_encoding.encode(structures.z),
        )

    def to_structures(
        self, state: State, num_atoms: Int[torch.Tensor, "n_struct"]
    ) -> Structures:
        """Crystals of a point of the flow, the elements read from their vectors."""
        assert state.elements is not None

        return Structures(
            lattice=lattice_from_polar_parameters(state.lattice),
            x=state.x % 1.0,
            z=self.type_encoding.decode(state.elements),
            num_atoms=num_atoms,
        )

    def sample_num_atoms(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> Int[torch.Tensor, "num_samples"]:
        """Draw numbers of atoms as the ones of the training set of MP-20."""
        return torch.multinomial(
            self.num_atoms_distribution,
            num_samples,
            replacement=True,
            generator=generator,
        )

    def integrate(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        state: State,
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
        guide_factor: float | None = None,
        anneal_slope: float | None = None,
        anneal_offset: float | None = None,
        anneal_elements: bool = False,
    ) -> State:
        """The flow from a point at the time 0 to the time 1, see sample_step."""
        num_steps = self.config.sampling.num_steps if num_steps is None else num_steps
        structures = self.transform(self.config)(self.to_structures(state, num_atoms))

        for step in range(1, num_steps + 1):
            state = self.sample_step(
                structures,
                state,
                step,
                num_steps,
                conditions,
                guide_factor,
                anneal_slope,
                anneal_offset,
                anneal_elements,
            )

        return state

    def sample(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        conditions: Float[torch.Tensor, "n_struct n_conditions"] | None = None,
        guide_factor: float | None = None,
        anneal_slope: float | None = None,
        anneal_offset: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Generate crystals of given numbers of atoms, with their elements.

        Args:
            num_atoms: Number of atoms of each structure, on the device of the
                model, see sample_num_atoms.
            num_steps: Number of steps of the flow, the one of the
                configuration by default.
            conditions: Properties of each structure, for a model with
                conditions, shape (n_struct, n_conditions).
            guide_factor: See sample_step.
            anneal_slope: See sample_step.
            anneal_offset: See sample_step.
            generator: Generator of the random draws, on the device of
                num_atoms.
        """
        state = self.integrate(
            self.initial_state(num_atoms, generator),
            num_atoms,
            num_steps,
            conditions,
            guide_factor,
            anneal_slope,
            anneal_offset,
        )

        return self.to_structures(state, num_atoms)
