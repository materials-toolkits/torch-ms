# Derived from FlowMM (https://github.com/facebookresearch/flowmm), Copyright (c)
# Meta Platforms, Inc. and affiliates, CC BY-NC 4.0 (see LICENSE in this
# directory), and from riemannian-fm, its submodule, under the same license.
# Its CSPNet extends the one of DiffCSP (MIT license, see
# torchms/models/diffcsp/LICENSE), whose layers are torchms.models.layers.cspnet.
#
# Changes: this file is a rewrite of the de novo generation of FlowMM
# (src/flowmm/model/arch.py, model_pl.py, solvers.py, rfm/manifold_getter.py
# and rfm/manifolds/{analog_bits,flat_torus,lattice_params}.py, and the Euler
# integrator of manifm/solvers.py), for inference only, in the variant of its
# published configuration `model=abits_params`: analog bits, a flat torus
# without drift and lattice parameters. The losses, the training, the
# likelihood, the other manifolds (simplex, SPD matrices, non-symmetric
# lattices, a fixed first atom), the prediction of a structure from its
# composition, the self-conditioning, the nearest neighbours, FlowLLM and the
# metrics are removed. The manifolds of geoopt and the padded "flat"
# representation of FlowMM are replaced by a State of the atoms and of the
# structures; the complete graph is built by torchms.graph; the noise of the
# base distribution is drawn by torch with a generator. Names of modules and
# parameters follow FlowMM to load its weights.
"""FlowMM, Riemannian flow matching of crystals.

A crystal is three things on three manifolds: the element of each atom as 7
analog bits, its fractional coordinates on a flat torus, and the cell as its
lattice parameters, three lengths and three angles. The angles live in an
interval, [59.9, 120.1] degrees, which a sigmoid maps from an unconstrained
space where the flow moves them. A network, a CSPNet, gives the velocity of
all of them at a time t: the flow is integrated by Euler steps from a base
distribution at the time 0 (normal bits, uniform coordinates, log-normal
lengths, uniform angles) to crystals at the time 1. The velocity of the
coordinates has no drift: their mean over the atoms of a structure does not
move.

Weights are listed in config.yaml: FlowMM trained on MP-20 by a third party,
the authors of ESS-Flow, as its authors publish none.
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
from torchms.graph.geometry import lattice_from_parameters, lattice_parameters
from torchms.graph.graph import Graph
from torchms.graph.transforms import AddCompleteGraph, attached_graph
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    check_configuration,
)
from torchms.models.estimator.weights import download
from torchms.models.flowmm.config import Config, FlowConfig, NetworkConfig
from torchms.models.layers import cspnet
from torchms.models.layers.basis import periodic_fourier


@dataclass(frozen=True)
class State:
    """A point of the flow, or a velocity: the atoms and the cells of structures.

    Args:
        bits: Analog bits of the element of each atom, shape (n_atoms,
            num_bits).
        x: Fractional coordinates of each atom, shape (n_atoms, 3).
        lattice: Lengths of each cell in Å, then its angles in the
            unconstrained space of the flow, shape (n_struct, 6).
    """

    bits: Float[torch.Tensor, "n_atoms num_bits"]
    x: Float[torch.Tensor, "n_atoms 3"]
    lattice: Float[torch.Tensor, "n_struct 6"]


def torus_difference(
    start: Float[torch.Tensor, "n 3"], end: Float[torch.Tensor, "n 3"]
) -> Float[torch.Tensor, "n 3"]:
    """Shortest difference from a point of the flat torus [0, 1)^3 to another.

    The logarithmic map of the torus: end - start modulo 1, in [-0.5, 0.5].

    Examples:
        >>> start = torch.tensor([[0.9, 0.1, 0.5]])
        >>> end = torch.tensor([[0.1, 0.8, 0.5]])
        >>> torus_difference(start, end)
        tensor([[ 0.2000, -0.3000,  0.0000]])
    """
    angle = 2 * math.pi * (end - start)

    return torch.atan2(torch.sin(angle), torch.cos(angle)) / (2 * math.pi)


def remove_drift(
    graph: Graph, velocity: Float[torch.Tensor, "n_atoms 3"]
) -> Float[torch.Tensor, "n_atoms 3"]:
    """Velocity of the atoms without its mean over each structure."""
    mean = graph.aggregate_nodes_to_structures(velocity, "mean")

    return velocity - graph.unpool_structures(mean)


class CSPLayer(cspnet.CSPLayer):
    """Layer of the CSPNet of FlowMM: the one of DiffCSP, reading more on an edge.

    The network of the messages reads the cosines of the edge with the
    lattice vectors, the features of its two atoms, the lattice parameters,
    the Fourier basis of the difference of the coordinates, and an embedding
    of the number of atoms of the structure, in this order.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__(config.hidden_dim, config.edge_dim)
        self.max_num_atoms = config.max_num_atoms
        self.num_atom_embedding = nn.Linear(
            config.max_num_atoms, config.hidden_dim, bias=False
        )

    def edge_inputs(
        self,
        graph: Graph,
        normalized: Float[torch.Tensor, "n_atoms d"],
        edge_features: Float[torch.Tensor, "n_edges edge_dim"],
    ) -> Float[torch.Tensor, "n_edges 3*d+edge_dim"]:
        """The edge features split around the features of the atoms.

        Args:
            graph: Complete graph of the atoms.
            normalized: Normalized features of the atoms.
            edge_features: The 3 cosines with the lattice vectors, then the
                lattice parameters and the Fourier basis.
        """
        # pylint: disable-next=not-callable
        one_hot = F.one_hot(graph.num_atoms, self.max_num_atoms).to(normalized.dtype)
        num_atoms = graph.unpool_structures(self.num_atom_embedding(one_hot))

        return torch.cat(
            [
                edge_features[:, :3],
                graph.unpool_senders(normalized),
                graph.unpool_receivers(normalized),
                edge_features[:, 3:],
                graph.unpool_senders(num_atoms),
            ],
            dim=1,
        )


class CSPNet(nn.Module):
    """CSPNet of FlowMM: a velocity of each atom and of each cell.

    Args:
        config: Hyperparameters of the network.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.time_emb = nn.Linear(1, config.time_dim, bias=False)
        self.node_embedding = nn.Linear(config.num_bits, hidden_dim, bias=False)
        self.atom_latent_emb = nn.Linear(
            hidden_dim + config.time_dim, hidden_dim, bias=False
        )
        for i in range(config.num_layers):
            self.add_module(f"csp_layer_{i}", CSPLayer(config))

        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        # the mean and the sum of the features of the atoms
        self.lattice_out = nn.Linear(2 * hidden_dim, 6)
        self.type_out = nn.Linear(hidden_dim, config.num_bits)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(  # pylint: disable=too-many-positional-arguments
        self,
        graph: Graph,
        time: Float[torch.Tensor, ""],
        bits: Float[torch.Tensor, "n_atoms num_bits"],
        x: Float[torch.Tensor, "n_atoms 3"],
        lattice: Float[torch.Tensor, "n_struct 6"],
        matrix: Float[torch.Tensor, "n_struct 3 3"],
    ) -> tuple[
        Float[torch.Tensor, "n_struct 6"],
        Float[torch.Tensor, "n_atoms 3"],
        Float[torch.Tensor, "n_atoms num_bits"],
    ]:
        """Outputs of the network, standardized as in its training.

        Args:
            graph: Complete graph of the atoms, without self loops.
            time: Time of the flow, from 0 to 1.
            bits: Analog bits of each atom.
            x: Fractional coordinates of each atom, in the cell.
            lattice: Standardized lattice parameters of each structure.
            matrix: Lattice vectors of each structure, as rows.

        Returns:
            Output for the lattice parameters, the coordinates and the bits.
        """
        # from the sender of an edge to its receiver, the shortest one on the torus
        difference = torus_difference(
            graph.unpool_senders(x), graph.unpool_receivers(x)
        )
        # cosines with the lattice vectors, normalized as the 3 of an edge: L L^T f
        products = graph.unpool_senders(
            graph.unpool_structures((matrix @ matrix.transpose(-1, -2)).flatten(1))
        ).view(-1, 3, 3)
        dots = torch.einsum("eij,ej->ei", products, difference)
        cosines = dots / dots.norm(dim=-1, keepdim=True)

        edge_features = torch.cat(
            [
                cosines,
                graph.unpool_senders(graph.unpool_structures(lattice)),
                periodic_fourier(difference, self.config.num_freqs),
            ],
            dim=1,
        )

        n_struct = len(graph.num_atoms)
        embedded_time = self.time_emb(time.reshape(1, 1)).expand(n_struct, -1)
        h = self.node_embedding(bits)
        h = self.atom_latent_emb(
            torch.cat([h, graph.unpool_structures(embedded_time)], dim=1)
        )
        for i in range(self.config.num_layers):
            h = self.get_submodule(f"csp_layer_{i}")(graph, h, edge_features)
        h = self.final_layer_norm(h)

        pooled = torch.cat(
            [
                graph.aggregate_nodes_to_structures(h, "mean"),
                graph.aggregate_nodes_to_structures(h, "sum"),
            ],
            dim=-1,
        )

        return self.lattice_out(pooled), self.coord_out(h), self.type_out(h)


class ProjectedConjugatedCSPNet(nn.Module):  # pylint: disable=abstract-method
    """The CSPNet with the standardization of its inputs and of its outputs.

    The buffers are the means and the deviations FlowMM computed on its
    training set; a model without weights has means of 0 and deviations of 1.

    Attributes:
        lat_x_t_mean: Mean of the lattice parameters on the paths of the flow.
        lat_x_t_std: Their deviation.
        lat_u_t_mean: Mean of their velocity.
        lat_u_t_std: Its deviation.
        coord_u_t_mean: Mean of the velocity of the coordinates.
        coord_u_t_std: Its deviation.
    """

    lat_x_t_mean: torch.Tensor
    lat_x_t_std: torch.Tensor
    lat_u_t_mean: torch.Tensor
    lat_u_t_std: torch.Tensor
    coord_u_t_mean: torch.Tensor
    coord_u_t_std: torch.Tensor

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.cspnet = CSPNet(config)
        for name, size in (("lat", 6), ("coord", 3)):
            for kind, value in (("mean", 0.0), ("std", 1.0)):
                for space in ("x_t", "u_t") if name == "lat" else ("u_t",):
                    self.register_buffer(
                        f"{name}_{space}_{kind}", torch.full((1, size), value)
                    )


class FlowMM(PretrainedModel[Config]):  # pylint: disable=abstract-method
    """FlowMM: generation of crystals by Riemannian flow matching.

    The model does not build the complete graph its network reads, see
    transform. The sampling builds it once, for the numbers of atoms given.

    Args:
        config: Configuration of the model, the one of config.yaml by default.

    Examples:
        The velocity of two crystals at the time 0.5:

        >>> model = FlowMM().eval()
        >>> structures = model.transform()(
        ...     Structures.collate([Structures.diamond(), Structures.zincblende()])
        ... )
        >>> state = model.state_of(structures)
        >>> with torch.no_grad():
        ...     velocity = model.velocity(structures, state, torch.tensor(0.5))
        >>> velocity.bits.shape, velocity.x.shape, velocity.lattice.shape
        (torch.Size([16, 7]), torch.Size([16, 3]), torch.Size([2, 6]))

        With pretrained weights, `model.sample(model.sample_num_atoms(8))`
        generates 8 crystals.
    """

    CONFIG = Config
    GRAPH_FIELDS = ("graph",)

    num_atoms_distribution: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.model = ProjectedConjugatedCSPNet(self.config.model)
        self.register_buffer(
            "num_atoms_distribution",
            torch.tensor(self.config.sampling.num_atoms_distribution),
            persistent=False,
        )

    # weights

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint of pytorch-lightning only holds tensors; the
        configuration of the training is a YAML file next to it.
        """
        checkpoint_path, config_path = download("flowmm", name, cache)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise InvalidWeights(f"{checkpoint_path} is not a checkpoint of FlowMM")

        with open(config_path, "r", encoding="utf-8") as fp:
            hparams = yaml.safe_load(fp)

        self.load_pretrained(checkpoint["state_dict"], hparams)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        hparams: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a MaterialsRFMLitModule of FlowMM.

        FlowMM evaluates its network with the moving average of its
        parameters, which the state dict has next to them: these are loaded,
        with the buffers of the network.

        Args:
            state_dict: State dict of the module of FlowMM.
            hparams: Configuration of its training, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, if the
                moving average does not have one value per parameter, or if
                the model was trained with another configuration.
        """
        upstream = self.config.upstream
        if hparams is not None:
            check_configuration(hparams, upstream.sections)
            found = hparams.get("base_distribution_from_data")
            if found != upstream.base_distribution_from_data:
                raise InvalidWeights(
                    f"weights of a model with base_distribution_from_data: {found}"
                )

        def renamed(name: str) -> str:
            return "model." + name[len(upstream.model_prefix) :]

        # the parameters of the network in the state dict, in their order:
        # everything under its prefix but its buffers
        buffers = set(self.state_dict()) - set(dict(self.named_parameters()))
        network = [
            name for name in state_dict if name.startswith(upstream.model_prefix)
        ]
        parameters = [name for name in network if renamed(name) not in buffers]

        averages = [f"{upstream.ema_prefix}{i}" for i in range(len(parameters))]
        num_averages = sum(name.startswith(upstream.ema_prefix) for name in state_dict)
        if num_averages != len(parameters) or any(
            name not in state_dict for name in averages
        ):
            raise InvalidWeights(
                f"{num_averages} averages of parameters for "
                f"{len(parameters)} parameters"
            )

        loaded = {renamed(name): state_dict[name] for name in network}
        loaded.update(
            {
                renamed(name): state_dict[average]
                for name, average in zip(parameters, averages)
            }
        )

        self.load_parameters(loaded)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the complete graph of each structure, without self loops.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return AddCompleteGraph(self_loops=graph.self_loops)

    # the spaces of the flow

    def _angle_bounds(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """First angle and width of the interval of the angles, rounded as in FlowMM."""
        flow: FlowConfig = self.config.flow
        bounds = torch.tensor(
            [flow.angle_min, flow.angle_max], dtype=like.dtype, device=like.device
        )

        return bounds[0], bounds[1] - bounds[0]

    def angles_in_degrees(
        self, unconstrained: Float[torch.Tensor, "... 3"]
    ) -> Float[torch.Tensor, "... 3"]:
        """Angles in degrees from the unconstrained space: a sigmoid, then the interval.

        The sigmoid is clipped as in torch.distributions, from the smallest
        positive number to 1 - epsilon.
        """
        finfo = torch.finfo(unconstrained.dtype)
        low, width = self._angle_bounds(unconstrained)
        unit = torch.sigmoid(unconstrained).clamp(min=finfo.tiny, max=1.0 - finfo.eps)

        return low + width * unit

    def unconstrained_angles(
        self, degrees: Float[torch.Tensor, "... 3"]
    ) -> Float[torch.Tensor, "... 3"]:
        """Angles in the unconstrained space, the inverse of angles_in_degrees."""
        finfo = torch.finfo(degrees.dtype)
        low, width = self._angle_bounds(degrees)
        unit = ((degrees - low) / width).clamp(min=finfo.tiny, max=1.0 - finfo.eps)

        return unit.log() - (-unit).log1p()

    def lattice_matrix(
        self, lattice: Float[torch.Tensor, "n_struct 6"]
    ) -> Float[torch.Tensor, "n_struct 3 3"]:
        """Lattice vectors, as rows, from the lattice parameters of the flow."""
        return lattice_from_parameters(
            lattice[:, :3], self.angles_in_degrees(lattice[:, 3:])
        )

    def analog_bits(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms num_bits"]:
        """Analog bits of atomic numbers: the bits of z - 1, as -scale or +scale.

        Examples:
            >>> FlowMM().analog_bits(torch.tensor([1, 6]))
            tensor([[-1., -1., -1., -1., -1., -1., -1.],
                    [ 1., -1.,  1., -1., -1., -1., -1.]])
        """
        num_bits = self.config.model.num_bits
        powers = torch.arange(num_bits, device=z.device)
        bits = torch.bitwise_right_shift((z - 1).unsqueeze(-1), powers) % 2

        return (bits * 2 - 1).float() * self.config.flow.analog_bits_scale

    def atomic_numbers(
        self, bits: Float[torch.Tensor, "n_atoms num_bits"]
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Atomic numbers from analog bits: a bit is set if it is positive.

        7 bits give atomic numbers from 1 to 128: the ones after 118 are not
        elements, as in FlowMM.

        Examples:
            >>> model = FlowMM()
            >>> model.atomic_numbers(model.analog_bits(torch.tensor([1, 26, 100])))
            tensor([  1,  26, 100])
        """
        powers = 2 ** torch.arange(bits.shape[-1], device=bits.device)

        return ((bits > 0).long() * powers).sum(dim=-1) + 1

    def state_of(self, structures: Structures) -> State:
        """The point of the flow of crystals."""
        lengths, angles = lattice_parameters(structures.lattice)

        return State(
            bits=self.analog_bits(structures.z),
            x=structures.x % 1.0,
            lattice=torch.cat([lengths, self.unconstrained_angles(angles)], dim=1),
        )

    def to_structures(
        self, state: State, num_atoms: Int[torch.Tensor, "n_struct"]
    ) -> Structures:
        """Crystals of a point of the flow, see atomic_numbers and lattice_matrix."""
        return Structures(
            lattice=self.lattice_matrix(state.lattice),
            x=state.x,
            z=self.atomic_numbers(state.bits),
            num_atoms=num_atoms,
        )

    # the flow

    def velocity(  # pylint: disable=too-many-positional-arguments
        self,
        structures: Structures,
        state: State,
        time: Float[torch.Tensor, ""],
        inference_anneal_slope: float | None = None,
        inference_anneal_offset: float | None = None,
    ) -> State:
        """Velocity of the flow at a point and a time.

        Args:
            structures: Structures with their complete graph, see transform:
                only the graph is read, the point is the state.
            state: Point of the flow. The coordinates are wrapped in the
                cell first.
            time: Time, from 0 to 1.
            inference_anneal_slope: The velocity of the coordinates is
                multiplied by 1 + slope * max(time - offset, 0). The one of
                the configuration by default.
            inference_anneal_offset: The offset of this factor.

        Returns:
            Velocity of the bits, of the coordinates, without drift, and of
            the lattice parameters.

        Raises:
            ValueError: If the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        network = self.model
        sampling = self.config.sampling

        x = state.x % 1.0
        standardized = (state.lattice - network.lat_x_t_mean) / network.lat_x_t_std
        lattice_out, coord_out, bits = network.cspnet(
            graph, time, state.bits, x, standardized, self.lattice_matrix(state.lattice)
        )

        lattice = lattice_out * network.lat_u_t_std + network.lat_u_t_mean
        coords = coord_out * network.coord_u_t_std + network.coord_u_t_mean
        coords = remove_drift(graph, coords)

        slope = (
            sampling.inference_anneal_slope
            if inference_anneal_slope is None
            else inference_anneal_slope
        )
        offset = (
            sampling.inference_anneal_offset
            if inference_anneal_offset is None
            else inference_anneal_offset
        )
        coords = coords * (slope * torch.relu(time - offset) + 1.0)

        return State(bits=bits, x=coords, lattice=lattice)

    def sample_step(  # pylint: disable=too-many-positional-arguments
        self,
        structures: Structures,
        state: State,
        time: Float[torch.Tensor, ""],
        next_time: Float[torch.Tensor, ""],
        inference_anneal_slope: float | None = None,
        inference_anneal_offset: float | None = None,
    ) -> State:
        """One Euler step of the flow, the coordinates wrapped in the cell after it.

        Args:
            structures: Structures with their complete graph, see velocity.
            state: Point of the flow at the time.
            time: Time of the point.
            next_time: Time after it.
            inference_anneal_slope: See velocity.
            inference_anneal_offset: See velocity.

        Returns:
            Point of the flow at the next time.
        """
        velocity = self.velocity(
            structures, state, time, inference_anneal_slope, inference_anneal_offset
        )
        step = next_time - time

        return State(
            bits=state.bits + step * velocity.bits,
            x=(state.x + step * velocity.x) % 1.0,
            lattice=state.lattice + step * velocity.lattice,
        )

    def times(self, num_steps: int | None = None) -> Float[torch.Tensor, "n_times"]:
        """Times of the steps, from 0 to 1 in float32, as FlowMM.

        Args:
            num_steps: Number of steps, the one of the configuration by
                default: the times are one more.
        """
        steps = self.config.sampling.num_steps if num_steps is None else num_steps
        device = self.num_atoms_distribution.device

        return torch.linspace(0.0, 1.0, steps + 1, device=device)

    def initial_state(
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        generator: torch.Generator | None = None,
    ) -> State:
        """A point of the base distribution of the flow.

        Normal bits, uniform coordinates, log-normal lengths with the
        statistics of MP-20, and uniform angles in their interval.

        Args:
            num_atoms: Number of atoms of each structure.
            generator: Generator of the random draws, on the device of
                num_atoms.
        """
        device = num_atoms.device
        n_atoms, n_struct = int(num_atoms.sum()), len(num_atoms)
        flow = self.config.flow

        def draw(*shape: int, uniform: bool = False) -> torch.Tensor:
            sample = torch.rand if uniform else torch.randn
            return sample(shape, generator=generator, device=device)

        bits = draw(n_atoms, self.config.model.num_bits)
        x = draw(n_atoms, 3, uniform=True)
        mean, deviation = (
            torch.tensor(values, device=device)
            for values in (flow.length_log_mean, flow.length_log_std)
        )
        lengths = torch.exp(mean + deviation * draw(n_struct, 3))
        low, width = self._angle_bounds(lengths)
        angles = self.unconstrained_angles(
            low + width * draw(n_struct, 3, uniform=True)
        )

        return State(bits=bits, x=x, lattice=torch.cat([lengths, angles], dim=1))

    def sample_num_atoms(
        self, num_samples: int, generator: torch.Generator | None = None
    ) -> Int[torch.Tensor, "num_samples"]:
        """Draw numbers of atoms as the ones of the training set of MP-20.

        Args:
            num_samples: Number of structures.
            generator: Generator of the random draws, on the device of the
                model.
        """
        return torch.multinomial(
            self.num_atoms_distribution,
            num_samples,
            replacement=True,
            generator=generator,
        )

    def integrate(  # pylint: disable=too-many-positional-arguments
        self,
        state: State,
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        inference_anneal_slope: float | None = None,
        inference_anneal_offset: float | None = None,
    ) -> State:
        """The flow from a point at the time 0 to the time 1, by Euler steps.

        The complete graph of the structures is built once: it only depends
        on the numbers of atoms.

        Args:
            state: Point of the flow at the time 0, see initial_state.
            num_atoms: Number of atoms of each structure.
            num_steps: Number of steps, the one of the configuration by
                default.
            inference_anneal_slope: See velocity.
            inference_anneal_offset: See velocity.
        """
        structures = self.transform(self.config)(self.to_structures(state, num_atoms))
        times = self.times(num_steps)

        for time, next_time in zip(times[:-1], times[1:]):
            state = self.sample_step(
                structures,
                state,
                time,
                next_time,
                inference_anneal_slope,
                inference_anneal_offset,
            )

        return state

    def sample(  # pylint: disable=too-many-positional-arguments
        self,
        num_atoms: Int[torch.Tensor, "n_struct"],
        num_steps: int | None = None,
        inference_anneal_slope: float | None = None,
        inference_anneal_offset: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        """Generate crystals of given numbers of atoms.

        Args:
            num_atoms: Number of atoms of each structure, on the device of the
                model, see sample_num_atoms.
            num_steps: Number of steps of the flow, the one of the
                configuration by default: one call of the network each.
            inference_anneal_slope: See velocity.
            inference_anneal_offset: See velocity.
            generator: Generator of the random draws, on the device of
                num_atoms.

        Returns:
            Crystals generated, without graph, see to_structures.
        """
        state = self.integrate(
            self.initial_state(num_atoms, generator),
            num_atoms,
            num_steps,
            inference_anneal_slope,
            inference_anneal_offset,
        )

        return self.to_structures(state, num_atoms)
