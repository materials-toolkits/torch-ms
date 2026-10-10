# Derived from mace (https://github.com/ACEsuit/mace), Copyright (c) 2022
# ACEsuit/mace, MIT license (see LICENSE in this directory), and from e3nn
# (https://github.com/e3nn/e3nn, MIT license) for the normalization of its
# linear layers and tensor products.
#
# Changes: this file is a rewrite of the energy prediction of the MACE-MP-0
# "small" model, whose features are invariant, for inference only. Forces,
# stress, the other models and blocks of mace, training and the conversion from
# ASE are removed. The layers of e3nn are written with torch alone, graphs are
# built by torchms.graph, and batches are torchms Structures.
"""Energy prediction of the pretrained MACE-MP-0 small model.

MACE only needs the pairs of atoms closer than 6 Å: there are no triplets in
its graph. Terms on several atoms come from products. In each of its two
layers, an atom first sums over its neighbours a radial function times the
spherical harmonics of the direction (l up to 3) times the features of the
neighbour: this is a description of its environment, which depends on two
atoms. The products of two and of three of these sums are then contracted to
invariant features, with coupling tensors stored with the weights: they depend
on three and four atoms. The energy is a sum over the atoms of a read-out of
each layer, and of a reference energy per element.

Weights are the ones published with mace, see config.yaml. They are a pickled
model, read without mace by reader.py.
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.models.mace.config import Config, NetworkConfig
from torchms.models.mace.reader import read_model
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.graph.geometry import cartesian
from torchms.graph.graph import Graph
from torchms.graph.transforms import AddRadiusGraph, attached_graph
from torchms.models.layers.basis import (
    DEGREES,
    NUM_HARMONICS,
    polynomial_cutoff,
    sine_bessel,
    spherical_harmonics,
)


def weight(*shape: int) -> nn.Parameter:
    """Parameter of a layer, drawn from a normal distribution as in e3nn."""
    return nn.Parameter(torch.randn(*shape, dtype=torch.float64))


def linear(
    features: Float[torch.Tensor, "n f"], matrix: Float[torch.Tensor, "f d"]
) -> Float[torch.Tensor, "n d"]:
    """Linear layer of e3nn, normalized by the square root of the size of its input."""
    return features @ matrix / math.sqrt(matrix.shape[0])


class Interaction(nn.Module):
    """Environment of each atom: a sum over its neighbours.

    The features of a neighbour are multiplied by a function of the distance,
    learned for each degree l, and by the spherical harmonics of the direction.
    """

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.num_channels, self.num_elements = (
            network.num_channels,
            network.num_elements,
        )
        self.num_degrees = network.max_l + 1
        channels, hidden = network.num_channels, network.radial_hidden

        self.linear_up = weight(channels, channels)
        self.radial = nn.ParameterList(
            [
                weight(network.num_bessel, hidden),
                weight(hidden, hidden),
                weight(hidden, hidden),
                weight(hidden, self.num_degrees * channels),
            ]
        )
        self.linear = weight(self.num_degrees, channels, channels)
        self.skip = weight(channels, network.num_elements, channels)

    def radial_weights(
        self, radial: Float[torch.Tensor, "n_edges b"], activation: float
    ) -> Float[torch.Tensor, "n_edges 4 c"]:
        """Weight of each edge for each degree l and each channel."""
        for i, matrix in enumerate(self.radial):
            radial = linear(radial, matrix)
            if i < len(self.radial) - 1:
                radial = activation * nn.functional.silu(radial)

        return radial.view(-1, self.num_degrees, self.num_channels)

    def forward(
        self,
        features: Float[torch.Tensor, "n_atoms c"],
        element: Int[torch.Tensor, "n_atoms"],
        graph: Graph,
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
        constants: tuple[float, float],
    ) -> tuple[Float[torch.Tensor, "n_atoms c 16"], Float[torch.Tensor, "n_atoms c"]]:
        activation, avg_num_neighbors = constants
        degrees = torch.tensor(DEGREES, device=features.device)

        # features of the atom itself, mixed with weights of its element
        skip = torch.einsum("bu,ubk->bk", features, self.skip[:, element]) / math.sqrt(
            self.num_channels * self.num_elements
        )

        features = linear(features, self.linear_up)

        # (n_edges, c, 16): one weight per degree, repeated on its harmonics
        weights = self.radial_weights(radial, activation)[:, degrees].transpose(1, 2)
        neighbors = graph.unpool_senders(features)
        messages = weights * neighbors.unsqueeze(-1) * harmonics.unsqueeze(1)
        received = graph.aggregate_edges_to_nodes(messages)

        # channels are mixed with one matrix per degree
        mixed = torch.einsum("bum,muv->bvm", received, self.linear[degrees])
        mixed = mixed / (math.sqrt(self.num_channels) * avg_num_neighbors)

        return mixed, skip


class Product(nn.Module):
    """Invariant products of one, two and three environments of an atom.

    The tensors U couple one, two and three sets of spherical harmonics to an
    invariant. Their weights depend on the element of the atom.
    """

    U_matrix_1: torch.Tensor
    U_matrix_2: torch.Tensor
    U_matrix_3: torch.Tensor

    def __init__(self, network: NetworkConfig):
        super().__init__()
        num_couplings = network.num_couplings
        elements, channels = network.num_elements, network.num_channels

        shape = (NUM_HARMONICS,)
        for order, num in enumerate(num_couplings, start=1):
            self.register_buffer(
                f"U_matrix_{order}",
                torch.zeros(*shape * order, num, dtype=torch.float64),
            )

        self.weights_1 = weight(elements, num_couplings[0], channels)
        self.weights_2 = weight(elements, num_couplings[1], channels)
        self.weights_3 = weight(elements, num_couplings[2], channels)
        self.linear = weight(channels, channels)

    def contract(
        self,
        environment: Float[torch.Tensor, "n_atoms c 16"],
        element: Int[torch.Tensor, "n_atoms"],
    ) -> Float[torch.Tensor, "n_atoms c"]:
        """Sum of the invariants of order one, two and three of the environment."""
        # order three, contracted with one environment
        weighted = environment.unsqueeze(-1) * self.weights_3[element].transpose(
            1, 2
        ).unsqueeze(2)
        out = torch.einsum("wxik,bcik->bcwx", self.U_matrix_3, weighted)

        # plus order two, contracted with a second environment
        out = out + torch.einsum(
            "wxk,bkc->bcwx", self.U_matrix_2, self.weights_2[element]
        )
        out = torch.einsum("bcwi,bci->bcw", out, environment)

        # plus order one, contracted with a third one
        out = out + torch.einsum(
            "wk,bkc->bcw", self.U_matrix_1, self.weights_1[element]
        )
        return torch.einsum("bci,bci->bc", out, environment)

    def forward(
        self,
        environment: Float[torch.Tensor, "n_atoms c 16"],
        skip: Float[torch.Tensor, "n_atoms c"],
        element: Int[torch.Tensor, "n_atoms"],
    ) -> Float[torch.Tensor, "n_atoms c"]:
        contracted = self.contract(environment, element)

        return linear(contracted, self.linear) + skip


# one tensor per group of weights of mace
# pylint: disable-next=too-many-instance-attributes
class MACE(PretrainedModel[Config]):
    """Total energy of crystal structures predicted by MACE-MP-0 small.

    The model computes in double precision, as mace.

    Examples:
        >>> model = MACE()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    CONFIG = Config

    atomic_numbers: torch.Tensor
    atomic_energies: torch.Tensor
    bessel_weights: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network, radius = self.config.model, self.config.graph.radius
        elements, channels = network.num_elements, network.num_channels

        self.register_buffer("atomic_numbers", torch.arange(1, elements + 1))
        self.register_buffer(
            "atomic_energies", torch.zeros(elements, dtype=torch.float64)
        )
        self.register_buffer(
            "bessel_weights",
            math.pi
            / radius
            * torch.arange(1, network.num_bessel + 1, dtype=torch.float64),
        )
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("shift", torch.tensor(0.0, dtype=torch.float64))

        self.embedding = weight(elements, channels)
        self.interactions = nn.ModuleList(
            Interaction(network) for _ in range(network.num_layers)
        )
        self.products = nn.ModuleList(
            Product(network) for _ in range(network.num_layers)
        )
        self.readout_0 = weight(channels, 1)
        self.readout_1 = nn.ParameterList(
            [weight(channels, network.readout_dim), weight(network.readout_dim, 1)]
        )

        # constant of the normalized activation of e3nn, and mean number of
        # neighbours of an atom in the training of the model
        self.activation = network.activation
        self.avg_num_neighbors = 1.0

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        Raises:
            InvalidWeights: If the file is not a MACE-MP-0 small model.
        """
        (path,) = download("mace", name, cache)
        state_dict, modules = read_model(path)

        interactions = [
            module
            for name, module in modules.items()
            if name.startswith("interactions.") and name.count(".") == 1
        ]
        upstream = self.config.upstream
        if modules[""].path != upstream.model_class or not interactions:
            raise InvalidWeights(f"{modules[''].path} is not a {upstream.model_class}")
        if any(module.path != upstream.interaction_class for module in interactions):
            raise InvalidWeights(
                f"the interactions are not {upstream.interaction_class}"
            )

        # constants of the normalized activations of e3nn
        constants = {
            name: float(module.attributes()["cst"])
            for name, module in modules.items()
            if "cst" in module.attributes()
        }

        self.load_pretrained(
            state_dict,
            float(interactions[0].attributes()["avg_num_neighbors"]),
            constants,
        )

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        avg_num_neighbors: float,
        activation_constants: Mapping[str, float] | None = None,
    ) -> None:
        """Load the weights of a MACE-MP-0 small model of mace.

        Args:
            state_dict: Parameters of the model of mace.
            avg_num_neighbors: Mean number of neighbours of an atom, an
                attribute of the interactions of the model of mace.
            activation_constants: Constants of the normalized activations of
                e3nn in the model of mace, which must all be equal.

        Raises:
            InvalidWeights: If parameters are missing or do not have the shape
                or the values of a MACE-MP-0 small model.
        """
        for name, value in self.config.expected_values().items():
            if name not in state_dict or float(state_dict[name]) != value:
                raise InvalidWeights(f"{name} must be {value}")

        own = {}
        for name, (target, shape) in self.config.upstream_parameters().items():
            if name not in state_dict:
                raise InvalidWeights(f"Missing parameter {name}")
            if state_dict[name].numel() != math.prod(shape):
                raise InvalidWeights(
                    f"{name} has {state_dict[name].numel()} values, expected {shape}"
                )
            own[target] = state_dict[name].reshape(shape)

        own["atomic_energies"] = own["atomic_energies"].double()

        self.load_parameters(own)

        if activation_constants:
            constants = set(activation_constants.values())
            if len(constants) != 1:
                raise InvalidWeights(f"different activation constants: {constants}")
            self.activation = constants.pop()

        self.avg_num_neighbors = float(avg_num_neighbors)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the graph of the model.

        Atoms closer than the radius, 6 Å, are connected, periodic images of
        the atom itself included.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return Compose([AddRadiusGraph(graph.radius, self_images=True)])

    def edges(
        self, structures: Structures
    ) -> tuple[Graph, Float[torch.Tensor, "n_edges 3"]]:
        """Graph of the structures and vector of each edge in Å, in double precision.

        The graph is the one attached to the structures by transform. The
        vector of an edge goes from its sender to its receiver.

        Raises:
            ValueError: If the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        x = structures.x.double()

        fractional = (
            graph.unpool_receivers(x) + graph.edge_shift - graph.unpool_senders(x)
        )
        vectors = cartesian(fractional, structures.lattice.double(), graph.edge_batch)

        return graph, vectors

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        element = element_index(structures.z, element_table(self.atomic_numbers))

        graph, vectors = self.edges(structures)
        distance = vectors.norm(dim=-1)

        harmonics = spherical_harmonics(vectors)
        network, radius = self.config.model, self.config.graph.radius
        envelope = polynomial_cutoff(distance, radius, p=network.cutoff_power)
        radial = sine_bessel(
            distance, self.bessel_weights, math.sqrt(2.0 / radius)
        ) * envelope.unsqueeze(-1)

        features = self.embedding[element] / math.sqrt(network.num_elements)
        constants = (self.activation, self.avg_num_neighbors)

        per_layer = []
        for interaction, product in zip(self.interactions, self.products):
            environment, skip = interaction(
                features, element, graph, radial, harmonics, constants
            )
            features = product(environment, skip, element)
            per_layer.append(features)

        # read-out of each layer: linear for the first one, a network for the last
        per_atom = linear(per_layer[0], self.readout_0)
        hidden = self.activation * nn.functional.silu(
            linear(per_layer[1], self.readout_1[0])
        )
        per_atom = per_atom + linear(hidden, self.readout_1[1])

        per_atom = self.scale * per_atom.squeeze(-1) + self.shift
        per_atom = per_atom + self.atomic_energies[element]

        return graph.aggregate_nodes_to_structures(per_atom)
