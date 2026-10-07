# Derived from mace (https://github.com/ACEsuit/mace), Copyright (c) 2022
# ACEsuit/mace, MIT license (see LICENSE in this directory), and from e3nn
# (https://github.com/e3nn/e3nn, MIT license) for the normalization of its
# linear layers and tensor products.
#
# Changes: this file is a rewrite of the energy prediction of the MACE-MP-0
# "small" model, whose features are invariant, for inference only. Forces,
# stress, the other models and blocks of mace, training and the conversion from
# ASE are removed. The layers of e3nn are written with torch alone, graphs are
# built by pycrystalgen.graph, and batches are pycrystalgen Structures.
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

The weights are converted from the ones of mace, see weights.py.
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from pycrystalgen.data.batching import Structures
from pycrystalgen.estimator.mace.weights import checkpoint_path
from pycrystalgen.estimator.orb.model import polynomial_cutoff, spherical_harmonics
from pycrystalgen.graph.cutoff import cutoff
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.sparse import scatter

RADIUS = 6.0
NUM_CHANNELS = 128
NUM_ELEMENTS = 89
NUM_BESSEL = 10
CUTOFF_POWER = 5
NUM_LAYERS = 2
MAX_L = 3
NUM_HARMONICS = (MAX_L + 1) ** 2
READOUT_DIM = 16

# degree l of each of the 16 spherical harmonics
DEGREES = tuple(degree for degree in range(MAX_L + 1) for _ in range(2 * degree + 1))


class InvalidWeights(Exception):
    """Raised when weights do not match the parameters of the model."""


def weight(*shape: int) -> nn.Parameter:
    """Parameter of a layer, drawn from a normal distribution as in e3nn."""
    return nn.Parameter(torch.randn(*shape, dtype=torch.float64))


class Interaction(nn.Module):
    """Environment of each atom: a sum over its neighbours.

    The features of a neighbour are multiplied by a function of the distance,
    learned for each degree l, and by the spherical harmonics of the direction.
    """

    def __init__(self):
        super().__init__()
        self.linear_up = weight(NUM_CHANNELS, NUM_CHANNELS)
        self.radial = nn.ParameterList(
            [
                weight(NUM_BESSEL, 64),
                weight(64, 64),
                weight(64, 64),
                weight(64, (MAX_L + 1) * NUM_CHANNELS),
            ]
        )
        self.linear = weight(MAX_L + 1, NUM_CHANNELS, NUM_CHANNELS)
        self.skip = weight(NUM_CHANNELS, NUM_ELEMENTS, NUM_CHANNELS)

    def radial_weights(
        self, radial: Float[torch.Tensor, "n_edges b"], activation: float
    ) -> Float[torch.Tensor, "n_edges 4 c"]:
        """Weight of each edge for each degree l and each channel."""
        for i, matrix in enumerate(self.radial):
            radial = radial @ matrix / math.sqrt(matrix.shape[0])
            if i < len(self.radial) - 1:
                radial = activation * nn.functional.silu(radial)

        return radial.view(-1, MAX_L + 1, NUM_CHANNELS)

    def forward(
        self,
        features: Float[torch.Tensor, "n_atoms c"],
        element: Int[torch.Tensor, "n_atoms"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
        constants: tuple[float, float],
    ) -> tuple[Float[torch.Tensor, "n_atoms c 16"], Float[torch.Tensor, "n_atoms c"]]:
        activation, avg_num_neighbors = constants
        sender, receiver = edge_index
        degrees = torch.tensor(DEGREES, device=features.device)

        # features of the atom itself, mixed with weights of its element
        skip = torch.einsum("bu,ubk->bk", features, self.skip[:, element]) / math.sqrt(
            NUM_CHANNELS * NUM_ELEMENTS
        )

        features = features @ self.linear_up / math.sqrt(NUM_CHANNELS)

        # (n_edges, c, 16): one weight per degree, repeated on its harmonics
        weights = self.radial_weights(radial, activation)[:, degrees].transpose(1, 2)
        messages = weights * features[sender].unsqueeze(-1) * harmonics.unsqueeze(1)
        received = scatter(messages, receiver, features.shape[0])

        # channels are mixed with one matrix per degree
        mixed = torch.einsum("bum,muv->bvm", received, self.linear[degrees])
        mixed = mixed / (math.sqrt(NUM_CHANNELS) * avg_num_neighbors)

        return mixed, skip


class Product(nn.Module):
    """Invariant products of one, two and three environments of an atom.

    The tensors U couple one, two and three sets of spherical harmonics to an
    invariant. Their weights depend on the element of the atom.
    """

    U_matrix_1: torch.Tensor
    U_matrix_2: torch.Tensor
    U_matrix_3: torch.Tensor

    def __init__(self, num_couplings: tuple[int, int, int] = (1, 4, 23)):
        super().__init__()
        shape = (NUM_HARMONICS,)
        for order, num in enumerate(num_couplings, start=1):
            self.register_buffer(
                f"U_matrix_{order}",
                torch.zeros(*shape * order, num, dtype=torch.float64),
            )

        self.weights_1 = weight(NUM_ELEMENTS, num_couplings[0], NUM_CHANNELS)
        self.weights_2 = weight(NUM_ELEMENTS, num_couplings[1], NUM_CHANNELS)
        self.weights_3 = weight(NUM_ELEMENTS, num_couplings[2], NUM_CHANNELS)
        self.linear = weight(NUM_CHANNELS, NUM_CHANNELS)

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

        return contracted @ self.linear / math.sqrt(NUM_CHANNELS) + skip


# parameters of mace → parameters of the rewrite, with their shape
def _parameters(layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    interaction, product = f"interactions.{layer}.", f"products.{layer}."
    contraction = product + "symmetric_contractions.contractions.0."
    c, e = NUM_CHANNELS, NUM_ELEMENTS

    return {
        interaction + "linear_up.weight": (f"interactions.{layer}.linear_up", (c, c)),
        interaction
        + "conv_tp_weights.layer0.weight": (
            f"interactions.{layer}.radial.0",
            (NUM_BESSEL, 64),
        ),
        interaction
        + "conv_tp_weights.layer1.weight": (
            f"interactions.{layer}.radial.1",
            (64, 64),
        ),
        interaction
        + "conv_tp_weights.layer2.weight": (
            f"interactions.{layer}.radial.2",
            (64, 64),
        ),
        interaction
        + "conv_tp_weights.layer3.weight": (
            f"interactions.{layer}.radial.3",
            (64, (MAX_L + 1) * c),
        ),
        interaction
        + "linear.weight": (
            f"interactions.{layer}.linear",
            (MAX_L + 1, c, c),
        ),
        interaction + "skip_tp.weight": (f"interactions.{layer}.skip", (c, e, c)),
        contraction + "U_matrix_1": (f"products.{layer}.U_matrix_1", (16, 1)),
        contraction + "U_matrix_2": (f"products.{layer}.U_matrix_2", (16, 16, 4)),
        contraction + "U_matrix_3": (f"products.{layer}.U_matrix_3", (16, 16, 16, 23)),
        contraction + "weights.1": (f"products.{layer}.weights_1", (e, 1, c)),
        contraction + "weights.0": (f"products.{layer}.weights_2", (e, 4, c)),
        contraction + "weights_max": (f"products.{layer}.weights_3", (e, 23, c)),
        product + "linear.weight": (f"products.{layer}.linear", (c, c)),
    }


PARAMETERS: dict[str, tuple[str, tuple[int, ...]]] = {
    "atomic_numbers": ("atomic_numbers", (NUM_ELEMENTS,)),
    "atomic_energies_fn.atomic_energies": ("atomic_energies", (NUM_ELEMENTS,)),
    "node_embedding.linear.weight": ("embedding", (NUM_ELEMENTS, NUM_CHANNELS)),
    "radial_embedding.bessel_fn.bessel_weights": ("bessel_weights", (NUM_BESSEL,)),
    "readouts.0.linear.weight": ("readout_0", (NUM_CHANNELS, 1)),
    "readouts.1.linear_1.weight": ("readout_1.0", (NUM_CHANNELS, READOUT_DIM)),
    "readouts.1.linear_2.weight": ("readout_1.1", (READOUT_DIM, 1)),
    "scale_shift.scale": ("scale", ()),
    "scale_shift.shift": ("shift", ()),
    **_parameters(0),
    **_parameters(1),
}

# values the rewrite relies on, checked in the weights of mace
EXPECTED = {
    "r_max": RADIUS,
    "num_interactions": NUM_LAYERS,
    "radial_embedding.cutoff_fn.p": CUTOFF_POWER,
    "radial_embedding.cutoff_fn.r_max": RADIUS,
    "radial_embedding.bessel_fn.r_max": RADIUS,
}


# one tensor per group of weights of mace
# pylint: disable-next=too-many-instance-attributes
class MaceEnergyModel(nn.Module):
    """Total energy of crystal structures predicted by MACE-MP-0 small.

    The model computes in double precision, as mace.

    Examples:
        >>> model = MaceEnergyModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    atomic_numbers: torch.Tensor
    atomic_energies: torch.Tensor
    bessel_weights: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.arange(1, NUM_ELEMENTS + 1))
        self.register_buffer(
            "atomic_energies", torch.zeros(NUM_ELEMENTS, dtype=torch.float64)
        )
        self.register_buffer(
            "bessel_weights",
            math.pi / RADIUS * torch.arange(1, NUM_BESSEL + 1, dtype=torch.float64),
        )
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("shift", torch.tensor(0.0, dtype=torch.float64))

        self.embedding = weight(NUM_ELEMENTS, NUM_CHANNELS)
        self.interactions = nn.ModuleList(Interaction() for _ in range(NUM_LAYERS))
        self.products = nn.ModuleList(Product() for _ in range(NUM_LAYERS))
        self.readout_0 = weight(NUM_CHANNELS, 1)
        self.readout_1 = nn.ParameterList(
            [weight(NUM_CHANNELS, READOUT_DIM), weight(READOUT_DIM, 1)]
        )

        # constant of the normalized activation of e3nn, and mean number of
        # neighbours of an atom in the training of the model
        self.activation = 1.6791767923989418
        self.avg_num_neighbors = 1.0

    @classmethod
    def from_pretrained(
        cls,
        name: str = "mace-mp-0-small",
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> MaceEnergyModel:
        """Load a model with pretrained weights, converted from the ones of mace.

        Args:
            name: Name of the weights, see pycrystalgen.estimator.mace.weights.
            device: Device of the model.
            cache: Directory of the converted weights.

        Raises:
            FileNotFoundError: If the weights were not converted.
        """
        path = checkpoint_path(name, cache)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist: the weights of mace must be converted "
                "once with scripts/reference_mace.py"
            )

        converted = torch.load(path, map_location="cpu", weights_only=True)

        model = cls()
        model.load_pretrained(
            converted["state_dict"],
            converted["avg_num_neighbors"],
            converted["activation_constants"],
        )

        return model.to(device).eval().requires_grad_(False)

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
        for name, value in EXPECTED.items():
            if name not in state_dict or float(state_dict[name]) != value:
                raise InvalidWeights(f"{name} must be {value}")

        own = {}
        for name, (target, shape) in PARAMETERS.items():
            if name not in state_dict:
                raise InvalidWeights(f"Missing parameter {name}")
            if state_dict[name].numel() != math.prod(shape):
                raise InvalidWeights(
                    f"{name} has {state_dict[name].numel()} values, expected {shape}"
                )
            own[target] = state_dict[name].reshape(shape)

        own["atomic_energies"] = own["atomic_energies"].double()

        try:
            self.load_state_dict(own, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e

        if activation_constants:
            constants = set(activation_constants.values())
            if len(constants) != 1:
                raise InvalidWeights(f"different activation constants: {constants}")
            self.activation = constants.pop()

        self.avg_num_neighbors = float(avg_num_neighbors)

    def edges(
        self, structures: Structures
    ) -> tuple[Graph, Float[torch.Tensor, "n_edges 3"]]:
        """Graph of the model and vector of each edge in Å, in double precision.

        Atoms closer than 6 Å are connected, periodic images of the atom itself
        included. The vector of an edge goes from its sender to its receiver.
        """
        graph = cutoff(structures, RADIUS, self_images=True)
        sender, receiver = graph.edge_index

        fractional = (
            structures.x.double()[receiver]
            + graph.edge_shift
            - structures.x.double()[sender]
        )
        vectors = torch.bmm(
            fractional.unsqueeze(1), structures.lattice.double()[graph.edge_batch]
        ).squeeze(1)

        return graph, vectors

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        known = structures.z.unsqueeze(-1) == self.atomic_numbers
        if not known.any(dim=-1).all():
            unknown = structures.z[~known.any(dim=-1)].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")
        element = known.long().argmax(dim=-1)

        graph, vectors = self.edges(structures)
        distance = vectors.norm(dim=-1, keepdim=True)

        harmonics = spherical_harmonics(vectors)
        radial = (
            math.sqrt(2.0 / RADIUS)
            * torch.sin(self.bessel_weights * distance)
            / distance
            * polynomial_cutoff(distance, RADIUS, p=CUTOFF_POWER)
        )

        features = self.embedding[element] / math.sqrt(NUM_ELEMENTS)
        constants = (self.activation, self.avg_num_neighbors)

        per_layer = []
        for interaction, product in zip(self.interactions, self.products):
            environment, skip = interaction(
                features, element, graph.edge_index, radial, harmonics, constants
            )
            features = product(environment, skip, element)
            per_layer.append(features)

        # read-out of each layer: linear for the first one, a network for the last
        per_atom = per_layer[0] @ self.readout_0 / math.sqrt(NUM_CHANNELS)
        hidden = per_layer[1] @ self.readout_1[0] / math.sqrt(NUM_CHANNELS)
        hidden = self.activation * nn.functional.silu(hidden)
        per_atom = per_atom + hidden @ self.readout_1[1] / math.sqrt(READOUT_DIM)

        per_atom = self.scale * per_atom.squeeze(-1) + self.shift
        per_atom = per_atom + self.atomic_energies[element]

        return scatter(per_atom, structures.batch, len(structures.num_atoms))
