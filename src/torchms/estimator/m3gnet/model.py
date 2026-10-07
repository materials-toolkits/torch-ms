# Derived from matgl (https://github.com/materialsvirtuallab/matgl), Copyright
# (c) 2022, Materials Virtual Lab, BSD 3-Clause license (see LICENSE in this
# directory).
#
# Changes: this file is a rewrite of the energy prediction of the M3GNet
# potential of matgl, for inference only. Forces, stress, state features, the
# other read-outs, training and the conversion from pymatgen are removed.
# Graphs are built by torchms.graph, the spherical Bessel functions and
# their roots are computed with torch and scipy instead of sympy, and batches
# are torchms Structures. Names of modules and parameters follow matgl to
# load its pretrained weights.
"""Energy prediction of the pretrained M3GNet potential.

The network has features on atoms and on bonds, the pairs of atoms closer than
5 Å. In each of its three blocks, a bond is first updated from the angles it
forms with the other bonds of its atom shorter than 4 Å, in a basis of
spherical Bessel functions of the length of the other bond and of spherical
harmonics of the angle. Bonds and atoms are then updated from each other. The
energy is a sum of an energy per atom and of a reference energy per element.

Weights are the ones published with matgl, see weights.py.
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.optimize import brentq
from scipy.special import spherical_jn
from torch import nn

from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number
from torchms.estimator.m3gnet.weights import download
from torchms.estimator.orb.model import polynomial_cutoff
from torchms.graph.cutoff import cutoff
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import Graph
from torchms.graph.line_graph import triplets
from torchms.graph.sparse import scatter

RADIUS = 5.0
THREE_BODY_RADIUS = 4.0
LATENT_DIM = 64
MAX_N = 3
MAX_L = 3
NUM_BLOCKS = 3

# atomic numbers of the elements of the model, in the order of its embedding:
# H to Bi, then Ac to Pu
ELEMENTS = (*range(1, 84), *range(89, 95))

# arguments of the model the rewrite is written for, as stored with the weights
MODEL_ARGS = {
    "dim_node_embedding": LATENT_DIM,
    "dim_edge_embedding": LATENT_DIM,
    "max_n": MAX_N,
    "max_l": MAX_L,
    "nblocks": NUM_BLOCKS,
    "rbf_type": "SphericalBessel",
    "is_intensive": False,
    "cutoff": RADIUS,
    "threebody_cutoff": THREE_BODY_RADIUS,
    "units": LATENT_DIM,
    "ntargets": 1,
    "use_smooth": False,
    "use_phi": False,
    "include_state": False,
    "activation_type": "swish",
}


class InvalidWeights(Exception):
    """Raised when weights do not match the parameters of the model."""


def spherical_bessel_roots(max_l: int, max_n: int) -> np.ndarray:
    """First roots of the spherical Bessel functions j_0 to j_(max_l - 1).

    The roots of j_l are between the ones of j_(l-1).

    Returns:
        Roots with shape (max_l, max_n).

    Examples:
        >>> spherical_bessel_roots(2, 2).round(4)
        array([[3.1416, 6.2832],
               [4.4934, 7.7253]])
    """
    zeros = np.arange(1, max_l + max_n + 1) * math.pi
    roots = [zeros[:max_n]]

    for degree in range(1, max_l):
        zeros = np.array(
            [
                brentq(lambda x, v=degree: spherical_jn(v, x), low, high)
                for low, high in zip(zeros[:-1], zeros[1:])
            ]
        )
        roots.append(zeros[:max_n])

    return np.array(roots)


def spherical_bessel(
    max_l: int, x: Float[torch.Tensor, "..."]
) -> list[Float[torch.Tensor, "..."]]:
    """Spherical Bessel functions j_0 to j_max_l, for x > 0.

    Examples:
        >>> j0, j1 = spherical_bessel(1, torch.tensor([math.pi / 2]))
        >>> j0, j1
        (tensor([0.6366]), tensor([0.4053]))
    """
    functions = [torch.sin(x) / x, torch.sin(x) / x**2 - torch.cos(x) / x]

    for degree in range(1, max_l):
        functions.append((2 * degree + 1) / x * functions[-1] - functions[-2])

    return functions[: max_l + 1]


def legendre(max_l: int, x: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n max_l"]:
    """Spherical harmonics Y_l0 of an angle from its cosine, for l < max_l.

    Examples:
        >>> legendre(3, torch.tensor([1.0, 0.0]))
        tensor([[ 0.2821,  0.4886,  0.6308],
                [ 0.2821,  0.0000, -0.3154]])
    """
    polynomials = [torch.ones_like(x), x]

    for degree in range(1, max_l - 1):
        polynomials.append(
            ((2 * degree + 1) * x * polynomials[-1] - degree * polynomials[-2])
            / (degree + 1)
        )

    return torch.stack(
        [
            math.sqrt((2 * degree + 1) / (4 * math.pi)) * polynomials[degree]
            for degree in range(max_l)
        ],
        dim=-1,
    )


class SphericalBesselFunction(nn.Module):
    """Radial basis j_l(z_ln r / r_max), with z_ln the roots of j_l.

    The functions are normalized on the sphere of radius r_max, and ordered by
    l then by n.
    """

    cutoff: torch.Tensor
    roots: torch.Tensor
    norm: torch.Tensor

    def __init__(self):
        super().__init__()
        roots = torch.tensor(spherical_bessel_roots(MAX_L, MAX_N), dtype=torch.float64)
        next_degree = torch.stack(
            [
                spherical_bessel(MAX_L, roots[degree])[degree + 1]
                for degree in range(MAX_L)
            ]
        )

        self.register_buffer("cutoff", torch.tensor(RADIUS))
        self.register_buffer("roots", roots.float(), persistent=False)
        self.register_buffer(
            "norm",
            (math.sqrt(2.0 / RADIUS**3) / next_degree.abs()).float(),
            persistent=False,
        )

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        scaled = (distance.clamp(max=self.cutoff) / self.cutoff).unsqueeze(-1)

        return torch.cat(
            [
                spherical_bessel(MAX_L, scaled * self.roots[degree])[degree]
                * self.norm[degree]
                for degree in range(MAX_L)
            ],
            dim=1,
        )


class BondExpansion(nn.Module):
    """Radial basis of the length of a bond."""

    def __init__(self):
        super().__init__()
        self.rbf = SphericalBesselFunction()

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        return self.rbf(distance)


class SphericalBesselWithHarmonics(nn.Module):
    """Basis of an angle: Bessel functions of a length times harmonics of the angle."""

    def __init__(self):
        super().__init__()
        self.sbf = SphericalBesselFunction()

    def forward(
        self, distance: Float[torch.Tensor, "n"], cosine: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        # the harmonic of degree l multiplies the max_n functions of degree l
        harmonics = legendre(MAX_L, cosine).repeat_interleave(MAX_N, dim=1)

        return self.sbf(distance) * harmonics


class MLP(nn.Module):
    """One linear layer followed by an activation."""

    def __init__(self, input_size: int, output_size: int, activation: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, output_size), activation])

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self.layers[1](self.layers[0](x))


class GatedMLP(nn.Module):
    """Product of a network and of a gate in [0, 1]."""

    def __init__(
        self, sizes: list[int], activate_last: bool = True, use_bias: bool = True
    ):
        super().__init__()
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()

        for i, (size_in, size_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            self.layers.append(nn.Linear(size_in, size_out, bias=use_bias))
            self.gates.append(nn.Linear(size_in, size_out, bias=use_bias))

            if not last or activate_last:
                self.layers.append(nn.SiLU())
            self.gates.append(nn.Sigmoid() if last else nn.SiLU())

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self.layers(x) * self.gates(x)


class EmbeddingBlock(nn.Module):
    """Initial features of atoms, from their element, and of bonds."""

    def __init__(self):
        super().__init__()
        self.layer_node_embedding = nn.Embedding(len(ELEMENTS), LATENT_DIM)
        self.layer_edge_embedding = MLP(MAX_N * MAX_L, LATENT_DIM, nn.SiLU())

    def forward(
        self,
        element: Int[torch.Tensor, "n_atoms"],
        basis: Float[torch.Tensor, "n_edges b"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        return self.layer_node_embedding(element), self.layer_edge_embedding(basis)


class ThreeBodyInteractions(nn.Module):
    """Update of the bonds from the angles they form with the other bonds."""

    def __init__(self):
        super().__init__()
        self.update_network_atom = MLP(LATENT_DIM, MAX_N * MAX_L, nn.Sigmoid())
        self.update_network_bond = GatedMLP([MAX_N * MAX_L, LATENT_DIM], use_bias=False)

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        triplet_basis: Float[torch.Tensor, "n_triplets b"],
        end_atom: Int[torch.Tensor, "n_triplets"],
        first_bond: Int[torch.Tensor, "n_triplets"],
    ) -> Float[torch.Tensor, "n_edges d"]:
        messages = triplet_basis * self.update_network_atom(atoms)[end_atom]
        received = scatter(messages, first_bond, bonds.shape[0])

        return bonds + self.update_network_bond(received)


class M3GNetGraphConv(nn.Module):
    """Update of the bonds from their atoms, then of the atoms from their bonds."""

    def __init__(self):
        super().__init__()
        sizes = [3 * LATENT_DIM, LATENT_DIM, LATENT_DIM, LATENT_DIM]
        self.edge_update_func = GatedMLP(sizes)
        self.edge_weight_func = nn.Linear(MAX_N * MAX_L, LATENT_DIM, bias=False)
        self.node_update_func = GatedMLP(sizes)
        self.node_weight_func = nn.Linear(MAX_N * MAX_L, LATENT_DIM, bias=False)

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        basis: Float[torch.Tensor, "n_edges b"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        sender, receiver = edge_index

        bonds = bonds + self.edge_update_func(
            torch.cat([atoms[sender], atoms[receiver], bonds], dim=1)
        ) * self.edge_weight_func(basis)

        messages = self.node_update_func(
            torch.cat([atoms[sender], atoms[receiver], bonds], dim=1)
        ) * self.node_weight_func(basis)

        return atoms + scatter(messages, sender, atoms.shape[0]), bonds


class M3GNetBlock(nn.Module):
    """Block of the network, a graph convolution."""

    def __init__(self):
        super().__init__()
        self.conv = M3GNetGraphConv()

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        basis: Float[torch.Tensor, "n_edges b"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        return self.conv(atoms, bonds, basis, edge_index)


class WeightedReadOut(nn.Module):
    """Energy of each atom from its features."""

    def __init__(self):
        super().__init__()
        self.gated = GatedMLP([LATENT_DIM] * 4 + [1], activate_last=False)

    def forward(
        self, atoms: Float[torch.Tensor, "n_atoms d"]
    ) -> Float[torch.Tensor, "n_atoms"]:
        return self.gated(atoms).squeeze(-1)


class M3GNet(nn.Module):
    """Graph network computing the energy of each atom, before scaling."""

    def __init__(self):
        super().__init__()
        self.bond_expansion = BondExpansion()
        self.embedding = EmbeddingBlock()
        self.basis_expansion = SphericalBesselWithHarmonics()
        self.three_body_interactions = nn.ModuleList(
            ThreeBodyInteractions() for _ in range(NUM_BLOCKS)
        )
        self.graph_layers = nn.ModuleList(M3GNetBlock() for _ in range(NUM_BLOCKS))
        self.final_layer = WeightedReadOut()

    def forward(
        self,
        element: Int[torch.Tensor, "n_atoms"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        distance: Float[torch.Tensor, "n_edges"],
        triplet_index: Int[torch.Tensor, "2 n_triplets"],
        cosine: Float[torch.Tensor, "n_triplets"],
    ) -> Float[torch.Tensor, "n_atoms"]:
        first_bond, second_bond = triplet_index
        basis = self.bond_expansion(distance)

        # an angle is weighted by the smooth cutoffs of its two bonds
        envelope = polynomial_cutoff(distance, THREE_BODY_RADIUS, p=3)
        triplet_basis = self.basis_expansion(distance[second_bond], cosine) * (
            envelope[first_bond] * envelope[second_bond]
        ).unsqueeze(-1)

        atoms, bonds = self.embedding(element, basis)

        for three_body, block in zip(self.three_body_interactions, self.graph_layers):
            bonds = three_body(
                atoms, bonds, triplet_basis, edge_index[1, second_bond], first_bond
            )
            atoms, bonds = block(atoms, bonds, basis, edge_index)

        return self.final_layer(atoms)


class AtomRef(nn.Module):
    """Reference energy of each element of the model."""

    property_offset: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("property_offset", torch.zeros(len(ELEMENTS)))

    def forward(
        self, element: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms"]:
        return self.property_offset.double()[element]


class M3gnetEnergyModel(nn.Module):
    """Total energy of crystal structures predicted by the M3GNet potential.

    Examples:
        >>> model = M3gnetEnergyModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    data_mean: torch.Tensor
    data_std: torch.Tensor
    element_index: torch.Tensor

    def __init__(self):
        super().__init__()
        self.model = M3GNet()
        self.element_refs = AtomRef()
        self.register_buffer("data_mean", torch.tensor(0.0))
        self.register_buffer("data_std", torch.tensor(1.0))

        # index of each atomic number in the embedding, -1 if it has none
        element_index = torch.full((119,), -1, dtype=torch.int64)
        element_index[torch.tensor(ELEMENTS)] = torch.arange(len(ELEMENTS))
        self.register_buffer("element_index", element_index, persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        name: str = "m3gnet-matpes-pbe-2025.2",
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> M3gnetEnergyModel:
        """Load a model with pretrained weights, downloaded on the first call.

        Args:
            name: Name of the weights, see torchms.estimator.m3gnet.weights.
            device: Device of the model.
            cache: Directory of the downloaded weights.
        """
        state_path, args_path = download(name, cache)
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        args = torch.load(args_path, map_location="cpu", weights_only=True)

        model = cls()
        model.load_pretrained(state_dict, args["model"]["init_args"])

        return model.to(device).eval().requires_grad_(False)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        model_args: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of an M3GNet potential of matgl.

        Args:
            state_dict: Parameters of the potential of matgl.
            model_args: Arguments of the model of matgl, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was built with other arguments or elements.
        """
        if model_args is not None:
            different = {
                key: model_args.get(key)
                for key, value in MODEL_ARGS.items()
                if model_args.get(key, value) != value
            }
            elements = model_args.get("element_types")
            if (
                isinstance(elements, (tuple, list))
                and tuple(symbol_to_number(symbol) for symbol in elements) != ELEMENTS
            ):
                different["element_types"] = elements
            if different:
                raise InvalidWeights(
                    f"weights of a model with other arguments: {different}"
                )

        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e

    def graphs(self, structures: Structures) -> tuple[
        Graph,
        Float[torch.Tensor, "n_edges"],
        Int[torch.Tensor, "2 n_triplets"],
        Float[torch.Tensor, "n_triplets"],
    ]:
        """Bonds and angles of the model.

        Atoms closer than 5 Å are bonded, periodic images of the atom itself
        included. Angles are the pairs of bonds of at most 4 Å starting from
        the same atom.

        Returns:
            Graph of the atoms, length of each bond in Å, shape (n_edges,),
            indices of the two bonds of each angle, shape (2, n_triplets), and
            cosine of each angle, shape (n_triplets,).
        """
        graph = cutoff(structures, RADIUS, self_images=True)
        vectors = graph.edges_vectors()
        distance = vectors.norm(dim=-1)

        triplet_index, _, _ = triplets(graph, distance <= THREE_BODY_RADIUS)
        cosine = triplet_cosines(vectors, triplet_index).clamp(-1 + 1e-7, 1 - 1e-7)

        return graph, distance, triplet_index, cosine

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        element = self.element_index[structures.z]
        if (element < 0).any():
            unknown = structures.z[element < 0].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

        graph, distance, triplet_index, cosine = self.graphs(structures)
        n_struct = len(structures.num_atoms)

        per_atom = self.model(
            element, graph.edge_index, distance, triplet_index, cosine
        )
        energy = self.data_std * scatter(per_atom, structures.batch, n_struct)
        energy = energy + self.data_mean

        return energy.double() + scatter(
            self.element_refs(element), structures.batch, n_struct
        )
