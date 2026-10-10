# Derived from matgl (https://github.com/materialsvirtuallab/matgl), Copyright
# (c) 2022, Materials Virtual Lab, BSD 3-Clause license (see LICENSE in this
# directory).
#
# Changes: this file is a rewrite of the energy prediction of the M3GNet
# potential of matgl, for inference only. Forces, stress, state features, the
# other read-outs, training and the conversion from pymatgen are removed.
# Graphs are built by torchms.graph, the spherical Bessel functions and
# their roots are computed with torch instead of sympy, and batches
# are torchms Structures. Names of modules and parameters follow matgl to
# load its pretrained weights.
"""Energy prediction of the pretrained M3GNet potential.

The network has features on atoms and on bonds, the pairs of atoms closer than
5 Å. In each of its three blocks, a bond is first updated from the angles it
forms with the other bonds of its atom shorter than 4 Å, in a basis of
spherical Bessel functions of the length of the other bond and of spherical
harmonics of the angle. Bonds and atoms are then updated from each other. The
energy is a sum of an energy per atom and of a reference energy per element.

Weights are the ones published with matgl, see config.yaml.
"""

from __future__ import annotations

import os
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.data.constants import symbol_to_number
from torchms.models.estimator.config import load_config
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
    element_index,
    element_table,
)
from torchms.models.estimator.weights import download
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddRadiusGraph,
    AddTriplets,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.layers.basis import (
    SphericalBesselBasis,
    legendre,
    polynomial_cutoff,
)

# every value of the model is in config.yaml, next to this module
CONFIG = load_config(__file__)

RADIUS: float = CONFIG["graph"]["radius"]
THREE_BODY_RADIUS: float = CONFIG["graph"]["three_body_radius"]

LATENT_DIM: int = CONFIG["model"]["latent_dim"]
MAX_N: int = CONFIG["model"]["max_n"]
MAX_L: int = CONFIG["model"]["max_l"]
NUM_BLOCKS: int = CONFIG["model"]["num_blocks"]
CUTOFF_POWER: int = CONFIG["model"]["cutoff_power"]
COSINE_MARGIN: float = CONFIG["model"]["cosine_margin"]

# atomic numbers of the elements of the model, in the order of its embedding
ELEMENTS: tuple[int, ...] = tuple(
    z for first, last in CONFIG["model"]["elements"] for z in range(first, last + 1)
)

# arguments of the model the rewrite is written for, as stored with the weights
MODEL_ARGS: dict[str, object] = CONFIG["upstream"]["model_args"]


class BondExpansion(nn.Module):
    """Radial basis of the length of a bond."""

    def __init__(self):
        super().__init__()
        self.rbf = SphericalBesselBasis(MAX_L, MAX_N, RADIUS)

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        return self.rbf(distance)


class SphericalBesselWithHarmonics(nn.Module):
    """Basis of an angle: Bessel functions of a length times harmonics of the angle."""

    def __init__(self):
        super().__init__()
        self.sbf = SphericalBesselBasis(MAX_L, MAX_N, RADIUS)

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
        graph: Graph,
        angles: EdgeTuples,
    ) -> Float[torch.Tensor, "n_edges d"]:
        # an angle i→j, i→k sends to its first bond the features of the atom k
        end_atoms = graph.unpool_receivers(self.update_network_atom(atoms))
        messages = triplet_basis * angles.unpool_edges(end_atoms, position=1)
        received = angles.aggregate_to_edges(messages, position=0)

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
        graph: Graph,
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        senders, receivers = graph.unpool_senders(atoms), graph.unpool_receivers(atoms)

        bonds = bonds + self.edge_update_func(
            torch.cat([senders, receivers, bonds], dim=1)
        ) * self.edge_weight_func(basis)

        messages = self.node_update_func(
            torch.cat([senders, receivers, bonds], dim=1)
        ) * self.node_weight_func(basis)

        return atoms + graph.aggregate_edges_to_nodes(messages, to="sender"), bonds


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
        graph: Graph,
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        return self.conv(atoms, bonds, basis, graph)


class WeightedReadOut(nn.Module):
    """Energy of each atom from its features."""

    def __init__(self):
        super().__init__()
        self.gated = GatedMLP([LATENT_DIM] * 4 + [1], activate_last=False)

    def forward(
        self, atoms: Float[torch.Tensor, "n_atoms d"]
    ) -> Float[torch.Tensor, "n_atoms"]:
        return self.gated(atoms).squeeze(-1)


class M3GNetNetwork(nn.Module):
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
        graph: Graph,
        distance: Float[torch.Tensor, "n_edges"],
        angles: EdgeTuples,
        cosine: Float[torch.Tensor, "n_triplets"],
    ) -> Float[torch.Tensor, "n_atoms"]:
        basis = self.bond_expansion(distance)

        # an angle is weighted by the smooth cutoffs of its two bonds
        envelope = polynomial_cutoff(distance, THREE_BODY_RADIUS, p=CUTOFF_POWER)
        triplet_basis = self.basis_expansion(
            angles.unpool_edges(distance, position=1), cosine
        ) * (
            angles.unpool_edges(envelope, position=0)
            * angles.unpool_edges(envelope, position=1)
        ).unsqueeze(
            -1
        )

        atoms, bonds = self.embedding(element, basis)

        for three_body, block in zip(self.three_body_interactions, self.graph_layers):
            bonds = three_body(atoms, bonds, triplet_basis, graph, angles)
            atoms, bonds = block(atoms, bonds, basis, graph)

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


class M3GNet(PretrainedModel):
    """Total energy of crystal structures predicted by the M3GNet potential.

    Examples:
        >>> model = M3GNet()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    DEFAULT_WEIGHTS = CONFIG["default_weights"]
    GRAPH_FIELDS = ("graph", "triplets")

    data_mean: torch.Tensor
    data_std: torch.Tensor
    element_table: torch.Tensor

    def __init__(self):
        super().__init__()
        self.model = M3GNetNetwork()
        self.element_refs = AtomRef()
        self.register_buffer("data_mean", torch.tensor(0.0))
        self.register_buffer("data_std", torch.tensor(1.0))

        # index of each atomic number in the embedding, -1 if it has none
        self.register_buffer(
            "element_table", element_table(torch.tensor(ELEMENTS)), persistent=False
        )

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them."""
        state_path, args_path = download("m3gnet", name, cache)
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        args = torch.load(args_path, map_location="cpu", weights_only=True)

        self.load_pretrained(state_dict, args["model"]["init_args"])

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
            different = different_arguments(model_args, MODEL_ARGS)
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

        self.load_parameters(state_dict)

    @classmethod
    def transform(cls) -> Transform:
        """Transform attaching the bonds and the angles of the model.

        Atoms closer than 5 Å are bonded, periodic images of the atom itself
        included. Angles are the pairs of bonds of at most 4 Å starting from
        the same atom.
        """
        return Compose(
            [
                AddRadiusGraph(RADIUS, self_images=True),
                AddTriplets(radius=THREE_BODY_RADIUS),
            ]
        )

    def graphs(self, structures: Structures) -> tuple[
        Graph,
        Float[torch.Tensor, "n_edges"],
        EdgeTuples,
        Float[torch.Tensor, "n_triplets"],
    ]:
        """Bonds and angles of the structures, with their geometry.

        They are the graph and the triplets attached to the structures by
        transform.

        Returns:
            Graph of the atoms, length of each bond in Å, shape (n_edges,),
            triplets of the graph, the two bonds of each angle, and cosine of
            each angle, shape (n_triplets,).
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        angles = attached_tuples(structures, "triplets", self)

        vectors = edges_vectors(graph)
        distance = vectors.norm(dim=-1)
        cosine = triplet_cosines(vectors, angles).clamp(
            -1 + COSINE_MARGIN, 1 - COSINE_MARGIN
        )

        return graph, distance, angles, cosine

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        element = element_index(structures.z, self.element_table)

        graph, distance, angles, cosine = self.graphs(structures)

        per_atom = self.model(element, graph, distance, angles, cosine)
        energy = self.data_std * graph.aggregate_nodes_to_structures(per_atom)
        energy = energy + self.data_mean

        return energy.double() + graph.aggregate_nodes_to_structures(
            self.element_refs(element)
        )
