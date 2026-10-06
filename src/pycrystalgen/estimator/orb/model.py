# Derived from orb-models (https://github.com/orbital-materials/orb-models),
# Copyright 2024 Orbital Materials, Inc., licensed under the Apache License,
# Version 2.0 (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the energy prediction of the orb-v3
# conservative model, for inference only. Forces, stress, charges, dispersion,
# training, datasets and the conversion from ASE are removed. Graphs are built
# by pycrystalgen.graph instead of nvalchemi-toolkit-ops, and batches are
# pycrystalgen Structures. Names of modules and parameters follow orb-models to
# load its pretrained weights.
#
# Spherical harmonics come from e3nn (https://github.com/e3nn/e3nn, MIT
# License), through orb-models.
"""Energy prediction of the pretrained orb-v3 conservative model.

The network is a graph neural network on the atoms closer than 6 Å: edge
features are the product of a radial basis and spherical harmonics, updated by
five attention layers. The energy is the sum of a head on the mean of the atom
features, a pair repulsion at short distance and a reference energy per atom.

Weights are the ones published by Orbital Materials, see weights.py.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn

from pycrystalgen.data.batching import Structures
from pycrystalgen.estimator.orb.weights import download
from pycrystalgen.graph.cutoff import cutoff, nearest_neighbors
from pycrystalgen.graph.graph import Graph

RADIUS = 6.0
MAX_NEIGHBORS = 120
LATENT_DIM = 256
NUM_ELEMENTS = 118

# parameters of the pretrained weights which are not used to predict the energy
UNUSED_PARAMETERS = (
    "model._decoder.",
    "heads.confidence.",
    "grad_forces_normalizer.",
    "grad_stress_normalizer.",
)

# Covalent radii in Å indexed by atomic number (Cordero et al., Dalton Trans.
# 2008, as distributed by ASE). Index 0 and elements above Cm are placeholders.
COVALENT_RADII = (
    (0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66)
    + (1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53)
    + (1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20)
    + (1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45)
    + (1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03)
    + (2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87)
    + (1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48)
    + (1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80)
    + (1.69,)
    + (2.00,) * 22
)


class InvalidWeights(Exception):
    """Raised when pretrained weights do not match the model."""


def polynomial_cutoff(
    distance: Float[torch.Tensor, "n"], radius: Float[torch.Tensor, "n"] | float, p: int
) -> Float[torch.Tensor, "n"]:
    """Smooth envelope going from 1 at a distance of 0 to 0 at the radius.

    Its first and second derivatives are zero at the radius. It is zero after.

    Example:
        >>> distance = torch.tensor([0.0, 3.0, 6.0, 7.0])
        >>> polynomial_cutoff(distance, 6.0, p=4)
        tensor([1.0000, 0.6562, 0.0000, 0.0000])
    """
    ratio = distance / radius
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(ratio, p)
        + p * (p + 2.0) * torch.pow(ratio, p + 1)
        - (p * (p + 1.0) / 2.0) * torch.pow(ratio, p + 2)
    )
    return torch.where(distance < radius, envelope, torch.zeros_like(envelope))


def spherical_harmonics(
    vectors: Float[torch.Tensor, "n 3"],
) -> Float[torch.Tensor, "n 16"]:
    """Real spherical harmonics up to l = 3 of the direction of vectors.

    Convention of e3nn with the "component" normalization: the 2l+1 harmonics
    of a degree l have a squared norm of 2l+1 for any direction. A vector of
    length zero gives the harmonics of the zero vector.

    Example:
        >>> harmonics = spherical_harmonics(torch.tensor([[0.0, 2.0, 0.0]]))
        >>> harmonics[0, :4]
        tensor([1.0000, 0.0000, 1.7321, 0.0000])
        >>> harmonics[0, 1:4].square().sum(), harmonics[0, 4:9].square().sum()
        (tensor(3.), tensor(5.))
    """
    x, y, z = F.normalize(vectors, dim=-1).unbind(dim=-1)

    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    degree_0 = torch.ones_like(x).unsqueeze(-1)
    degree_1 = torch.stack([x, y, z], dim=-1)
    degree_2 = torch.stack(
        [
            sh_2_0,
            math.sqrt(3.0) * x * y,
            y2 - 0.5 * x2z2,
            math.sqrt(3.0) * y * z,
            sh_2_4,
        ],
        dim=-1,
    )
    degree_3 = torch.stack(
        [
            math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x),
            math.sqrt(5.0) * sh_2_0 * y,
            math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x,
            0.5 * y * (2.0 * y2 - 3.0 * x2z2),
            math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2),
            math.sqrt(5.0) * sh_2_4 * y,
            math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x),
        ],
        dim=-1,
    )

    return torch.cat(
        [
            degree_0,
            math.sqrt(3.0) * degree_1,
            math.sqrt(5.0) * degree_2,
            math.sqrt(7.0) * degree_3,
        ],
        dim=-1,
    )


def segment_sum(
    values: Float[torch.Tensor, "n ..."], index: Int[torch.Tensor, "n"], size: int
) -> Float[torch.Tensor, "size ..."]:
    """Sum values with the same index."""
    return values.new_zeros(size, *values.shape[1:]).index_add_(0, index, values)


def build_mlp(
    input_size: int, hidden_sizes: list[int], output_size: int
) -> nn.Sequential:
    """Fully connected network with SiLU activations between its layers."""
    sizes = [input_size, *hidden_sizes, output_size]
    mlp = nn.Sequential()

    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        mlp.add_module(f"NN-{i}", nn.Linear(sizes[i], sizes[i + 1]))
        mlp.add_module(f"Act-{i}", nn.Identity() if last else nn.SiLU())

    return mlp


def mlp_and_norm(input_size: int, hidden_size: int = 1024) -> nn.Sequential:
    """Network with two hidden layers followed by a normalization."""
    return nn.Sequential(
        OrderedDict(
            mlp=build_mlp(input_size, [hidden_size, hidden_size], LATENT_DIM),
            layer_norm=nn.RMSNorm(LATENT_DIM),
        )
    )


class BesselBasis(nn.Module):
    """Radial basis sqrt(2 / r_max) sin(n pi r / r_max) / r, for n = 1..num_bases."""

    bessel_weights: torch.Tensor
    r_max: torch.Tensor
    prefactor: torch.Tensor

    def __init__(self, r_max: float = RADIUS, num_bases: int = 8):
        super().__init__()
        self.register_buffer(
            "bessel_weights", math.pi / r_max * torch.arange(1.0, num_bases + 1.0)
        )
        self.register_buffer("r_max", torch.tensor(r_max))
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / r_max)))

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_bases"]:
        distance = distance.unsqueeze(-1)
        return self.prefactor * torch.sin(self.bessel_weights * distance) / distance


class AtomEmbedding(nn.Module):
    """Learned features of each element, indexed by atomic number."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(NUM_ELEMENTS + 1, LATENT_DIM)

    def forward(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms d"]:
        return self.embeddings(z)


class Encoder(nn.Module):
    """Initial features of atoms and edges."""

    def __init__(self, num_edge_features: int):
        super().__init__()
        self._node_fn = mlp_and_norm(LATENT_DIM)
        self._edge_fn = mlp_and_norm(num_edge_features)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_atoms d"],
        edges: Float[torch.Tensor, "n_edges f"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        return self._node_fn(nodes), self._edge_fn(edges)


class AttentionInteractionNetwork(nn.Module):
    """Message passing layer updating the features of atoms and edges.

    Messages are weighted by a sigmoid attention times the smooth cutoff of the
    edge. An atom is updated from the messages it sends and the ones it
    receives.
    """

    def __init__(self):
        super().__init__()
        self._node_mlp = mlp_and_norm(3 * LATENT_DIM)
        self._edge_mlp = mlp_and_norm(3 * LATENT_DIM)
        self._receive_attn = nn.Linear(LATENT_DIM, 1)
        self._send_attn = nn.Linear(LATENT_DIM, 1)

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_atoms d"],
        edges: Float[torch.Tensor, "n_edges d"],
        senders: Int[torch.Tensor, "n_edges"],
        receivers: Int[torch.Tensor, "n_edges"],
        envelope: Float[torch.Tensor, "n_edges 1"],
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        receive_attn = torch.sigmoid(self._receive_attn(edges)) * envelope
        send_attn = torch.sigmoid(self._send_attn(edges)) * envelope

        updated_edges = self._edge_mlp(
            torch.cat([edges, nodes[senders], nodes[receivers]], dim=1)
        )

        sent = segment_sum(updated_edges * send_attn, senders, nodes.shape[0])
        received = segment_sum(updated_edges * receive_attn, receivers, nodes.shape[0])

        updated_nodes = self._node_mlp(torch.cat([nodes, received, sent], dim=1))

        return nodes + updated_nodes, edges + updated_edges


class MoleculeGNS(nn.Module):
    """Graph network computing the features of each atom."""

    def __init__(self, num_message_passing_steps: int = 5):
        super().__init__()
        self.rbf_transform = BesselBasis()
        self.atom_emb = AtomEmbedding()
        self._encoder = Encoder(num_edge_features=8 * 16)
        self.gnn_stacks = nn.ModuleList(
            AttentionInteractionNetwork() for _ in range(num_message_passing_steps)
        )

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        vectors: Float[torch.Tensor, "n_edges 3"],
        senders: Int[torch.Tensor, "n_edges"],
        receivers: Int[torch.Tensor, "n_edges"],
    ) -> Float[torch.Tensor, "n_atoms d"]:
        distance = vectors.norm(dim=1)
        envelope = polynomial_cutoff(distance, RADIUS, p=4).unsqueeze(-1)

        # outer product of the radial and the angular bases
        radial = self.rbf_transform(distance)
        angular = spherical_harmonics(vectors)
        edges = envelope * (radial[:, :, None] * angular[:, None, :]).flatten(1)

        nodes, edges = self._encoder(self.atom_emb(z), edges)

        for layer in self.gnn_stacks:
            nodes, edges = layer(nodes, edges, senders, receivers, envelope)

        return nodes


class ScalarNormalizer(nn.Module):
    """Mean and standard deviation of the energy per atom seen in training."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1, affine=False, momentum=None)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        running_mean, running_var = self.bn.running_mean, self.bn.running_var
        if running_mean is None or running_var is None:
            raise InvalidWeights("the normalizer of the energy has no statistics")

        return values * torch.sqrt(running_var) + running_mean


class LinearReferenceEnergy(nn.Module):
    """Energy of a structure as a sum of an energy per atom of each element."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_ELEMENTS, 1, bias=False)

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        batch: Int[torch.Tensor, "n_atoms"],
        size: int,
    ) -> Float[torch.Tensor, "size"]:
        return segment_sum(self.linear.weight[0].double()[z], batch, size)


class EnergyHead(nn.Module):
    """Energy of a structure from the mean of the features of its atoms."""

    def __init__(self):
        super().__init__()
        self.normalizer = ScalarNormalizer()
        self.mlp = build_mlp(LATENT_DIM, [LATENT_DIM], 1)
        self.reference = LinearReferenceEnergy()

    def forward(
        self,
        nodes: Float[torch.Tensor, "n_atoms d"],
        batch: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct"]:
        mean = segment_sum(nodes, batch, len(num_atoms)) / num_atoms.unsqueeze(-1)
        per_atom = self.normalizer.inverse(self.mlp(mean).squeeze(-1))

        return per_atom * num_atoms


class ZBLBasis(nn.Module):
    """Ziegler-Biersack-Littmark repulsion between close atoms.

    The repulsion of a pair is cut at the sum of the covalent radii of its
    atoms. As in the training of the pretrained weights, the energy of a
    structure is the mean over its atoms and atomic numbers are shifted by one.
    """

    c: torch.Tensor
    d: torch.Tensor
    covalent_radii: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "c", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]), persistent=False
        )
        self.register_buffer(
            "d", torch.tensor([3.2, 0.9423, 0.4028, 0.2016]), persistent=False
        )
        self.register_buffer(
            "covalent_radii", torch.tensor(COVALENT_RADII), persistent=False
        )

    def forward(
        self,
        z: Int[torch.Tensor, "n_atoms"],
        vectors: Float[torch.Tensor, "n_edges 3"],
        senders: Int[torch.Tensor, "n_edges"],
        receivers: Int[torch.Tensor, "n_edges"],
        batch: Int[torch.Tensor, "n_atoms"],
        num_atoms: Int[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct"]:
        # orb-models reads the atomic number from a one-hot encoding indexed by
        # the atomic number and adds one: the repulsion of an element is the
        # one of the next element. The pretrained weights were trained with it.
        z_send, z_receive = z[senders] + 1, z[receivers] + 1
        distance = vectors.norm(dim=1)

        # screening length of the pair
        length = 0.4543 * 0.529 / (torch.pow(z_send, 0.3) + torch.pow(z_receive, 0.3))
        screening = (
            self.c * torch.exp(-self.d * (distance / length).unsqueeze(-1))
        ).sum(dim=-1)

        radius = self.covalent_radii[z_send] + self.covalent_radii[z_receive]
        pair = (
            0.5
            * (14.3996 * z_send * z_receive / distance)
            * screening
            * polynomial_cutoff(distance, radius, p=6)
        )

        per_atom = segment_sum(pair, senders, len(z))
        return segment_sum(per_atom, batch, len(num_atoms)) / num_atoms


class OrbEnergyModel(nn.Module):
    """Total energy of crystal structures predicted by orb-v3 conservative.

    Example:
        >>> model = OrbEnergyModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    def __init__(self):
        super().__init__()
        self.model = MoleculeGNS()
        self.heads = nn.ModuleDict({"energy": EnergyHead()})
        self.pair_repulsion_fn = ZBLBasis()

    @classmethod
    def from_pretrained(
        cls,
        name: str = "orb-v3-conservative-inf-mpa",
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> OrbEnergyModel:
        """Load a model with pretrained weights, downloaded on the first call.

        Args:
            name: Name of the weights, see pycrystalgen.estimator.orb.weights.
            device: Device of the model.
            cache: Directory of the downloaded weights.
        """
        state_dict = torch.load(
            download(name, cache), map_location="cpu", weights_only=True
        )

        model = cls()
        model.load_pretrained(state_dict)

        return model.to(device).eval().requires_grad_(False)

    def load_pretrained(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the weights of an orb-v3 conservative model of orb-models.

        Raises:
            InvalidWeights: If parameters are missing or unknown.
        """
        used = {
            name: values
            for name, values in state_dict.items()
            if not name.startswith(UNUSED_PARAMETERS)
        }

        reference = "heads.energy.reference.linear.weight"
        if reference in used:
            used[reference] = used[reference].reshape(1, NUM_ELEMENTS)

        try:
            self.load_state_dict(used, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e

    def edges(
        self, structures: Structures
    ) -> tuple[Graph, Float[torch.Tensor, "n_edges 3"]]:
        """Graph of the model and vector of each edge, in Å.

        Atoms are connected to their 120 nearest neighbours closer than 6 Å,
        periodic images of the atom itself included. The vector of an edge goes
        from its sender to its receiver.
        """
        graph = nearest_neighbors(
            cutoff(structures, RADIUS, self_images=True), MAX_NEIGHBORS
        )

        senders, receivers = graph.edge_index
        fractional = structures.x[receivers] + graph.edge_shift - structures.x[senders]
        vectors = torch.bmm(
            fractional.unsqueeze(1), structures.lattice[graph.edge_batch]
        ).squeeze(1)

        return graph, vectors

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision."""
        graph, vectors = self.edges(structures)
        senders, receivers = graph.edge_index
        z, batch, num_atoms = structures.z, structures.batch, structures.num_atoms

        nodes = self.model(z, vectors, senders, receivers)

        head = self.heads["energy"]
        if not isinstance(head, EnergyHead):
            raise TypeError("the energy head of the model has been replaced")

        energy = head(nodes, batch, num_atoms)
        energy = energy + self.pair_repulsion_fn(
            z, vectors, senders, receivers, batch, num_atoms
        )

        return energy.double() + head.reference(z, batch, len(num_atoms))
