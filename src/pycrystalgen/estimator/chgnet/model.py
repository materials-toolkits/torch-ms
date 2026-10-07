# Derived from chgnet (https://github.com/CederGroupHub/chgnet), Copyright (c)
# 2023, The Regents of the University of California, through Lawrence Berkeley
# National Laboratory, modified BSD license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the energy prediction of CHGNet 0.3.0, for
# inference only. Magnetic moments, forces, stress, training, datasets and the
# conversion from pymatgen are removed. Graphs are built by pycrystalgen.graph
# instead of the graph converter of chgnet, and batches are pycrystalgen
# Structures. Names of modules and parameters follow chgnet to load its
# pretrained weights.
"""Energy prediction of the pretrained CHGNet model.

The network has features on atoms, on bonds and on angles. Atoms closer than
6 Å are bonded; the angles are the pairs of bonds shorter than 3 Å around an
atom. A bond and its reverse share their features. Each of the first three
layers updates the atoms from their bonds, the bonds from their angles, then
the angles; a last layer updates the atoms. The energy is a sum of an energy
per atom and of a reference energy per element.

Weights are the ones published with chgnet, see weights.py.
"""

# attributes are named as in chgnet, to load its weights
# pylint: disable=invalid-name

from __future__ import annotations

import math
import os
from typing import Mapping

import torch
from jaxtyping import Float, Int
from torch import nn

from pycrystalgen.data.batching import Structures
from pycrystalgen.estimator.chgnet.weights import download
from pycrystalgen.estimator.orb.model import polynomial_cutoff
from pycrystalgen.graph.cutoff import cutoff
from pycrystalgen.graph.geometry import triplet_cosines
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.line_graph import reverse_edges, triplets
from pycrystalgen.graph.sparse import scatter

RADIUS = 6.0
BOND_RADIUS = 3.0
LATENT_DIM = 64
NUM_ELEMENTS = 94
NUM_RADIAL = 31
NUM_ANGULAR = 31
CUTOFF_POWER = 8

# parameters of the pretrained weights which are not used to predict the energy
UNUSED_PARAMETERS = ("site_wise.",)

# arguments of the model the rewrite is written for, as stored with the weights
MODEL_ARGS = {
    "atom_fea_dim": LATENT_DIM,
    "bond_fea_dim": LATENT_DIM,
    "angle_fea_dim": LATENT_DIM,
    "composition_model": "MPtrj",
    "num_radial": NUM_RADIAL,
    "num_angular": NUM_ANGULAR,
    "n_conv": 4,
    "atom_conv_hidden_dim": 64,
    "update_bond": True,
    "bond_conv_hidden_dim": 64,
    "update_angle": True,
    "angle_layer_hidden_dim": 0,
    "read_out": "ave",
    "mlp_hidden_dims": [64, 64, 64],
    "mlp_first": True,
    "is_intensive": True,
    "non_linearity": "silu",
    "atom_graph_cutoff": 6,
    "bond_graph_cutoff": 3,
    "cutoff_coeff": CUTOFF_POWER,
    "learnable_rbf": True,
    "gMLP_norm": "layer",
    "readout_norm": "layer",
}


class InvalidWeights(Exception):
    """Raised when weights do not match the parameters of the model."""


class MLP(nn.Module):
    """Fully connected network with SiLU activations between its layers.

    The layer before the last one is the place of the dropout of chgnet: the
    indices of the layers are the ones of its weights.
    """

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int, bias=True
    ):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        layers: list[nn.Module] = []

        for size_in, size_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(size_in, size_out, bias=bias), nn.SiLU()]

        layers += [nn.Identity(), nn.Linear(sizes[-1], output_size, bias=bias)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        return self.layers(x)


class GatedMLP(nn.Module):
    """Product of a network and of a gate in [0, 1], both normalized."""

    def __init__(self, input_size: int, hidden_sizes: list[int]):
        super().__init__()
        self.mlp_core = MLP(input_size, hidden_sizes, LATENT_DIM)
        self.mlp_gate = MLP(input_size, hidden_sizes, LATENT_DIM)
        self.bn1 = nn.LayerNorm(LATENT_DIM)
        self.bn2 = nn.LayerNorm(LATENT_DIM)

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        core = nn.functional.silu(self.bn1(self.mlp_core(x)))
        gate = torch.sigmoid(self.bn2(self.mlp_gate(x)))

        return core * gate


class RadialBessel(nn.Module):
    """Radial basis sqrt(2 / r_max) sin(f r / r_max) / r with a smooth cutoff.

    The frequencies f are parameters, n pi for n = 1..num_radial before training.
    """

    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.frequencies = nn.Parameter(
            math.pi * torch.arange(1.0, NUM_RADIAL + 1.0), requires_grad=False
        )

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_radial"]:
        distance = distance.unsqueeze(-1)
        basis = (
            math.sqrt(2.0 / self.radius)
            * torch.sin(self.frequencies * distance / self.radius)
            / distance
        )

        return polynomial_cutoff(distance, self.radius, p=CUTOFF_POWER) * basis


class BondEncoder(nn.Module):
    """Radial bases of a bond for the cutoffs of the bonds and of the angles."""

    def __init__(self):
        super().__init__()
        self.rbf_expansion_ag = RadialBessel(RADIUS)
        self.rbf_expansion_bg = RadialBessel(BOND_RADIUS)


class Fourier(nn.Module):
    """Fourier basis of an angle: a constant, then sines and cosines of f angle.

    The frequencies f are parameters, 1..order before training.
    """

    def __init__(self):
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.arange(1.0, (NUM_ANGULAR - 1) // 2 + 1.0), requires_grad=False
        )

    def forward(
        self, angle: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_angular"]:
        phase = angle.unsqueeze(-1) * self.frequencies
        constant = torch.full_like(angle, math.sqrt(0.5)).unsqueeze(-1)

        return torch.cat(
            [constant, torch.sin(phase), torch.cos(phase)], dim=-1
        ) / math.sqrt(math.pi)


class AngleEncoder(nn.Module):
    """Fourier basis of the angle between two bonds."""

    def __init__(self):
        super().__init__()
        self.fourier_expansion = Fourier()

    def forward(
        self, cosine: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_angular"]:
        # as chgnet, the cosine is shrunk to keep the angle differentiable
        return self.fourier_expansion(torch.acos(cosine * (1.0 - 1e-6)))


class AtomEmbedding(nn.Module):
    """Learned features of each element, indexed by atomic number minus one."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_ELEMENTS, LATENT_DIM)

    def forward(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms d"]:
        return self.embedding(z - 1)


class AtomConv(nn.Module):
    """Update of the atoms from their bonds and their neighbours."""

    def __init__(self):
        super().__init__()
        self.twoBody_atom = GatedMLP(3 * LATENT_DIM, [LATENT_DIM])
        self.mlp_out = MLP(LATENT_DIM, [], LATENT_DIM, bias=False)

    def forward(
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        weights: Float[torch.Tensor, "n_bonds d"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        edge_bond: Int[torch.Tensor, "n_edges"],
    ) -> Float[torch.Tensor, "n_atoms d"]:
        center, neighbor = edge_index
        messages = self.twoBody_atom(
            torch.cat([atoms[center], bonds[edge_bond], atoms[neighbor]], dim=1)
        )
        received = scatter(messages * weights[edge_bond], center, atoms.shape[0])

        return atoms + self.mlp_out(received)


class BondConv(nn.Module):
    """Update of the bonds from the angles they form with the other bonds."""

    def __init__(self):
        super().__init__()
        self.twoBody_bond = GatedMLP(4 * LATENT_DIM, [LATENT_DIM])
        self.mlp_out = MLP(LATENT_DIM, [], LATENT_DIM, bias=False)

    def forward(
        self,
        triplet_features: Float[torch.Tensor, "n_triplets 4*d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        weights: Float[torch.Tensor, "n_bonds d"],
        triplet_bond: Int[torch.Tensor, "2 n_triplets"],
    ) -> Float[torch.Tensor, "n_bonds d"]:
        messages = (
            self.twoBody_bond(triplet_features)
            * weights[triplet_bond[0]]
            * weights[triplet_bond[1]]
        )
        received = scatter(messages, triplet_bond[0], bonds.shape[0])

        return bonds + self.mlp_out(received)


class AngleUpdate(nn.Module):
    """Update of the angles from their two bonds and their atom."""

    def __init__(self):
        super().__init__()
        self.twoBody_bond = GatedMLP(4 * LATENT_DIM, [])

    def forward(
        self,
        triplet_features: Float[torch.Tensor, "n_triplets 4*d"],
        angles: Float[torch.Tensor, "n_triplets d"],
    ) -> Float[torch.Tensor, "n_triplets d"]:
        return angles + self.twoBody_bond(triplet_features)


class AtomRef(nn.Module):
    """Reference energy of each element, indexed by atomic number minus one."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NUM_ELEMENTS, 1, bias=False)

    def forward(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms"]:
        return self.fc.weight[0].double()[z - 1]


# one module per module of chgnet
# pylint: disable-next=too-many-instance-attributes
class ChgnetEnergyModel(nn.Module):
    """Total energy of crystal structures predicted by CHGNet 0.3.0.

    Examples:
        >>> model = ChgnetEnergyModel()
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float64)
    """

    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.composition_model = AtomRef()
        self.atom_embedding = AtomEmbedding()
        self.bond_basis_expansion = BondEncoder()
        self.bond_embedding = nn.Linear(NUM_RADIAL, LATENT_DIM, bias=False)
        self.bond_weights_ag = nn.Linear(NUM_RADIAL, LATENT_DIM, bias=False)
        self.bond_weights_bg = nn.Linear(NUM_RADIAL, LATENT_DIM, bias=False)
        self.angle_basis_expansion = AngleEncoder()
        self.angle_embedding = nn.Linear(NUM_ANGULAR, LATENT_DIM, bias=False)

        self.atom_conv_layers = nn.ModuleList(AtomConv() for _ in range(num_layers))
        self.bond_conv_layers = nn.ModuleList(BondConv() for _ in range(num_layers - 1))
        self.angle_layers = nn.ModuleList(AngleUpdate() for _ in range(num_layers - 1))

        self.readout_norm = nn.LayerNorm(LATENT_DIM)
        self.mlp = MLP(LATENT_DIM, [LATENT_DIM] * 3, 1)

    @classmethod
    def from_pretrained(
        cls,
        name: str = "chgnet-0.3.0",
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> ChgnetEnergyModel:
        """Load a model with pretrained weights, downloaded on the first call.

        Args:
            name: Name of the weights, see pycrystalgen.estimator.chgnet.weights.
            device: Device of the model.
            cache: Directory of the downloaded weights.
        """
        checkpoint = torch.load(
            download(name, cache), map_location="cpu", weights_only=True
        )

        model = cls()
        model.load_pretrained(
            checkpoint["model"]["state_dict"], checkpoint["model"]["model_args"]
        )

        return model.to(device).eval().requires_grad_(False)

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        model_args: Mapping[str, object] | None = None,
    ) -> None:
        """Load the weights of a CHGNet model of chgnet.

        Args:
            state_dict: Parameters of the model of chgnet.
            model_args: Arguments of the model of chgnet, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model was built with other arguments than CHGNet 0.3.0.
        """
        if model_args is not None:
            different = {
                key: model_args.get(key)
                for key, value in MODEL_ARGS.items()
                if model_args.get(key, value) != value
            }
            if different:
                raise InvalidWeights(
                    f"weights of a model with other arguments: {different}"
                )

        used = {
            name: values
            for name, values in state_dict.items()
            if not name.startswith(UNUSED_PARAMETERS)
        }

        try:
            self.load_state_dict(used, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e

    def graphs(self, structures: Structures) -> tuple[
        Graph,
        Int[torch.Tensor, "n_edges"],
        Float[torch.Tensor, "n_bonds"],
        Int[torch.Tensor, "2 n_triplets"],
        Float[torch.Tensor, "n_triplets"],
    ]:
        """Bonds and angles of the model.

        Atoms closer than 6 Å are connected, periodic images of the atom itself
        included. An edge and its reverse are the same bond. Angles are the
        pairs of edges shorter than 3 Å starting from the same atom.

        Returns:
            Graph of the atoms, index of the bond of each edge, shape
            (n_edges,), length of each bond in Å, shape (n_bonds,), indices of
            the two edges of each angle, shape (2, n_triplets), and cosine of
            each angle, shape (n_triplets,).
        """
        device = structures.device()
        graph = cutoff(structures, RADIUS, self_images=True)
        distance = graph.edges_distances()

        # an edge and its reverse share the smallest of their two indices
        edge = torch.arange(graph.edge_index.shape[1], device=device)
        first, edge_bond = torch.unique(
            torch.minimum(edge, reverse_edges(graph)), return_inverse=True
        )

        triplet_index, _, _ = triplets(graph, distance < BOND_RADIUS)
        cosines = triplet_cosines(graph.edges_vectors(), triplet_index)

        return graph, edge_bond, distance[first], triplet_index, cosines

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in double precision."""
        graph, edge_bond, bond_length, triplet_index, cosines = self.graphs(structures)

        center = graph.edge_index[0, triplet_index[0]]
        triplet_bond = edge_bond[triplet_index]

        basis = self.bond_basis_expansion.rbf_expansion_ag(bond_length)

        atoms = self.atom_embedding(structures.z)
        bonds = self.bond_embedding(basis)
        weights = self.bond_weights_ag(basis)
        weights_angles = self.bond_weights_bg(
            self.bond_basis_expansion.rbf_expansion_bg(bond_length)
        )
        angles = self.angle_embedding(self.angle_basis_expansion(cosines))

        for atom_layer, bond_layer, angle_layer in zip(
            self.atom_conv_layers, self.bond_conv_layers, self.angle_layers
        ):
            atoms = atom_layer(atoms, bonds, weights, graph.edge_index, edge_bond)

            bonds = bond_layer(
                self.triplet_features(atoms, bonds, angles, center, triplet_bond),
                bonds,
                weights_angles,
                triplet_bond,
            )
            angles = angle_layer(
                self.triplet_features(atoms, bonds, angles, center, triplet_bond),
                angles,
            )

        atoms = self.atom_conv_layers[-1](
            atoms, bonds, weights, graph.edge_index, edge_bond
        )

        # energy of each atom, and reference energy of its element
        per_atom = self.mlp(self.readout_norm(atoms)).squeeze(-1).double()
        per_atom = per_atom + self.composition_model(structures.z)

        return scatter(per_atom, structures.batch, len(structures.num_atoms))

    @staticmethod
    def triplet_features(
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_bonds d"],
        angles: Float[torch.Tensor, "n_triplets d"],
        center: Int[torch.Tensor, "n_triplets"],
        triplet_bond: Int[torch.Tensor, "2 n_triplets"],
    ) -> Float[torch.Tensor, "n_triplets 4*d"]:
        """Features of the two bonds, of the angle and of the atom of each angle."""
        return torch.cat(
            [bonds[triplet_bond[0]], bonds[triplet_bond[1]], angles, atoms[center]],
            dim=1,
        )
