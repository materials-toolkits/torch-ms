"""Model of the energy with products of spherical harmonics on the pair graph."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.data.batching import Structures
from torchms.estimator.orb.model import (
    BesselBasis,
    polynomial_cutoff,
    spherical_harmonics,
)
from torchms.graph.cutoff import cutoff
from torchms.graph.sparse import scatter
from torchms.models.basis import DEGREES, NUM_HARMONICS, gaunt_coefficients

NUM_ELEMENTS = 118
NUM_DEGREES = 4
CUTOFF_POWER = 4


class EquivariantLayer(nn.Module):
    """Invariants of the environment of each atom, up to a number of factors.

    The environment of an atom is a sum over its neighbours of a learned
    function of the distance times the spherical harmonics of the direction
    times the features of the neighbour. Its invariants are, for each channel:

    * with one factor, its constant part (2 atoms);
    * with two factors, its squared norm in each degree l (3 atoms);
    * with three factors, the scalar product in each degree l of the
      environment with its own square, taken with the Gaunt coefficients
      (4 atoms).
    """

    gaunt: torch.Tensor
    degrees: torch.Tensor

    def __init__(self, correlation: int, channels: int, num_basis: int):
        super().__init__()
        self.correlation = correlation
        self.channels = channels

        self.linear_up = nn.Linear(channels, channels, bias=False)
        self.radial = nn.Sequential(
            nn.Linear(num_basis, channels),
            nn.SiLU(),
            nn.Linear(channels, NUM_DEGREES * channels),
        )

        num_invariants = (1, 1 + NUM_DEGREES, 1 + 2 * NUM_DEGREES)[correlation - 1]
        self.update = nn.Sequential(
            nn.Linear(num_invariants * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )

        self.register_buffer("gaunt", gaunt_coefficients(), persistent=False)
        self.register_buffer("degrees", torch.tensor(DEGREES), persistent=False)

    def environment(
        self,
        features: Float[torch.Tensor, "n_atoms c"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
    ) -> Float[torch.Tensor, "n_atoms c 16"]:
        """Sum over the neighbours of radial functions times harmonics."""
        sender, receiver = edge_index

        # (n_edges, c, 16): one weight per degree, repeated on its harmonics
        weights = self.radial(radial).view(-1, NUM_DEGREES, self.channels)
        weights = weights[:, self.degrees].transpose(1, 2)

        messages = (
            weights * self.linear_up(features)[sender].unsqueeze(-1)
        ) * harmonics.unsqueeze(1)

        return scatter(messages, receiver, features.shape[0])

    def invariants(
        self, environment: Float[torch.Tensor, "n_atoms c 16"]
    ) -> Float[torch.Tensor, "n_atoms c*k"]:
        """Invariants of the environment with one, two and three factors."""
        n_atoms = environment.shape[0]
        result = [environment[:, :, 0]]

        def per_degree(values: torch.Tensor) -> torch.Tensor:
            # sum of the components of each degree: (n_atoms, c, 16) → (n_atoms, c * 4)
            summed = values.new_zeros(n_atoms, self.channels, NUM_DEGREES)
            index = self.degrees.expand(n_atoms, self.channels, NUM_HARMONICS)
            return summed.scatter_add_(2, index, values).flatten(1)

        if self.correlation >= 2:
            result.append(per_degree(environment**2))

        if self.correlation >= 3:
            square = torch.einsum(
                "ijk,bci,bcj->bck", self.gaunt, environment, environment
            )
            result.append(per_degree(square * environment))

        return torch.cat(result, dim=1)

    def forward(
        self,
        features: Float[torch.Tensor, "n_atoms c"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
    ) -> Float[torch.Tensor, "n_atoms c"]:
        environment = self.environment(features, edge_index, radial, harmonics)

        return features + self.update(self.invariants(environment))


class EquivariantNetwork(nn.Module):
    """Energy from products of sums of spherical harmonics, as MACE.

    The graph of the model only has the pairs of atoms closer than a radius:
    no triplet is listed. An atom sums spherical harmonics (l up to 3) of the
    directions of its neighbours, and products of this sum with itself give
    terms on three and four atoms. The products are taken with the Gaunt
    coefficients, the couplings which do not change in a mirror.

    The energy is a sum over the atoms. It does not depend on the order of the
    atoms, on the orientation and the origin of the cell (rotations and
    reflections) or on the basis of the lattice, and it is multiplied by the
    number of cells in a supercell. Every contribution vanishes smoothly at
    the radius.

    Args:
        correlation: Number of factors of the products: 1 for terms on two
            atoms, 2 for three atoms, 3 for four atoms in each layer.
        radius: Radius of the graph in Å.
        channels: Number of features of the atoms.
        num_layers: Number of layers.
        num_basis: Number of functions of the basis of the distances.

    Examples:
        >>> model = EquivariantNetwork(correlation=3, channels=16)
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> model(structures).shape
        torch.Size([2])
    """

    def __init__(
        self,
        correlation: int = 3,
        radius: float = 5.0,
        channels: int = 64,
        num_layers: int = 2,
        num_basis: int = 8,
    ):
        super().__init__()
        if correlation not in (1, 2, 3):
            raise ValueError("correlation must be 1, 2 or 3")

        self.radius = radius
        self.atom_embedding = nn.Embedding(NUM_ELEMENTS + 1, channels)
        self.distance_basis = BesselBasis(radius, num_basis)
        self.layers = nn.ModuleList(
            EquivariantLayer(correlation, channels, num_basis)
            for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1)
        )

    def build_graph(self, structures: Structures) -> tuple[
        Int[torch.Tensor, "2 n_edges"],
        Float[torch.Tensor, "n_edges b"],
        Float[torch.Tensor, "n_edges 16"],
    ]:
        """Pairs of atoms of the model.

        Returns:
            Atoms of each edge, shape (2, n_edges), radial basis of its length
            with a smooth cutoff, shape (n_edges, num_basis), and spherical
            harmonics of its direction, shape (n_edges, 16).
        """
        graph = cutoff(structures, self.radius, self_images=True)
        vectors = graph.edges_vectors()
        distance = vectors.norm(dim=-1)

        radial = self.distance_basis(distance) * polynomial_cutoff(
            distance, self.radius, p=CUTOFF_POWER
        ).unsqueeze(-1)

        return graph.edge_index, radial, spherical_harmonics(vectors)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the energy of each structure."""
        edge_index, radial, harmonics = self.build_graph(structures)

        features = self.atom_embedding(structures.z)
        for layer in self.layers:
            features = layer(features, edge_index, radial, harmonics)

        per_atom = self.readout(features).squeeze(-1)

        return scatter(per_atom, structures.batch, len(structures.num_atoms))
