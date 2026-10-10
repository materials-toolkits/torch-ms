"""Network of the energy with products of spherical harmonics on the pair graph.

The network does not build its graph: it reads the graph attached to the
structures, and gives the transform which attaches it,
EquivariantNetwork.transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.graph.graph import Graph
from torchms.graph.transforms import AddRadiusGraph, attached_graph, edges_vectors
from torchms.models.layers.basis import (
    DEGREES,
    MAX_DEGREE,
    BesselBasis,
    gaunt_coefficients,
    polynomial_cutoff,
    spherical_harmonics,
)

# degrees l of the spherical harmonics of torchms.models.layers.basis, which
# implements them and their Gaunt coefficients up to MAX_DEGREE: not a
# hyperparameter
NUM_DEGREES = MAX_DEGREE + 1


@dataclass(frozen=True)
class Config:
    """Hyperparameters of an EquivariantNetwork.

    Args:
        correlation: Number of factors of the products: 1 for terms on two
            atoms, 2 for three atoms, 3 for four atoms in each layer.
        radius: Radius of the graph in Å.
        channels: Number of features of the atoms.
        num_layers: Number of layers.
        num_basis: Number of functions of the basis of the distances.
        num_elements: Largest atomic number of an atom.
        cutoff_power: Power of the smooth cutoff of the edges.

    Raises:
        ValueError: If the correlation is not 1, 2 or 3.

    Examples:
        >>> Config(correlation=2).radius
        5.0
        >>> Config(correlation=4)
        Traceback (most recent call last):
        ...
        ValueError: correlation must be 1, 2 or 3
    """

    correlation: int = 3
    radius: float = 5.0
    channels: int = 64
    num_layers: int = 2
    num_basis: int = 8
    num_elements: int = 118
    cutoff_power: int = 4

    def __post_init__(self):
        if self.correlation not in (1, 2, 3):
            raise ValueError("correlation must be 1, 2 or 3")


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
    degree_of_harmonic: torch.Tensor

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
        # 1 where a harmonic has a degree, shape (num_harmonics, num_degrees):
        # a product with it sums the harmonics of each degree, a product with
        # its transpose gives each harmonic the value of its degree
        self.register_buffer(
            "degree_of_harmonic",
            nn.functional.one_hot(torch.tensor(DEGREES), NUM_DEGREES).float(),
            persistent=False,
        )

    def environment(
        self,
        features: Float[torch.Tensor, "n_atoms c"],
        graph: Graph,
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
    ) -> Float[torch.Tensor, "n_atoms c 16"]:
        """Sum over the neighbours of radial functions times harmonics."""
        # (n_edges, c, 16): one weight per degree, repeated on its harmonics
        weights = self.radial(radial).view(-1, NUM_DEGREES, self.channels)
        weights = weights.transpose(1, 2) @ self.degree_of_harmonic.t()

        neighbours = graph.unpool_senders(self.linear_up(features))
        messages = (weights * neighbours.unsqueeze(-1)) * harmonics.unsqueeze(1)

        return graph.aggregate_edges_to_nodes(messages)

    def invariants(
        self, environment: Float[torch.Tensor, "n_atoms c 16"]
    ) -> Float[torch.Tensor, "n_atoms c*k"]:
        """Invariants of the environment with one, two and three factors."""
        result = [environment[:, :, 0]]

        def per_degree(values: torch.Tensor) -> torch.Tensor:
            # sum of the components of each degree: (n_atoms, c, 16) → (n_atoms, c * 4)
            return (values @ self.degree_of_harmonic).flatten(1)

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
        graph: Graph,
        radial: Float[torch.Tensor, "n_edges b"],
        harmonics: Float[torch.Tensor, "n_edges 16"],
    ) -> Float[torch.Tensor, "n_atoms c"]:
        environment = self.environment(features, graph, radial, harmonics)

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

    The network does not build graphs: it reads the graph attached to the
    structures by its transform, which is applied when the structures are
    loaded or to a batch. Structures without a graph are refused.

    Args:
        config: Hyperparameters of the network, the default ones of Config
            without argument.

    Examples:
        >>> model = EquivariantNetwork(Config(correlation=3, channels=16))
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> model(structures).shape
        torch.Size([2])
    """

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config = Config() if config is None else config
        channels = config.channels

        self.radius = config.radius
        self.cutoff_power = config.cutoff_power
        self.atom_embedding = nn.Embedding(config.num_elements + 1, channels)
        self.distance_basis = BesselBasis(config.radius, config.num_basis)
        self.layers = nn.ModuleList(
            EquivariantLayer(config.correlation, channels, config.num_basis)
            for _ in range(config.num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1)
        )

    def transform(self) -> Transform:
        """Transform attaching the graph of the network.

        Atoms closer than the radius are connected, periodic images of the
        atom itself included. It depends on the radius of the network.
        """
        return Compose([AddRadiusGraph(self.radius, self_images=True)])

    def check_graphs(self, structures: Structures) -> None:
        """Check that structures have the graph of the network.

        What is attached is expected to be what transform attaches: only its
        presence is checked.

        Raises:
            ValueError: If the structures have no graph.
        """
        if structures.graph is None:
            name = type(self).__name__
            raise ValueError(
                f"{name} needs structures with their graph: "
                "apply the transform() of the network to the structures"
            )

    def edges(self, structures: Structures) -> tuple[
        Graph,
        Float[torch.Tensor, "n_edges b"],
        Float[torch.Tensor, "n_edges 16"],
    ]:
        """Graph of the structures, with the bases of its edges.

        The graph is the one attached to the structures by transform.

        Returns:
            Graph of the atoms, radial basis of the length of each edge
            with a smooth cutoff, shape (n_edges, num_basis), and spherical
            harmonics of its direction, shape (n_edges, 16).

        Raises:
            ValueError: If the structures have no graph.
        """
        self.check_graphs(structures)
        graph = attached_graph(structures, self)
        vectors = edges_vectors(graph)
        distance = vectors.norm(dim=-1)

        radial = self.distance_basis(distance) * polynomial_cutoff(
            distance, self.radius, p=self.cutoff_power
        ).unsqueeze(-1)

        return graph, radial, spherical_harmonics(vectors)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the energy of each structure.

        Args:
            structures: Structures with the graph of the network, see
                transform.

        Raises:
            ValueError: If the structures have no graph.
        """
        graph, radial, harmonics = self.edges(structures)

        features = self.atom_embedding(structures.z)
        for layer in self.layers:
            features = layer(features, graph, radial, harmonics)

        per_atom = self.readout(features).squeeze(-1)

        return graph.aggregate_nodes_to_structures(per_atom)
