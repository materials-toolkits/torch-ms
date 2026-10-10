"""Network of the energy with explicit angles and dihedral angles.

The network does not build its graph: it reads the graph, the triplets and
the quadruplets attached to the structures, and gives the transform which
attaches them, LineGraphNetwork.transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.graph.geometry import dihedral_angles, triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddQuadruplets,
    AddRadiusGraph,
    AddTriplets,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.layers.basis import AngleBasis, BesselBasis, mlp, polynomial_cutoff

# what the network reads on the structures, for each order
FIELDS = ("graph", "triplets", "quadruplets")


@dataclass(frozen=True)
class Features:
    """Features of each order of a network.

    Attributes:
        atoms: Features of the atoms, shape (n_atoms, hidden).
        bonds: Features of the bonds, shape (n_edges, hidden).
        triplets: Features of the triplets, shape (n_triplets, hidden), None
            for an order of 1.
        quadruplets: Features of the quadruplets, shape (n_quadruplets,
            hidden), None for an order below 3.
    """

    atoms: Float[torch.Tensor, "n_atoms hidden"]
    bonds: Float[torch.Tensor, "n_edges hidden"]
    triplets: Float[torch.Tensor, "n_triplets hidden"] | None = None
    quadruplets: Float[torch.Tensor, "n_quadruplets hidden"] | None = None


@dataclass(frozen=True)
class Geometry:
    """Geometry and smooth cutoff of the bonds, triplets and quadruplets.

    The cutoff of a triplet or of a quadruplet is the product of the ones of
    its bonds, for the radius of its order.

    Attributes:
        distance: Length of each bond in Å, shape (n_edges,).
        bond_cutoff: Smooth cutoff of each bond, shape (n_edges, 1).
        angle: Angle of each triplet in radians, shape (n_triplets,), None
            for an order of 1.
        triplet_cutoff: Smooth cutoff of each triplet, shape (n_triplets, 1).
        dihedral: Dihedral angle of each quadruplet in radians, shape
            (n_quadruplets,), None for an order below 3.
        quadruplet_cutoff: Smooth cutoff of each quadruplet, shape
            (n_quadruplets, 1).
    """

    distance: Float[torch.Tensor, "n_edges"]
    bond_cutoff: Float[torch.Tensor, "n_edges 1"]
    angle: Float[torch.Tensor, "n_triplets"] | None = None
    triplet_cutoff: Float[torch.Tensor, "n_triplets 1"] | None = None
    dihedral: Float[torch.Tensor, "n_quadruplets"] | None = None
    quadruplet_cutoff: Float[torch.Tensor, "n_quadruplets 1"] | None = None


class LineGraphLayer(nn.Module):
    """Update from the highest order down: dihedrals, angles, bonds, atoms."""

    def __init__(self, order: int, hidden: int):
        super().__init__()
        self.order = order
        self.bond_update = mlp(3 * hidden, hidden, hidden)

        if order >= 2:
            self.triplet_update = mlp(4 * hidden, hidden, hidden)
        if order >= 3:
            self.quadruplet_update = mlp(4 * hidden, hidden, hidden)

    def forward(
        self,
        features: Features,
        geometry: Geometry,
        graph: Graph,
        triplets: EdgeTuples | None,
        quadruplets: EdgeTuples | None,
    ) -> Features:
        """Updated features of each order.

        Args:
            features: Features of the atoms, of the bonds, and of the triplets
                and the quadruplets of the order of the layer.
            geometry: Smooth cutoff of the bonds, triplets and quadruplets.
            graph: Graph of the atoms.
            triplets: Pairs of bonds of an atom, for an order of 2 and more.
            quadruplets: Chains of three bonds, for an order of 3.
        """
        atoms, bonds = features.atoms, features.bonds
        angles, dihedrals = features.triplets, features.quadruplets

        if quadruplets is not None and dihedrals is not None:
            message = self.quadruplet_update(
                torch.cat(
                    [
                        dihedrals,
                        quadruplets.unpool_edges(bonds, position=0),
                        quadruplets.unpool_edges(bonds, position=1),
                        quadruplets.unpool_edges(bonds, position=2),
                    ],
                    dim=1,
                )
            )
            dihedrals = dihedrals + message
            # a dihedral angle turns around its middle bond
            bonds = bonds + quadruplets.aggregate_to_edges(
                message * geometry.quadruplet_cutoff, position=1
            )

        if triplets is not None and angles is not None:
            # atom of each bond, the one of the angle for the bonds of a triplet
            center = graph.unpool_senders(atoms)
            message = self.triplet_update(
                torch.cat(
                    [
                        angles,
                        triplets.unpool_edges(bonds, position=0),
                        triplets.unpool_edges(bonds, position=1),
                        triplets.unpool_edges(center, position=0),
                    ],
                    dim=1,
                )
            )
            angles = angles + message
            bonds = bonds + triplets.aggregate_to_edges(
                message * geometry.triplet_cutoff, position=0
            )

        message = self.bond_update(
            torch.cat(
                [bonds, graph.unpool_senders(atoms), graph.unpool_receivers(atoms)],
                dim=1,
            )
        )
        atoms = atoms + graph.aggregate_edges_to_nodes(
            message * geometry.bond_cutoff, to="sender"
        )

        return Features(atoms, bonds + message, angles, dihedrals)


class LineGraphNetwork(nn.Module):
    """Energy from the bonds, the angles and the dihedral angles of a crystal.

    The model has features on the atoms and on the pairs of atoms closer than
    a radius. With order 2 it also has features on the triplets of atoms, the
    pairs of bonds around an atom, initialized from their angle. With order 3
    it has features on the quadruplets, the chains of three bonds, initialized
    from their dihedral angle. Each order has its own radius: the number of
    triplets grows with the square of the number of neighbours, the number of
    quadruplets with its cube.

    The energy is a sum over the atoms. It does not depend on the order of the
    atoms, on the orientation and the origin of the cell or on the basis of the
    lattice, and it is multiplied by the number of cells in a supercell. Every
    contribution vanishes smoothly at the radius of its order.

    The network does not build graphs: it reads the graph, the triplets and
    the quadruplets attached to the structures by its transform, which is
    applied when the structures are loaded or to a batch. Structures without
    them are refused.

    Args:
        order: 1 for bonds only, 2 to add angles, 3 to add dihedral angles.
        radius: Radius in Å of each order, in decreasing order.
        hidden: Number of features of atoms, bonds, angles and dihedral angles.
        num_layers: Number of layers.
        num_basis: Number of functions of the bases of distances and angles.
        chiral: If True, the model distinguishes a crystal from its mirror
            image, with the sign of the dihedral angles. Only for order 3.
        num_elements: Largest atomic number of an atom.
        cutoff_power: Power of the smooth cutoff of each order.
        cosine_margin: The cosines of the angles are multiplied by one minus
            this margin, which keeps them away from ±1 where the gradient of
            the angle is infinite.

    Examples:
        >>> model = LineGraphNetwork(order=2, hidden=16)
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> structures = model.transform()(structures)
        >>> model(structures).shape
        torch.Size([2])
    """

    def __init__(
        self,
        order: int = 2,
        radius: tuple[float, ...] = (5.0, 3.5, 3.0),
        hidden: int = 64,
        num_layers: int = 3,
        num_basis: int = 8,
        chiral: bool = False,
        num_elements: int = 118,
        cutoff_power: int = 4,
        cosine_margin: float = 1e-6,
    ):
        super().__init__()
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if len(radius) < order or any(a < b for a, b in zip(radius, radius[1:])):
            raise ValueError("one radius per order is needed, in decreasing order")
        if chiral and order < 3:
            raise ValueError("only dihedral angles distinguish mirror images")

        self.order = order
        self.radius = tuple(radius[:order])
        self.cutoff_power = cutoff_power
        self.cosine_margin = cosine_margin

        self.atom_embedding = nn.Embedding(num_elements + 1, hidden)
        self.distance_basis = BesselBasis(self.radius[0], num_basis)
        self.bond_embedding = mlp(num_basis, hidden, hidden)

        if order >= 2:
            self.angle_basis = AngleBasis(num_basis)
            self.triplet_embedding = mlp(num_basis, hidden, hidden)
        if order >= 3:
            self.dihedral_basis = AngleBasis(num_basis, sine=chiral)
            self.quadruplet_embedding = mlp(
                self.dihedral_basis.num_features, hidden, hidden
            )

        self.layers = nn.ModuleList(
            LineGraphLayer(order, hidden) for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def transform(self) -> Transform:
        """Transform attaching the graph and the tuples of edges of the network.

        Atoms closer than the first radius are bonded, periodic images of the
        atom itself included. With order 2, the triplets are the pairs of
        bonds of an atom shorter than the second radius. With order 3, the
        quadruplets are the chains of three bonds shorter than the third one.
        It depends on the order and on the radii of the network.
        """
        transforms: list[Transform] = [AddRadiusGraph(self.radius[0], self_images=True)]
        if self.order >= 2:
            transforms.append(AddTriplets(radius=self.radius[1], strict=True))
        if self.order >= 3:
            transforms.append(AddQuadruplets(radius=self.radius[2], strict=True))

        return Compose(transforms)

    def check_graphs(self, structures: Structures) -> None:
        """Check that structures have the graph and the tuples of edges of the network.

        What is attached is expected to be what transform attaches: only its
        presence is checked.

        Raises:
            ValueError: If one of them is not attached.
        """
        missing = [
            name for name in FIELDS[: self.order] if getattr(structures, name) is None
        ]
        if missing:
            name = type(self).__name__
            raise ValueError(
                f"{name} needs structures with their {', '.join(missing)}: "
                "apply the transform() of the network to the structures"
            )

    def envelope(
        self, distance: Float[torch.Tensor, "n"], radius: float
    ) -> Float[torch.Tensor, "n 1"]:
        """Smooth cutoff of each bond for a radius."""
        return polynomial_cutoff(distance, radius, p=self.cutoff_power).unsqueeze(-1)

    def geometry(self, structures: Structures) -> Geometry:
        """Geometry and smooth cutoff of the bonds, triplets and quadruplets.

        They are the ones of the graph and of the tuples of edges attached to
        the structures by transform.

        Raises:
            ValueError: If the structures do not have them.
        """
        self.check_graphs(structures)
        vectors = edges_vectors(attached_graph(structures, self))
        distance = vectors.norm(dim=-1)

        angle, triplet_cutoff, dihedral, quadruplet_cutoff = None, None, None, None

        if self.order >= 2:
            triplets = attached_tuples(structures, "triplets", self)
            envelope = self.envelope(distance, self.radius[1])
            angle = torch.acos(
                triplet_cosines(vectors, triplets) * (1.0 - self.cosine_margin)
            )
            triplet_cutoff = triplets.unpool_edges(
                envelope, position=0
            ) * triplets.unpool_edges(envelope, position=1)

        if self.order >= 3:
            quadruplets = attached_tuples(structures, "quadruplets", self)
            envelope = self.envelope(distance, self.radius[2])
            dihedral = dihedral_angles(vectors, quadruplets.index)
            quadruplet_cutoff = (
                quadruplets.unpool_edges(envelope, position=0)
                * quadruplets.unpool_edges(envelope, position=1)
                * quadruplets.unpool_edges(envelope, position=2)
            )

        return Geometry(
            distance,
            self.envelope(distance, self.radius[0]),
            angle,
            triplet_cutoff,
            dihedral,
            quadruplet_cutoff,
        )

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the energy of each structure.

        Args:
            structures: Structures with the graph and the tuples of edges of
                the network, see transform.

        Raises:
            ValueError: If the structures do not have them.
        """
        geometry = self.geometry(structures)
        graph = attached_graph(structures, self)

        # the tuples of the order of the network, None above it
        triplets = structures.triplets if self.order >= 2 else None
        quadruplets = structures.quadruplets if self.order >= 3 else None

        features = Features(
            atoms=self.atom_embedding(structures.z),
            bonds=self.bond_embedding(self.distance_basis(geometry.distance)),
            triplets=(
                None
                if geometry.angle is None
                else self.triplet_embedding(self.angle_basis(geometry.angle))
            ),
            quadruplets=(
                None
                if geometry.dihedral is None
                else self.quadruplet_embedding(self.dihedral_basis(geometry.dihedral))
            ),
        )

        for layer in self.layers:
            features = layer(features, geometry, graph, triplets, quadruplets)

        per_atom = self.readout(features.atoms).squeeze(-1)

        return graph.aggregate_nodes_to_structures(per_atom)
