"""Model of the energy with explicit angles and dihedral angles."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import nn

from torchms.data.batching import Structures
from torchms.estimator.orb.model import BesselBasis, polynomial_cutoff
from torchms.graph.cutoff import cutoff
from torchms.graph.geometry import dihedral_angles, triplet_cosines
from torchms.graph.line_graph import quadruplets, triplets
from torchms.graph.sparse import scatter
from torchms.models.basis import AngleBasis, mlp

NUM_ELEMENTS = 118
CUTOFF_POWER = 4


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
        features: dict[str, torch.Tensor],
        index: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        atoms, bonds = features["atoms"], features["bonds"]
        sender, receiver = index["bonds"]
        updated = {}

        if self.order >= 3:
            first, middle, last = index["quadruplets"]
            message = self.quadruplet_update(
                torch.cat(
                    [features["quadruplets"], bonds[first], bonds[middle], bonds[last]],
                    dim=1,
                )
            )
            updated["quadruplets"] = features["quadruplets"] + message
            # a dihedral angle turns around its middle bond
            bonds = bonds + scatter(
                message * weights["quadruplets"], middle, bonds.shape[0]
            )

        if self.order >= 2:
            first, second = index["triplets"]
            message = self.triplet_update(
                torch.cat(
                    [
                        features["triplets"],
                        bonds[first],
                        bonds[second],
                        atoms[sender[first]],
                    ],
                    dim=1,
                )
            )
            updated["triplets"] = features["triplets"] + message
            bonds = bonds + scatter(
                message * weights["triplets"], first, bonds.shape[0]
            )

        message = self.bond_update(
            torch.cat([bonds, atoms[sender], atoms[receiver]], dim=1)
        )
        updated["bonds"] = bonds + message
        updated["atoms"] = atoms + scatter(
            message * weights["bonds"], sender, atoms.shape[0]
        )

        return updated


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

    Args:
        order: 1 for bonds only, 2 to add angles, 3 to add dihedral angles.
        radius: Radius in Å of each order, in decreasing order.
        hidden: Number of features of atoms, bonds, angles and dihedral angles.
        num_layers: Number of layers.
        num_basis: Number of functions of the bases of distances and angles.
        chiral: If True, the model distinguishes a crystal from its mirror
            image, with the sign of the dihedral angles. Only for order 3.

    Examples:
        >>> model = LineGraphNetwork(order=2, hidden=16)
        >>> structures = Structures.collate([Structures.diamond(), Structures.zincblende()])
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

        self.atom_embedding = nn.Embedding(NUM_ELEMENTS + 1, hidden)
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

    def build_graph(self, structures: Structures) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Bonds, triplets and quadruplets of the model.

        Returns:
            Indices of each order ("bonds": atoms of each bond, "triplets" and
            "quadruplets": bonds of each one), their geometry ("bonds":
            lengths, "triplets": angles, "quadruplets": dihedral angles) and
            their smooth cutoff, with one column.
        """
        graph = cutoff(structures, self.radius[0], self_images=True)
        vectors = graph.edges_vectors()
        distance = vectors.norm(dim=-1)

        index = {"bonds": graph.edge_index}
        geometry = {"bonds": distance}
        weights = {"bonds": self.envelope(distance, self.radius[0])}

        if self.order >= 2:
            envelope = self.envelope(distance, self.radius[1])
            index["triplets"], _, _ = triplets(graph, distance < self.radius[1])
            geometry["triplets"] = torch.acos(
                triplet_cosines(vectors, index["triplets"]) * (1.0 - 1e-6)
            )
            weights["triplets"] = envelope[index["triplets"]].prod(dim=0)

        if self.order >= 3:
            envelope = self.envelope(distance, self.radius[2])
            index["quadruplets"], _, _ = quadruplets(graph, distance < self.radius[2])
            geometry["quadruplets"] = dihedral_angles(vectors, index["quadruplets"])
            weights["quadruplets"] = envelope[index["quadruplets"]].prod(dim=0)

        return index, geometry, weights

    @staticmethod
    def envelope(
        distance: Float[torch.Tensor, "n"], radius: float
    ) -> Float[torch.Tensor, "n 1"]:
        """Smooth cutoff of each bond for a radius."""
        return polynomial_cutoff(distance, radius, p=CUTOFF_POWER).unsqueeze(-1)

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the energy of each structure."""
        index, geometry, weights = self.build_graph(structures)

        features = {
            "atoms": self.atom_embedding(structures.z),
            "bonds": self.bond_embedding(self.distance_basis(geometry["bonds"])),
        }
        if self.order >= 2:
            features["triplets"] = self.triplet_embedding(
                self.angle_basis(geometry["triplets"])
            )
        if self.order >= 3:
            features["quadruplets"] = self.quadruplet_embedding(
                self.dihedral_basis(geometry["quadruplets"])
            )

        for layer in self.layers:
            features = layer(features, index, weights)

        per_atom = self.readout(features["atoms"]).squeeze(-1)

        return scatter(per_atom, structures.batch, len(structures.num_atoms))
