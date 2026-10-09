"""Batching utilities for atomic structures: crystals, molecules, or both."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch.utils.data._utils.collate import default_collate_fn_map

if TYPE_CHECKING:
    from torchms.graph.graph import EdgeTuples, Graph

TensorLike = torch.Tensor | np.ndarray | Sequence[Any]


# fields of Structures holding tuples of edges of the graph
TUPLES_FIELDS = ("line_graph", "triplets", "quadruplets", "bonds")


class InvalidStructure(Exception):
    """Exception raised when a structure has invalid data."""


@dataclass(init=False)
class Structures:  # pylint: disable=too-many-instance-attributes
    """Container for batched structures, crystals or molecules, and their properties.

    The position of an atom in Å is x @ lattice, for every structure. For a
    crystal, lattice is its cell and x its fractional coordinates. A molecule
    has no cell: its lattice is only a frame, the identity unless it was
    rotated, and x its coordinates in this frame, in Å. pbc says along which
    lattice vectors a structure is periodic. Crystals and molecules can be
    batched together.

    Attributes:
        lattice: Lattice vectors with shape (n_struct, 3, 3).
        x: Atomic coordinates in the lattice with shape (n_atoms, 3): in [0, 1)
            along a periodic lattice vector, unbounded along the others.
        z: Atomic numbers with shape (n_atoms,).
        num_atoms: Number of atoms per structure with shape (n_struct,).
        batch: Batch index for each atom with shape (n_atoms,).
        ptr: Index pointers for each structure with shape (n_struct + 1,).
        total_energy: Total energy per structure in eV with shape (n_struct,).
        energy_above_hull: Energy above hull per structure in eV with shape (n_struct,).
        band_gap: Band gap per structure in eV with shape (n_struct,).
        formation_energy_per_atom: Formation energy per structure in eV/atom
            with shape (n_struct,).
        pbc: Whether each structure is periodic along each of its lattice
            vectors, with shape (n_struct, 3). All True for a crystal, the
            default, all False for a molecule.
        properties: Other properties per structure, by name, each one with
            shape (n_struct,). NaN for a structure without value. Empty by
            default.
        graph: Graph of the atoms, None unless attached by a transform of
            torchms.graph.transforms (AddRadiusGraph, AddAtomicRadiusGraph).
            Its edges are pairs of atoms of the batch, see Graph for its fields.
        line_graph: Line graph of the graph, None unless attached
            (AddLineGraph): pairs of edges u→v then v→w, the first one arrives
            at the atom the second one leaves. See EdgeTuples for its fields.
        triplets: Triplets of atoms of the graph, None unless attached
            (AddTriplets): pairs of edges i→j and i→k leaving the same atom.
            See EdgeTuples for its fields.
        quadruplets: Quadruplets of atoms of the graph, None unless attached
            (AddQuadruplets): chains of three edges u→v, v→w, w→x. See
            EdgeTuples for its fields.
        bonds: Bonds of the graph, None unless attached (AddBonds): pairs of
            an edge and its reverse, one pair per bond. See EdgeTuples for its
            fields.

    The graph and the tuples of edges describe the structures as they are: they
    are batched with the structures by collate, moved with them by to, and
    removed by the augmentations, which change the atoms or the cell.

    Examples:
        Creating a Diamond crystal structure (C, a=3.567 A, 8 atoms):

        >>> a = 3.567
        >>> lattice = torch.tensor(
        ...     [[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]],
        ...     dtype=torch.float32,
        ... )
        >>> x = [
        ...     [0.0, 0.0, 0.0],
        ...     [0.5, 0.5, 0.0],
        ...     [0.5, 0.0, 0.5],
        ...     [0.0, 0.5, 0.5],
        ...     [0.25, 0.25, 0.25],
        ...     [0.75, 0.75, 0.25],
        ...     [0.75, 0.25, 0.75],
        ...     [0.25, 0.75, 0.75],
        ... ]
        >>> z = torch.tensor([6] * 8, dtype=torch.int64)
        >>> struct = Structures(lattice=lattice, x=x, z=z)
        >>> struct.num_atoms
        tensor([8])
    """

    lattice: Float[torch.Tensor, "n_struct 3 3"]
    x: Float[torch.Tensor, "n_atoms 3"]
    z: Int[torch.Tensor, "n_atoms"]
    num_atoms: Int[torch.Tensor, "n_struct"]
    batch: Int[torch.Tensor, "n_atoms"]
    ptr: Int[torch.Tensor, "n_struct+1"]
    total_energy: Float[torch.Tensor, "n_struct"]
    energy_above_hull: Float[torch.Tensor, "n_struct"]
    band_gap: Float[torch.Tensor, "n_struct"]
    formation_energy_per_atom: Float[torch.Tensor, "n_struct"]
    pbc: Bool[torch.Tensor, "n_struct 3"]
    properties: dict[str, Float[torch.Tensor, "n_struct"]]
    graph: Graph | None
    line_graph: EdgeTuples | None
    triplets: EdgeTuples | None
    quadruplets: EdgeTuples | None
    bonds: EdgeTuples | None
    _validate: bool

    def __init__(
        self,
        lattice: TensorLike,
        x: TensorLike,
        z: TensorLike,
        num_atoms: TensorLike | None = None,
        batch: TensorLike | None = None,
        ptr: TensorLike | None = None,
        total_energy: TensorLike | None = None,
        energy_above_hull: TensorLike | None = None,
        band_gap: TensorLike | None = None,
        formation_energy_per_atom: TensorLike | None = None,
        graph: Graph | None = None,
        line_graph: EdgeTuples | None = None,
        triplets: EdgeTuples | None = None,
        quadruplets: EdgeTuples | None = None,
        bonds: EdgeTuples | None = None,
        pbc: TensorLike | None = None,
        properties: Mapping[str, TensorLike] | None = None,
        _validate: bool = False,
    ):
        self.lattice = self.as_tensor("lattice", lattice, torch.float32)
        self.x = self.as_tensor("x", x, torch.float32)
        self.z = self.as_tensor("z", z, torch.int64)

        device = self.lattice.device
        n_struct = len(self.lattice)

        # Missing fields are derived from the others, they are checked with
        # the given ones in check_tensors.
        if num_atoms is None:
            self.num_atoms = torch.tensor(
                [len(self.z)], dtype=torch.int64, device=device
            )
        else:
            self.num_atoms = self.as_tensor("num_atoms", num_atoms, torch.int64)

        if batch is None:
            counts = self.num_atoms.flatten().long().clamp(min=0)
            self.batch = torch.arange(
                len(counts), dtype=torch.int64, device=counts.device
            ).repeat_interleave(counts)
        else:
            self.batch = self.as_tensor("batch", batch, torch.int64)

        if ptr is None:
            counts = self.num_atoms.flatten().long().clamp(min=0)
            self.ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)
        else:
            self.ptr = self.as_tensor("ptr", ptr, torch.int64)

        self.total_energy = self.as_property(
            "total_energy", total_energy, n_struct, device
        )
        self.energy_above_hull = self.as_property(
            "energy_above_hull", energy_above_hull, n_struct, device
        )
        self.band_gap = self.as_property("band_gap", band_gap, n_struct, device)
        self.formation_energy_per_atom = self.as_property(
            "formation_energy_per_atom", formation_energy_per_atom, n_struct, device
        )

        if pbc is None:
            self.pbc = torch.ones((n_struct, 3), dtype=torch.bool, device=device)
        else:
            self.pbc = self.as_tensor("pbc", pbc, torch.bool)

        self.properties = {
            name: self.as_tensor(name, values, torch.float32)
            for name, values in (properties or {}).items()
        }

        self.graph = graph
        self.line_graph = line_graph
        self.triplets = triplets
        self.quadruplets = quadruplets
        self.bonds = bonds

        self._validate = _validate
        if self._validate:
            return

        self.check_tensors()

        self._validate = True

    @classmethod
    def as_tensor(
        cls, field: str, values: TensorLike, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert the values of a field to a tensor, tensors are left unchanged.

        Raises:
            InvalidStructure: If values cannot be converted to a tensor.
        """
        if isinstance(values, torch.Tensor):
            return values

        try:
            return torch.tensor(values, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise InvalidStructure(f"invalid field {field}: {e}") from e

    @classmethod
    def as_property(
        cls, field: str, values: TensorLike | None, n_struct: int, device: torch.device
    ) -> torch.Tensor:
        """Convert a property per structure to a tensor, NaN if unknown."""
        if values is None:
            return torch.full(
                (n_struct,), torch.nan, dtype=torch.float32, device=device
            )

        return cls.as_tensor(field, values, torch.float32)

    def check_tensors(self):
        """Validate structure data and cast fields to their dtype.

        Raises:
            InvalidStructure: If lattice or atom data is invalid.
        """
        device = self.lattice.device

        n_struct = len(self.lattice)
        n_atoms = len(self.z)

        if n_struct < 1:
            raise InvalidStructure("must contain at least one structure")

        if n_atoms < 1:
            raise InvalidStructure("must contain at least one atom per structure")

        self.to_tensor(
            "lattice", dtype=torch.float32, shape=(n_struct, 3, 3), device=device
        )
        self.to_tensor("x", dtype=torch.float32, shape=(n_atoms, 3), device=device)
        self.to_tensor("z", dtype=torch.int64, shape=(n_atoms,), device=device)
        self.to_tensor("num_atoms", dtype=torch.int64, shape=(n_struct,), device=device)

        if (self.num_atoms <= 0).any():
            raise InvalidStructure("must contain at least one atom per structure")

        self.to_tensor("batch", dtype=torch.int64, shape=(n_atoms,), device=device)
        if (self.batch[:-1] > self.batch[1:]).any():
            raise InvalidStructure(
                "atoms of structures must be continuous and index in "
                'ascending order in the field "batch"'
            )

        self.to_tensor("ptr", dtype=torch.int64, shape=(n_struct + 1,), device=device)
        if (self.ptr[:-1] > self.ptr[1:]).any():
            raise InvalidStructure(
                "atoms of structures must be continuous and index in "
                'ascending order in the field "ptr"'
            )

        self.to_tensor(
            "total_energy", dtype=torch.float32, shape=(n_struct,), device=device
        )
        self.to_tensor(
            "energy_above_hull", dtype=torch.float32, shape=(n_struct,), device=device
        )
        self.to_tensor(
            "band_gap", dtype=torch.float32, shape=(n_struct,), device=device
        )
        self.to_tensor(
            "formation_energy_per_atom",
            dtype=torch.float32,
            shape=(n_struct,),
            device=device,
        )

        self.to_tensor("pbc", dtype=torch.bool, shape=(n_struct, 3), device=device)

        for name, values in self.properties.items():
            if values.shape != (n_struct,):
                raise InvalidStructure(
                    f'invalid shape {values.shape} for property "{name}", '
                    f"must be {(n_struct,)}"
                )

            if values.device != device:
                raise InvalidStructure(
                    f'inconsistent device for property "{name}", '
                    f"{device}!={values.device}"
                )

            self.properties[name] = values.to(torch.float32)

        self.check_attached()

    def check_attached(self):
        """Validate the graph and the tuples of edges attached to the structures.

        Raises:
            InvalidStructure: If the graph is not a graph of these structures,
                or if tuples of edges are not tuples of this graph.
        """
        device = self.lattice.device

        if self.graph is not None:
            if self.graph.device() != device:
                raise InvalidStructure(
                    "inconsistent device for the graph, "
                    f"{device}!={self.graph.device()}"
                )

            if not torch.equal(self.graph.num_atoms, self.num_atoms):
                raise InvalidStructure(
                    "the graph attached must have the atoms of the structures"
                )

        for name in TUPLES_FIELDS:
            tuples: EdgeTuples | None = getattr(self, name)

            if tuples is None:
                continue

            if self.graph is None:
                raise InvalidStructure(f'"{name}" needs a graph attached')

            if tuples.device() != device:
                raise InvalidStructure(
                    f'inconsistent device for "{name}", {device}!={tuples.device()}'
                )

            if not torch.equal(tuples.num_edges, self.graph.num_edges):
                raise InvalidStructure(
                    f'"{name}" must be tuples of the edges of the graph attached'
                )

    def without_graphs(self) -> Structures:
        """The structures without their graph and their tuples of edges.

        Returns:
            Structures sharing the same tensors with nothing attached, or the
            structures themselves if they have nothing attached.

        Examples:
            >>> from torchms.graph.transforms import AddRadiusGraph
            >>> structures = AddRadiusGraph(2.0)(Structures.diamond())
            >>> structures.graph.num_edges
            tensor([32])
            >>> structures.without_graphs().graph is None
            True
        """
        if self.graph is None and all(
            getattr(self, name) is None for name in TUPLES_FIELDS
        ):
            return self

        return replace(self, graph=None, **{name: None for name in TUPLES_FIELDS})

    def device(self) -> torch.device:
        """Get the device of the lattice tensor.

        Returns:
            torch.device: Device on which the structure tensors are placed.
        """
        return self.lattice.device

    def to(self, device: torch.device | str) -> Structures:
        """Copy of the structures with all their tensors on a device.

        Args:
            device: Device of the copy, "cuda" for instance.

        Returns:
            Structures on the device, with their graph and their tuples of
            edges if they have some. Tensors already on it are shared with the
            original ones.

        Examples:
            >>> structures = Structures.diamond().to("cpu")
            >>> structures.device()
            device(type='cpu')
        """
        tensors = {
            field.name: getattr(self, field.name).to(device)
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        }

        properties = {
            name: values.to(device) for name, values in self.properties.items()
        }

        moved = replace(self.without_graphs(), properties=properties, **tensors)

        if self.graph is None:
            return moved

        # the graph refers to the structures moved, with nothing attached
        attached: dict[str, Any] = {"graph": self.graph.moved(device, moved)}
        for name in TUPLES_FIELDS:
            tuples: EdgeTuples | None = getattr(self, name)
            attached[name] = None if tuples is None else tuples.to(device)

        return replace(moved, **attached)

    def to_tensor(
        self,
        field: str,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Check a field and cast it to the specified dtype.

        Args:
            field: Name of the attribute to check.
            dtype: Target tensor dtype.
            shape: Expected tensor shape.
            device: Expected tensor device.

        Raises:
            InvalidStructure: If field value is invalid or incompatible.
        """
        tensor: torch.Tensor = getattr(self, field)

        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)

        if tensor.shape != shape:
            raise InvalidStructure(
                f'invalid shape {tensor.shape} for field "{field}", must be {shape}'
            )

        if tensor.device != device:
            raise InvalidStructure(
                f'inconsistent device for field "{field}", {device}!={tensor.device}'
            )

        setattr(self, field, tensor)

    @classmethod
    def diamond(cls, a: float = 3.567, element: int = 6) -> Structures:
        """Create a diamond crystal structure.

        Args:
            a: Lattice parameter in Angstroms (default: 3.567 Å for carbon).
            element: Atomic number (default: 6 for carbon).

        Returns:
            Structures object with diamond crystal structure
            (8 atoms per conventional cell).

        References:
            https://lampx.tugraz.at/~hadley/ss1/crystalstructure/structures/diamond/diamond.php
        """
        lattice = torch.tensor(
            [[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]],
            dtype=torch.float32,
        )
        positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
        coords = torch.tensor(positions, dtype=torch.float32)
        z = torch.tensor([element] * 8, dtype=torch.int64)

        return cls(
            lattice=lattice,
            x=coords,
            z=z,
        )

    @classmethod
    def zincblende(
        cls, a: float = 5.41, cation: int = 30, anion: int = 16
    ) -> Structures:
        """Create a zincblende (sphalerite) crystal structure.

        Args:
            a: Lattice parameter in Angstroms (default: 5.41 Å for ZnS).
            cation: Atomic number for cation (default: 30 for Zn).
            anion: Atomic number for anion (default: 16 for S).

        Returns:
            Structures object with zincblende crystal structure
            (8 atoms per conventional cell: 4 cations + 4 anions).

        References:
            http://lampz.tugraz.at/~hadley/ss1/crystalstructure/structures/zincblende/zincblende_jsmol.php
        """
        lattice = torch.tensor(
            [[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]],
            dtype=torch.float32,
        )
        positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
        coords = torch.tensor(positions, dtype=torch.float32)
        z = torch.tensor([anion] * 4 + [cation] * 4, dtype=torch.int64)

        return cls(
            lattice=lattice,
            x=coords,
            z=z,
        )

    @classmethod
    def molecule(cls, positions: TensorLike, z: TensorLike) -> Structures:
        """Create a molecule, a structure which is not periodic.

        Its lattice is the identity, x is then its positions in Å.

        Args:
            positions: Positions of the atoms in Å, shape (n_atoms, 3).
            z: Atomic numbers, shape (n_atoms,).

        Returns:
            Structures object with one molecule.

        Examples:
            >>> molecule = Structures.molecule([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], [1, 1])
            >>> molecule.pbc
            tensor([[False, False, False]])
            >>> molecule.positions()
            tensor([[0.0000, 0.0000, 0.0000],
                    [0.7400, 0.0000, 0.0000]])
        """
        x = cls.as_tensor("x", positions, torch.float32)

        return cls(
            lattice=torch.eye(3, dtype=torch.float32, device=x.device).unsqueeze(0),
            x=x,
            z=z,
            pbc=torch.zeros((1, 3), dtype=torch.bool, device=x.device),
        )

    @classmethod
    def water(cls) -> Structures:
        """Create a molecule of water: O-H of 0.9572 Å, H-O-H of 104.52°.

        Returns:
            Structures object with one molecule, the oxygen first.
        """
        return cls.molecule(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]], [8, 1, 1]
        )

    def positions(self) -> Float[torch.Tensor, "n_atoms 3"]:
        """Positions of the atoms in Å: x @ lattice, for each structure.

        Returns:
            Cartesian positions with shape (n_atoms, 3).

        Examples:
            >>> Structures.diamond(a=2.0).positions()[4]
            tensor([0.5000, 0.5000, 0.5000])
        """
        # the rows of a lattice are its vectors
        return (self.x.unsqueeze(-1) * self.lattice[self.batch]).sum(dim=1)

    def _properties_of(
        self, names: Sequence[str]
    ) -> dict[str, Float[torch.Tensor, "n_struct"]]:
        """The properties with these names, NaN for the ones the structures lack."""
        n_struct = len(self.num_atoms)

        return {
            name: (
                self.properties[name]
                if name in self.properties
                else self.as_property(name, None, n_struct, self.device())
            )
            for name in names
        }

    @classmethod
    def collate(cls, structures: list[Structures]) -> Structures:
        """Combine multiple structures into a single batched structure.

        The graph and the tuples of edges attached to the structures are
        batched with them. The atoms of the edges of an element are shifted by
        the number of atoms of the elements before it, and the edges of its
        tuples by the number of edges of the elements before it: in the result,
        graph.edge_index points to the atoms of the batch, and the index of the
        tuples to the edges of the batch.

        Crystals and molecules can be batched together. The batch has every
        property of its elements, NaN for the structures of an element without
        this property.

        Args:
            structures: List of Structures to batch together, each one with one
                or several structures.

        Returns:
            A single Structures object with concatenated tensors.

        Raises:
            ValueError: If structures list is empty, or if only some of the
                elements have a graph, tuples of edges or their geometry.

        Examples:
            >>> struct = Structures.diamond()
            >>> batched = Structures.collate([struct, struct])
            >>> batched.lattice.shape
            torch.Size([2, 3, 3])

            A crystal and a molecule:

            >>> mixed = Structures.collate([struct, Structures.water()])
            >>> mixed.num_atoms, mixed.pbc.all(dim=1)
            (tensor([8, 3]), tensor([ True, False]))

            With a graph: 8 atoms and 32 edges per structure.

            >>> from torchms.graph.transforms import AddRadiusGraph, AddTriplets
            >>> struct = AddTriplets()(AddRadiusGraph(2.0)(struct))
            >>> batched = Structures.collate([struct, struct])
            >>> batched.graph.edge_ptr
            tensor([ 0, 32, 64])
            >>> int(batched.graph.edge_index[:, 32:].min())  # atoms of the second
            8
            >>> int(batched.triplets.index[:, 96:].min())  # edges of the second
            32
        """
        if len(structures) == 0:
            raise ValueError("structures list cannot be empty")

        device = structures[0].device()

        lattice = torch.cat([s.lattice for s in structures], dim=0)
        x = torch.cat([s.x for s in structures], dim=0)
        z = torch.cat([s.z for s in structures], dim=0)
        num_atoms = torch.cat([s.num_atoms for s in structures], dim=0)

        # a structure of the list may be a batch of several structures
        batch = torch.arange(
            len(num_atoms), dtype=torch.int64, device=device
        ).repeat_interleave(num_atoms)

        ptr = F.pad(torch.cumsum(num_atoms, dim=0), (1, 0), value=0)

        total_energy = torch.cat([s.total_energy for s in structures])
        energy_above_hull = torch.cat([s.energy_above_hull for s in structures])
        band_gap = torch.cat([s.band_gap for s in structures])
        formation_energy_per_atom = torch.cat(
            [s.formation_energy_per_atom for s in structures]
        )
        pbc = torch.cat([s.pbc for s in structures], dim=0)

        # every property of an element, in the order they are met
        names = list(dict.fromkeys(name for s in structures for name in s.properties))
        each = [
            s._properties_of(names) for s in structures
        ]  # pylint: disable=protected-access
        properties = {name: torch.cat([p[name] for p in each]) for name in names}

        batched = cls(
            lattice=lattice,
            x=x,
            z=z,
            num_atoms=num_atoms,
            batch=batch,
            ptr=ptr,
            total_energy=total_energy,
            energy_above_hull=energy_above_hull,
            band_gap=band_gap,
            formation_energy_per_atom=formation_energy_per_atom,
            pbc=pbc,
            properties=properties,
            _validate=True,
        )

        attached: dict[str, Any] = {}
        for name in ("graph", *TUPLES_FIELDS):
            elements = [getattr(s, name) for s in structures]
            present = [element for element in elements if element is not None]

            if len(present) == 0:
                continue

            if len(present) != len(elements):
                raise ValueError(
                    f'cannot collate: only some of the structures have "{name}"'
                )

            if name == "graph":
                attached[name] = type(present[0]).collate(present, batched)
            else:
                attached[name] = type(present[0]).collate(present)

        if len(attached) == 0:
            return batched

        return replace(batched, **attached)

    @classmethod
    def separate(cls, structures: Structures) -> list[Structures]:
        """Split a batched structure into individual structures.

        Args:
            structures: A batched Structures object.

        Returns:
            List of individual Structures objects, each one with its graph and
            its tuples of edges if the batch has some.

        Examples:
            >>> struct = Structures.diamond()
            >>> batched = Structures.collate([struct, struct])
            >>> individual = Structures.separate(batched)
            >>> len(individual)
            2
        """
        n_structs = len(structures.num_atoms)
        result = []

        for i in range(n_structs):
            start = structures.ptr[i].item()
            end = structures.ptr[i + 1].item()

            struct = cls(
                lattice=structures.lattice[i : i + 1],
                x=structures.x[start:end],
                z=structures.z[start:end],
                num_atoms=structures.num_atoms[i : i + 1],
                batch=structures.batch[start:end] - i,
                ptr=torch.tensor(
                    [0, structures.ptr[i + 1] - structures.ptr[i]],
                    device=structures.device(),
                ),
                total_energy=structures.total_energy[i : i + 1],
                energy_above_hull=structures.energy_above_hull[i : i + 1],
                band_gap=structures.band_gap[i : i + 1],
                formation_energy_per_atom=structures.formation_energy_per_atom[
                    i : i + 1
                ],
                pbc=structures.pbc[i : i + 1],
                properties={
                    name: values[i : i + 1]
                    for name, values in structures.properties.items()
                },
                _validate=True,
            )
            result.append(struct)

        if structures.graph is None:
            return result

        attached: dict[str, list[Any]] = {
            "graph": type(structures.graph).separate(structures.graph, result)
        }
        for name in TUPLES_FIELDS:
            tuples: EdgeTuples | None = getattr(structures, name)
            if tuples is not None:
                attached[name] = type(tuples).separate(tuples)

        return [
            replace(struct, **{name: values[i] for name, values in attached.items()})
            for i, struct in enumerate(result)
        ]

    def select(self, index: Int[torch.Tensor, "n"]) -> Structures:
        """Batch of some of the structures, in any order.

        Args:
            index: Indices of the structures, on the device of the structures,
                shape (n,). In any order, the same one can be given several
                times.

        Returns:
            Structures in the order of the indices, with copies of their atoms.

        Raises:
            ValueError: If the structures have a graph: its edges are not
                selected. Build the graphs of the batch selected.

        Examples:
            >>> structures = [Structures.diamond(), Structures.zincblende()]
            >>> batch = Structures.collate(structures)
            >>> selected = batch.select(torch.tensor([1, 1, 0]))
            >>> selected.num_atoms, selected.ptr
            (tensor([8, 8, 8]), tensor([ 0,  8, 16, 24]))
            >>> selected.z[:8].unique(), selected.z[16:].unique()
            (tensor([16, 30]), tensor([6]))
        """
        if self.without_graphs() is not self:
            raise ValueError(
                "structures with a graph cannot be selected: build the graphs "
                "of the batch selected"
            )

        device = self.lattice.device
        num_atoms = self.num_atoms[index]
        ptr = F.pad(torch.cumsum(num_atoms, dim=0), (1, 0), value=0)
        batch = torch.arange(len(index), device=device).repeat_interleave(num_atoms)

        # atoms of the structures selected: the atom k of a structure is the
        # atom k after the first one of this structure
        atoms = (
            torch.arange(len(batch), device=device)
            + (self.ptr[index] - ptr[:-1])[batch]
        )

        return Structures(
            lattice=self.lattice[index],
            x=self.x[atoms],
            z=self.z[atoms],
            num_atoms=num_atoms,
            batch=batch,
            ptr=ptr,
            total_energy=self.total_energy[index],
            energy_above_hull=self.energy_above_hull[index],
            band_gap=self.band_gap[index],
            formation_energy_per_atom=self.formation_energy_per_atom[index],
            pbc=self.pbc[index],
            properties={
                name: values[index] for name, values in self.properties.items()
            },
            _validate=True,
        )

    def split(self, batch_size: int) -> list[Structures]:
        """Split a batch into smaller batches of consecutive structures.

        The structures keep their order: the first batch_size ones, then the
        next ones, and a last batch which can be smaller. The graph and the
        tuples of edges attached are split with them, their indices starting
        again at zero in each batch: the result is what collate gives on the
        structures of each batch.

        It is meant for batches on a GPU. Building the graphs of one large
        batch then splitting it is faster than building the graphs of each
        small batch. The batches share the memory of the tensors which are not
        indices: the large batch is freed when all of them are.

        Args:
            batch_size: Number of structures per batch.

        Returns:
            List of Structures, in the order of the structures.

        Raises:
            ValueError: If batch_size is not positive.

        Examples:
            >>> from torchms.graph.transforms import AddRadiusGraph
            >>> batch = Structures.collate([Structures.diamond()] * 5)
            >>> batch = AddRadiusGraph(2.0)(batch)
            >>> [len(part.num_atoms) for part in batch.split(2)]
            [2, 2, 1]
            >>> last = batch.split(2)[-1]
            >>> last.ptr, last.graph.edge_ptr, int(last.graph.edge_index.max())
            (tensor([0, 8]), tensor([ 0, 32]), 7)
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        n_struct = len(self.num_atoms)
        bounds = list(range(0, n_struct, batch_size)) + [n_struct]

        # first atom, edge and tuple of each batch, read from the device once
        # for all the batches
        pointers = [self.ptr]
        if self.graph is not None:
            pointers.append(self.graph.edge_ptr)
        names = [name for name in TUPLES_FIELDS if getattr(self, name) is not None]
        pointers += [getattr(self, name).ptr for name in names]
        firsts = torch.stack(pointers)[:, bounds].tolist()

        result = []
        for k, (i, j) in enumerate(zip(bounds[:-1], bounds[1:])):
            a0, a1 = firsts[0][k], firsts[0][k + 1]

            part = Structures(
                lattice=self.lattice[i:j],
                x=self.x[a0:a1],
                z=self.z[a0:a1],
                num_atoms=self.num_atoms[i:j],
                batch=self.batch[a0:a1] - i,
                ptr=self.ptr[i : j + 1] - a0,
                total_energy=self.total_energy[i:j],
                energy_above_hull=self.energy_above_hull[i:j],
                band_gap=self.band_gap[i:j],
                formation_energy_per_atom=self.formation_energy_per_atom[i:j],
                pbc=self.pbc[i:j],
                properties={
                    name: values[i:j] for name, values in self.properties.items()
                },
                _validate=True,
            )

            if self.graph is None:
                result.append(part)
                continue

            e0, e1 = firsts[1][k], firsts[1][k + 1]
            attached: dict[str, Any] = {
                "graph": _graph_part(self.graph, part, (i, j), a0, (e0, e1))
            }
            for row, name in enumerate(names, start=2):
                attached[name] = _tuples_part(
                    getattr(self, name),
                    (i, j),
                    e0,
                    (firsts[row][k], firsts[row][k + 1]),
                )

            result.append(replace(part, **attached))

        return result


def _graph_part(
    graph: Graph,
    structures: Structures,
    span: tuple[int, int],
    first_atom: int,
    edges: tuple[int, int],
) -> Graph:
    """Graph of consecutive structures of a batch, see Structures.split.

    Args:
        graph: Graph of the batch.
        structures: The structures of the part.
        span: First structure of the part and the one after its last.
        first_atom: First atom of the part.
        edges: First edge of the part and the one after its last.
    """
    (i, j), (e0, e1) = span, edges

    return type(graph)(
        edge_index=graph.edge_index[:, e0:e1] - first_atom,
        edge_shift=graph.edge_shift[e0:e1],
        num_atoms=graph.num_atoms[i:j],
        num_edges=graph.num_edges[i:j],
        edge_batch=graph.edge_batch[e0:e1] - i,
        edge_ptr=graph.edge_ptr[i : j + 1] - e0,
        structures=structures,
        edge_vector=None if graph.edge_vector is None else graph.edge_vector[e0:e1],
        edge_distance=(
            None if graph.edge_distance is None else graph.edge_distance[e0:e1]
        ),
        _validate=True,
    )


def _tuples_part(
    tuples: EdgeTuples, span: tuple[int, int], first_edge: int, items: tuple[int, int]
) -> EdgeTuples:
    """Tuples of edges of consecutive structures of a batch, see Structures.split.

    Args:
        tuples: Tuples of the batch.
        span: First structure of the part and the one after its last.
        first_edge: First edge of the part.
        items: First tuple of the part and the one after its last.
    """
    (i, j), (t0, t1) = span, items

    return type(tuples)(
        index=tuples.index[:, t0:t1] - first_edge,
        num_edges=tuples.num_edges[i:j],
        num_tuples=tuples.num_tuples[i:j],
        batch=tuples.batch[t0:t1] - i,
        ptr=tuples.ptr[i : j + 1] - t0,
        angle=None if tuples.angle is None else tuples.angle[t0:t1],
        _validate=True,
    )


def _collate_structures(batch: list[Structures], *, collate_fn_map=None) -> Structures:
    """Collate of Structures for the default collate of torch."""
    del collate_fn_map

    return Structures.collate(batch)


# a DataLoader batches Structures without a collate function
default_collate_fn_map[Structures] = _collate_structures
