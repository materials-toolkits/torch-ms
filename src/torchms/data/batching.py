"""Batching utilities for crystal structures."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

TensorLike = torch.Tensor | np.ndarray | Sequence[Any]


class InvalidStructure(Exception):
    """Exception raised when a structure has invalid data."""


@dataclass(init=False)
class Structures:
    """Container for batched crystal structures and their properties.

    Attributes:
        lattice: Lattice vectors with shape (n_struct, 3, 3).
        x: Atomic coordinates normalized to [0, 1) with shape (n_atoms, 3).
        z: Atomic numbers with shape (n_atoms,).
        num_atoms: Number of atoms per structure with shape (n_struct,).
        batch: Batch index for each atom with shape (n_atoms,).
        ptr: Index pointers for each structure with shape (n_struct + 1,).
        total_energy: Total energy per structure in eV with shape (n_struct,).
        energy_above_hull: Energy above hull per structure in eV with shape (n_struct,).
        band_gap: Band gap per structure in eV with shape (n_struct,).
        formation_energy_per_atom: Formation energy per structure in eV/atom
            with shape (n_struct,).

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

        counts = self.num_atoms.flatten().long().clamp(min=0)

        if batch is None:
            self.batch = torch.arange(
                len(counts), dtype=torch.int64, device=counts.device
            ).repeat_interleave(counts)
        else:
            self.batch = self.as_tensor("batch", batch, torch.int64)

        if ptr is None:
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
            raise InvalidStructure("must contain at least one crystal")

        if n_atoms < 1:
            raise InvalidStructure("must contain at least one atom per crystal")

        self.to_tensor(
            "lattice", dtype=torch.float32, shape=(n_struct, 3, 3), device=device
        )
        self.to_tensor("x", dtype=torch.float32, shape=(n_atoms, 3), device=device)
        self.to_tensor("z", dtype=torch.int64, shape=(n_atoms,), device=device)
        self.to_tensor("num_atoms", dtype=torch.int64, shape=(n_struct,), device=device)

        if (self.num_atoms <= 0).any():
            raise InvalidStructure("must contain at least one atom per crystal")

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
            Structures on the device. Tensors already on it are shared with the
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

        return replace(self, **tensors)

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
    def collate(cls, structures: list[Structures]) -> Structures:
        """Combine multiple structures into a single batched structure.

        Args:
            structures: List of Structures to batch together, each one with one
                or several structures.

        Returns:
            A single Structures object with concatenated tensors.

        Raises:
            ValueError: If structures list is empty.

        Examples:
            >>> struct = Structures.diamond()
            >>> batched = Structures.collate([struct, struct])
            >>> batched.lattice.shape
            torch.Size([2, 3, 3])
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

        return cls(
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
            _validate=True,
        )

    @classmethod
    def separate(cls, structures: Structures) -> list[Structures]:
        """Split a batched structure into individual structures.

        Args:
            structures: A batched Structures object.

        Returns:
            List of individual Structures objects.

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
                _validate=True,
            )
            result.append(struct)

        return result
