"""Batching utilities for crystal structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


class InvalidStructure(Exception):
    """Exception raised when a structure has invalid data."""


@dataclass
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

    Example:
        Creating a Diamond crystal structure (C, a=3.567 A, 8 atoms):

        >>> a = 3.567
        >>> lattice = torch.tensor(
        ...     [[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]],
        ...     dtype=torch.float32,
        ... )
        >>> positions = [
        ...     [0.0, 0.0, 0.0],
        ...     [0.5, 0.5, 0.0],
        ...     [0.5, 0.0, 0.5],
        ...     [0.0, 0.5, 0.5],
        ...     [0.25, 0.25, 0.25],
        ...     [0.75, 0.75, 0.25],
        ...     [0.75, 0.25, 0.75],
        ...     [0.25, 0.75, 0.75],
        ... ]
        >>> x = torch.tensor(positions, dtype=torch.float32) * a
        >>> z = torch.tensor([6] * 8, dtype=torch.int64)
        >>> struct = Structures(lattice=lattice, x=x, z=z)
        >>> struct.num_atoms
        tensor([8])
    """

    lattice: torch.FloatTensor
    x: torch.FloatTensor
    z: torch.LongTensor
    num_atoms: torch.LongTensor | None = None
    batch: torch.LongTensor | None = None
    ptr: torch.LongTensor | None = None
    total_energy: torch.FloatTensor | None = None
    energy_above_hull: torch.FloatTensor | None = None
    band_gap: torch.FloatTensor | None = None
    _validate: bool = False

    def __post_init__(self):
        if self._validate:
            return

        self.check_tensors()

        self._validate = True

    def check_tensors(self):
        """Validate structure data and convert fields to proper tensors.

        Raises:
            InvalidStructure: If lattice or atom data is invalid.
        """
        device = torch.device("cpu")
        if isinstance(self.lattice, torch.Tensor):
            device = self.lattice.device

        n_struct = len(self.lattice)
        n_atoms = len(self.z)

        if n_struct < 1:
            raise InvalidStructure("must contain at least one crystal")

        if n_atoms < 1:
            raise InvalidStructure("must contain at least one atom per crystal")

        if n_struct > 1 and not hasattr(self, "num_atoms"):
            raise InvalidStructure(
                'must specify "num_atoms" when multiple structures are given'
            )

        self.to_tensor(
            "lattice",
            dtype=torch.float32,
            shape=(n_struct, 3, 3),
            device=device,
        )
        self.to_tensor(
            "x",
            dtype=torch.float32,
            shape=(n_atoms, 3),
            device=device,
        )
        self.to_tensor(
            "z",
            dtype=torch.int64,
            shape=(n_atoms,),
            device=device,
        )
        self.to_tensor(
            "num_atoms",
            dtype=torch.int64,
            shape=(n_struct,),
            device=device,
            default=lambda struct: torch.tensor(
                [n_atoms], dtype=torch.int64, device=device
            ),
        )

        if (self.num_atoms <= 0).any():
            raise InvalidStructure("must contain at least one atom per crystal")

        self.to_tensor(
            "batch",
            dtype=torch.int64,
            shape=(n_atoms,),
            device=device,
            default=lambda struct: torch.arange(
                0, n_struct, dtype=torch.int32, device=device
            ).repeat_interleave(self.num_atoms, dim=0),
        )
        self.to_tensor(
            "ptr",
            dtype=torch.int64,
            shape=(n_struct + 1,),
            device=device,
            default=lambda struct: F.pad(
                torch.cumsum(struct.num_atoms, dim=0), (1, 0), value=0
            ),
        )
        self.to_tensor(
            "total_energy",
            dtype=torch.float32,
            shape=(n_struct,),
            device=device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )
        self.to_tensor(
            "energy_above_hull",
            dtype=torch.float32,
            shape=(n_struct,),
            device=device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )
        self.to_tensor(
            "band_gap",
            dtype=torch.float32,
            shape=(n_struct,),
            device=device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )

    def device(self) -> torch.device:
        """Get the device of the lattice tensor.

        Returns:
            torch.device: Device on which the structure tensors are placed.
        """
        return self.lattice.device

    def to_tensor(
        self,
        field: str,
        *,
        dtype: torch.Tensor.type,
        shape: tuple[int, ...],
        device: torch.device,
        default: Callable[["Structures"], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Convert a field to a tensor with specified properties.

        Args:
            field: Name of the attribute to convert.
            dtype: Target tensor dtype.
            shape: Expected tensor shape.
            device: Target tensor device.
            default: Optional factory function to create default values.

        Returns:
            The converted tensor.

        Raises:
            InvalidStructure: If field value is invalid or incompatible.
        """
        values = getattr(self, field)

        if values is None:
            if default is None:
                raise InvalidStructure(f'no default value for "{field}"')
            tensor = default(self)
        elif not isinstance(values, torch.Tensor):
            try:
                tensor = torch.tensor(values, dtype=dtype)
            except TypeError as e:
                raise InvalidStructure(f"invalid field {field}: {e}") from e
        else:
            tensor = values

        if tensor.dtype != dtype:
            tensor = tensor.type(dtype)

        if tensor.shape != shape:
            raise InvalidStructure(
                f'invalid shape {tensor.shape} for field "{field}", must be {shape}'
            )

        if tensor.device != device:
            raise InvalidStructure(
                f'inconsistend device for field "{field}", {device}!={tensor.device}'
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
    def collate(cls, structures: list[Structures]) -> Structures:
        """Combine multiple structures into a single batched structure.

        Args:
            structures: List of Structures to batch together.

        Returns:
            A single Structures object with concatenated tensors.

        Raises:
            ValueError: If structures list is empty.

        Example:
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

        batch = torch.arange(
            len(structures), dtype=torch.int64, device=device
        ).repeat_interleave(num_atoms)

        ptr = F.pad(torch.cumsum(num_atoms, dim=0), (1, 0), value=0)

        total_energy = torch.cat([s.total_energy for s in structures])
        energy_above_hull = torch.cat([s.energy_above_hull for s in structures])
        band_gap = torch.cat([s.band_gap for s in structures])

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
            _validate=True,
        )

    @classmethod
    def separate(cls, structures: Structures) -> list[Structures]:
        """Split a batched structure into individual structures.

        Args:
            structures: A batched Structures object.

        Returns:
            List of individual Structures objects.

        Example:
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
                batch=structures.batch[start:end],
                ptr=torch.tensor(
                    [0, structures.ptr[i + 1] - structures.ptr[i]],
                    device=structures.device(),
                ),
                total_energy=structures.total_energy[i : i + 1],
                energy_above_hull=structures.energy_above_hull[i : i + 1],
                band_gap=structures.band_gap[i : i + 1],
                _validate=True,
            )
            result.append(struct)

        return result
