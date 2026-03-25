from __future__ import annotations
from dataclasses import dataclass

import torch
import numpy as np
import numpy.typing as npt


class InvalidStructure(Exception):
    pass


@dataclass
class Structures:
    lattice: torch.FloatTensor  # lattice shape (shape: n_{struct}x3x3)
    x: (
        torch.FloatTensor
    )  # atomic cooridanate inside ov the lattice, clipped between 0 and 1 (shape: n_{atoms}x3)
    z: torch.LongTensor  # atomic number (shape: n_{atoms})
    num_atoms: torch.LongTensor | None = None  # number of atoms (shape: n_{struct})
    batch: torch.LongTensor | None = None  # batch index of the atoms (shape: n_{atoms})
    ptr: torch.LongTensor | None = (
        None  # index of atoms from structue i (shape: n_{struct}+1)
    )
    total_energy: torch.FloatTensor | None = (
        None  # total energy of a structure in eV/atoms (shape: n_{struct})
    )
    energy_above_hull: torch.FloatTensor | None = (
        None  # energy above the hull of a structure in eV/atoms (shape: n_{struct})
    )
    band_gap: torch.FloatTensor | None = (
        None  # band gap of a structure in eV (shape: n_{struct})
    )
    _validate: bool = (
        False  # is data already validate or should we check and fill default value?
    )

    def __post_init__(self):
        if self._validate:
            return

        self.lattice = self.to_tensor(self.lattice, torch.float32)
        self.x = self.to_tensor(self.x, torch.float32)
        self.z = self.to_tensor(self.z, torch.int64)

        if self.num_atoms is None:
            self.num_atoms = torch.tensor(
                [self.z.shape[0]], dtype=torch.int64, device=self.z.device
            )
        else:
            self.num_atoms = self.to_tensor(self.num_atoms, torch.int64)

        if self.batch is None:
            self.batch = torch.zeros_like(self.z)
        else:
            self.batch = self.to_tensor(self.batch, torch.int64)

        if self.ptr is None:
            self.ptr = torch.tensor(
                [0, self.z.shape[0]], dtype=torch.int64, device=self.z.device
            )
        else:
            self.ptr = self.to_tensor(self.ptr, torch.int64)

        if self.total_energy is None:
            self.total_energy = torch.full_like(
                self.num_atoms, fill_value=torch.nan, dtype=torch.float32
            )
        else:
            self.total_energy = self.to_tensor(self.total_energy, torch.float32)

        if self.energy_above_hull is None:
            self.energy_above_hull = torch.full_like(
                self.total_energy, fill_value=torch.nan
            )
        else:
            self.energy_above_hull = self.to_tensor(
                self.energy_above_hull, torch.float32
            )

        if self.band_gap is None:
            self.band_gap = torch.full_like(self.total_energy, fill_value=torch.nan)
        else:
            self.band_gap = self.to_tensor(self.band_gap, torch.float32)

        self._validate = True

    @classmethod
    def to_tensor(
        cls, values: torch.Tensor | npt.ArrayLike, dtype: torch.Tensor.type
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            try:
                return torch.tensor(values, dtype=dtype)
            except TypeError as e:
                raise InvalidStructure(e)

        if values.dtype != dtype:
            return values.type_as(dtype)

        return values

    @classmethod
    def collate(cls, strctures: list[Structures]) -> Structures:
        pass

    @classmethod
    def separate(cls, strctures: Structures) -> list[Structures]:
        pass
