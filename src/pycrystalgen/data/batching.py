from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
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

        self.check_tensors()

        self._validate = True

    def check_tensors(self):
        device = torch.device("cpu")
        if isinstance(self.lattice, torch.Tensor):
            device = self.lattice.device

        n_struct = len(self.lattice)
        n_atoms = len(self.z)

        if n_struct > 1 and not hasattr(self,"num_atoms"):
            raise InvalidStructure("must specify \"num_atoms\" when multiple structures are given")

        self.to_tensor("lattice", torch.float32, (n_struct, 3, 3), device)
        self.to_tensor("x", torch.float32, (n_atoms, 3), device)
        self.to_tensor("z", torch.int64, (n_atoms,), device)
        self.to_tensor(
            "num_atoms",
            torch.int64,
            (n_struct,),
            device,
            default=lambda struct: torch.tensor(
                [n_atoms], dtype=torch.int64, device=device
            ),
        )
        self.to_tensor(
            "batch",
            torch.int64,
            (n_atoms,),
            device,
            default=lambda struct: torch.arange(
                0, n_struct, dtype=torch.int32, device=device
            ).repeat_interleave(self.num_atoms, dim=0),
        )
        self.to_tensor(
            "ptr",
            torch.int64,
            (n_struct + 1,),
            device,
            default=lambda struct: F.pad(torch.cumsum(struct.num_atoms,dim=0),(1,0),value=0),
        )
        self.to_tensor(
            "total_energy",
            torch.float32,
            (n_struct,),
            device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )
        self.to_tensor(
            "energy_above_hull",
            torch.float32,
            (n_struct,),
            device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )
        self.to_tensor(
            "band_gap",
            torch.float32,
            (n_struct,),
            device,
            default=lambda struct: torch.full_like(
                struct.num_atoms, fill_value=torch.nan, dtype=torch.float32
            ),
        )

    def device(self) -> torch.device:
        return self.lattice.device

    def to_tensor(
        self,
        field: str,
        dtype: torch.Tensor.type,
        shape: torch.Size,
        device: torch.device,
        default: Callable[[Self], torch.Tensor] = None,
    ) -> torch.Tensor:
        values = getattr(self, field)

        if values is None:
            if default is None:
                raise InvalidStructure(f'no default value for "{field}"')
            tensor = default(self)
        elif not isinstance(values, torch.Tensor):
            try:
                tensor = torch.tensor(values, dtype=dtype)
            except TypeError as e:
                raise InvalidStructure(f"invalid field {field}: {e}")
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
    def collate(cls, strctures: list[Structures]) -> Structures:
        pass

    @classmethod
    def separate(cls, strctures: Structures) -> list[Structures]:
        pass
