import dataclasses
import os
from collections.abc import Iterable
from typing import Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batching import Structures


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file: str | os.PathLike, transform=None):
        self.data_file = h5py.File(hdf5_file, "r")
        self.transform = transform

    def __len__(self) -> int:
        return self.data_file["num_atoms"].shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
        elif isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, torch.Tensor):
            indices = idx.type(torch.int64).flatten().numpy()
        else:
            indices = np.array(idx, dtype=np.int64)

        # Load structure data from indices
        lattice = torch.from_numpy(self.data_file["lattice"][indices])
        num_atoms = torch.from_numpy(self.data_file["num_atoms"][indices])
        ptr = torch.from_numpy(self.data_file["ptr"][indices])
        total_energy = torch.from_numpy(self.data_file["total_energy"][indices])
        energy_above_hull = torch.from_numpy(
            self.data_file["energy_above_hull"][indices]
        )
        band_gap = torch.from_numpy(self.data_file["band_gap"][indices])

        # Computing indices of atoms in the dataset
        batch = torch.arange(num_atoms.shape[0], dtype=torch.int64).repeat_interleave(
            num_atoms
        )
        atoms_idx = torch.arange(batch.shape[0], dtype=torch.int64)
        atoms_ptr = F.pad(num_atoms.cumsum(0), (1, 0), value=0)
        atoms_idx += ptr[batch] - atoms_ptr[batch]

        # Load atoms from atom's indices
        x = torch.from_numpy(self.data_file["x"][atoms_idx])
        z = torch.from_numpy(self.data_file["z"][atoms_idx])

        # Build validated structures
        samples = Structures(
            lattice=lattice,
            x=x,
            z=z,
            num_atoms=num_atoms,
            batch=batch,
            ptr=atoms_ptr,
            total_energy=total_energy,
            energy_above_hull=energy_above_hull,
            band_gap=band_gap,
            _validate=True,
        )

        if self.transform is not None:
            samples = self.transform(samples)

        return samples

    @classmethod
    def get_tensor_type(cls, annotated_type: str) -> Union[torch.dtype, None]:
        types = {"LongTensor": np.int64, "FloatTensor": np.float32}

        for annot, dtype in types.items():
            if annot in annotated_type:
                return dtype

        return None

    @classmethod
    def from_structures(
        cls, hdf5_file: str | os.PathLike, structures: Structures | Iterable[Structures]
    ):
        if isinstance(structures, Structures):
            iterator = iter((structures,))
        else:
            iterator = iter(structures)

        first = next(iterator)

        data_file = h5py.File(hdf5_file, "w")

        datasets = {}
        for field in dataclasses.fields(Structures):
            if field.name.startswith("_"):
                continue

            dtype = cls.get_tensor_type(field.type)

            if dtype is None:
                continue

            tensor = getattr(first, field.name).cpu()

            dataset = data_file.create_dataset(
                field.name,
                tensor.shape,
                maxshape=(None, *tensor.shape[1:]),
                dtype=dtype,
            )
            dataset[...] = getattr(first, field.name)

            offset = tensor.shape[0]
            datasets[field.name] = (dataset, offset)

        for struct in iterator:
            for name, (dataset, offset) in datasets.items():
                tensor = getattr(struct, name).cpu()

                new_size = offset + tensor.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[offset:new_size, ...] = tensor

                datasets[name] = (dataset, new_size)

        data_file.close()
