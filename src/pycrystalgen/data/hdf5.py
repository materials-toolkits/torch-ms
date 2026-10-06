"""HDF5 dataset implementation for loading and saving crystal structures."""

import dataclasses
import os
from collections.abc import Iterable
from typing import Union
import logging

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batching import Structures

logger = logging.getLogger(__name__)


class HDF5Dataset(Dataset):
    """HDF5 dataset for loading crystal structures from HDF5 files."""

    def __init__(self, hdf5_file: str | os.PathLike, transform=None):
        self.data_file = h5py.File(hdf5_file, "r")
        self.transform = transform

    def dataset(self, name: str) -> h5py.Dataset:
        """Get a dataset of the file by name.

        Raises:
            KeyError: If the file has no dataset with this name.
        """
        dataset = self.data_file[name]

        if not isinstance(dataset, h5py.Dataset):
            raise KeyError(f'"{name}" is not a dataset of {self.data_file.filename}')

        return dataset

    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        return self.dataset("num_atoms").shape[0]

    def __getitem__(self, idx):
        """Get structures by index.

        Supports int, slice, tensor, or iterable of indices.
        Should be in ascending order.
        """
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
        elif isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, torch.Tensor):
            indices = idx.type(torch.int64).flatten().numpy()
        else:
            indices = np.array(idx, dtype=np.int64)

        # Load structure data from indices
        lattice = torch.from_numpy(self.dataset("lattice")[indices])
        num_atoms = torch.from_numpy(self.dataset("num_atoms")[indices])
        ptr = torch.from_numpy(self.dataset("ptr")[indices])
        total_energy = torch.from_numpy(self.dataset("total_energy")[indices])
        energy_above_hull = torch.from_numpy(self.dataset("energy_above_hull")[indices])
        band_gap = torch.from_numpy(self.dataset("band_gap")[indices])

        # Computing indices of atoms in the dataset
        batch = torch.arange(num_atoms.shape[0], dtype=torch.int64).repeat_interleave(
            num_atoms
        )
        atoms_idx = torch.arange(batch.shape[0], dtype=torch.int64)
        atoms_ptr = F.pad(num_atoms.cumsum(0), (1, 0), value=0)
        atoms_idx += ptr[batch] - atoms_ptr[batch]

        # Load atoms from atom's indices
        x = torch.from_numpy(self.dataset("x")[atoms_idx.numpy()])
        z = torch.from_numpy(self.dataset("z")[atoms_idx.numpy()])

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
    def from_hub(cls, repo_id: str, filename: str, transform=None, **kwargs):
        """Load a preprocessed dataset from a Hugging Face dataset repository.

        Requires huggingface_hub (pip install pycrystalgen[hub]).

        Args:
            repo_id: Repository name, e.g. "user/dataset".
            filename: Path of the HDF5 file inside the repository.
            transform: Optional transform to apply to structures.
            **kwargs: Forwarded to huggingface_hub.hf_hub_download
                (revision, token, cache_dir, ...).
        """
        from huggingface_hub import (  # pylint: disable=import-outside-toplevel
            hf_hub_download,
        )

        hdf5_file = hf_hub_download(repo_id, filename, repo_type="dataset", **kwargs)

        return cls(hdf5_file, transform=transform)

    @classmethod
    def get_tensor_type(cls, annotated_type: str) -> Union[torch.dtype, None]:
        """Get torch dtype from annotated type string."""
        types = {
            "LongTensor": np.int64,
            "FloatTensor": np.float32,
            "Int[": np.int64,
            "Float[": np.float32,
        }

        for annot, dtype in types.items():
            if annot in annotated_type:
                return dtype

        return None

    @classmethod
    def _field_tensor(
        cls, structures: Structures, name: str, atoms_offset: int
    ) -> torch.Tensor:
        """Get a field as stored on disk.

        "ptr" is stored as the global index of the first atom of each structure,
        so batches can be appended one after the other.
        """
        tensor = getattr(structures, name).cpu()

        if name == "ptr":
            tensor = tensor[:-1] + atoms_offset

        return tensor

    @classmethod
    def from_structures(
        cls, hdf5_file: str | os.PathLike, structures: Structures | Iterable[Structures]
    ):
        """Save structures to HDF5 file."""
        if isinstance(structures, Structures):
            iterator = iter((structures,))
        else:
            iterator = iter(structures)

        try:
            first = next(iterator)
        except StopIteration:
            logger.warning(f'Empty iterator when building "{hdf5_file}"')
            return

        data_file = h5py.File(hdf5_file, "w")

        datasets = {}
        for field in dataclasses.fields(Structures):
            if field.name.startswith("_"):
                continue

            dtype = cls.get_tensor_type(str(field.type))

            if dtype is None:
                continue

            tensor = cls._field_tensor(first, field.name, 0)

            dataset = data_file.create_dataset(
                field.name,
                tensor.shape,
                maxshape=(None, *tensor.shape[1:]),
                dtype=dtype,
            )
            dataset[...] = tensor

            offset = tensor.shape[0]
            datasets[field.name] = (dataset, offset)

        atoms_offset = int(first.num_atoms.sum())

        for struct in iterator:
            for name, (dataset, offset) in datasets.items():
                tensor = cls._field_tensor(struct, name, atoms_offset)

                new_size = offset + tensor.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[offset:new_size, ...] = tensor

                datasets[name] = (dataset, new_size)

            atoms_offset += int(struct.num_atoms.sum())

        data_file.close()
