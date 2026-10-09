"""HDF5 dataset implementation for loading and saving structures."""

import dataclasses
import os
from collections.abc import Callable, Iterable
from typing import Any, Union
import logging

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batching import Structures

logger = logging.getLogger(__name__)

# group of the file holding the properties of Structures.properties
PROPERTIES_GROUP = "properties"


class HDF5Dataset(Dataset):
    """HDF5 dataset for loading structures, crystals or molecules, from HDF5 files.

    The file is opened once in each process which reads the dataset: the
    dataset can be given to a DataLoader with several workers, and pickled.

    A DataLoader batches it without sampler nor collate function. The
    structures of a batch are read in one call and the transform is applied
    once, to the whole batch.

    Args:
        hdf5_file: Path of the HDF5 file.
        transform: Optional transform applied to each batch of structures read:
            augmentations, graph construction (torchms.graph.transforms).

    Examples:
        >>> import tempfile
        >>> from torch.utils.data import DataLoader
        >>> batch = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     file = os.path.join(directory, "structures.hdf5")
        ...     HDF5Dataset.from_structures(file, [batch, batch])
        ...     loader = DataLoader(HDF5Dataset(file), batch_size=3, shuffle=True)
        ...     [len(structures.num_atoms) for structures in loader]
        [3, 1]
    """

    def __init__(
        self,
        hdf5_file: str | os.PathLike,
        transform: Callable[[Structures], Structures] | None = None,
    ):
        self.hdf5_file = os.fspath(hdf5_file)
        self.transform = transform

        self._data_file: h5py.File | None = None
        self._pid: int | None = None

        # opened now, to fail here if the file cannot be read
        _ = self.data_file

    @property
    def data_file(self) -> h5py.File:
        """The HDF5 file, opened in the current process.

        A file opened in another process, before a fork, is not used: an open
        HDF5 file must not be shared between processes.
        """
        if self._data_file is None or self._pid != os.getpid():
            self._data_file = h5py.File(self.hdf5_file, "r")
            self._pid = os.getpid()

        return self._data_file

    def __getstate__(self) -> dict[str, Any]:
        """State to pickle: an open file cannot be, it is opened again."""
        state = self.__dict__.copy()
        state["_data_file"] = None
        state["_pid"] = None
        return state

    def dataset(self, name: str) -> h5py.Dataset:
        """Get a dataset of the file by name.

        Raises:
            KeyError: If the file has no dataset with this name.
        """
        dataset = self.data_file[name]

        if not isinstance(dataset, h5py.Dataset):
            raise KeyError(f'"{name}" is not a dataset of {self.hdf5_file}')

        return dataset

    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        return self.dataset("num_atoms").shape[0]

    def has_pbc(self) -> bool:
        """Whether the file says which structures are periodic.

        A file without it holds crystals only: every structure is periodic.
        """
        return "pbc" in self.data_file

    def property_names(self) -> list[str]:
        """Names of the properties of the file read in Structures.properties."""
        group = self.data_file.get(PROPERTIES_GROUP)

        if not isinstance(group, h5py.Group):
            return []

        return list(group.keys())

    def read(self, name: str, indices: np.ndarray) -> torch.Tensor:
        """Read rows of a dataset of the file, in any order.

        HDF5 reads rows in ascending order without duplicates: other indices
        are read sorted, then put back in the order asked.

        Args:
            name: Name of the dataset.
            indices: Indices of the rows, shape (n,). In any order, the same
                row can be asked several times.

        Returns:
            Rows in the order of indices, shape (n, ...).
        """
        dataset = self.dataset(name)

        if len(indices) < 2 or bool((indices[1:] > indices[:-1]).all()):
            return torch.from_numpy(dataset[indices])

        unique, inverse = np.unique(indices, return_inverse=True)

        return torch.from_numpy(dataset[unique][inverse])

    def __getitem__(self, idx):
        """Get structures by index.

        Supports int, slice, tensor, or iterable of indices, in any order. The
        structures are returned as one batch, in the order of the indices.
        """
        if isinstance(idx, slice):
            indices = np.arange(*idx.indices(len(self)), dtype=np.int64)
        elif isinstance(idx, torch.Tensor):
            indices = idx.type(torch.int64).flatten().numpy()
        else:
            indices = np.array(idx, dtype=np.int64).reshape(-1)

        # indices from the end, as for a list
        indices = np.where(indices < 0, indices + len(self), indices)

        # Load structure data from indices
        lattice = self.read("lattice", indices)
        num_atoms = self.read("num_atoms", indices)
        ptr = self.read("ptr", indices)
        total_energy = self.read("total_energy", indices)
        energy_above_hull = self.read("energy_above_hull", indices)
        band_gap = self.read("band_gap", indices)
        formation_energy_per_atom = self.read("formation_energy_per_atom", indices)
        pbc = self.read("pbc", indices) if self.has_pbc() else None
        properties = {
            name: self.read(f"{PROPERTIES_GROUP}/{name}", indices)
            for name in self.property_names()
        }

        # Computing indices of atoms in the dataset
        batch = torch.arange(num_atoms.shape[0], dtype=torch.int64).repeat_interleave(
            num_atoms
        )
        atoms_idx = torch.arange(batch.shape[0], dtype=torch.int64)
        atoms_ptr = F.pad(num_atoms.cumsum(0), (1, 0), value=0)
        atoms_idx += ptr[batch] - atoms_ptr[batch]

        # Load atoms from atom's indices
        x = self.read("x", atoms_idx.numpy())
        z = self.read("z", atoms_idx.numpy())

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
            formation_energy_per_atom=formation_energy_per_atom,
            pbc=pbc,
            properties=properties,
            _validate=True,
        )

        if self.transform is not None:
            samples = self.transform(samples)

        return samples

    def __getitems__(self, indices: list[int]) -> list[Structures]:
        """Get the structures of a batch of a DataLoader.

        Called by a DataLoader with a batch size instead of __getitem__ on each
        index: the structures are read in one call and transformed once.

        Args:
            indices: Indices of the structures of the batch, in any order.

        Returns:
            List of one batch of structures, for the collate function.
        """
        return [self[indices]]

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        filename: str,
        transform: Callable[[Structures], Structures] | None = None,
        **kwargs: Any,
    ) -> "HDF5Dataset":
        """Load a preprocessed dataset from a Hugging Face dataset repository.

        Requires huggingface_hub (pip install torchms[hub]).

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
            "Bool[": np.bool_,
            "Int[": np.int64,
            "Float[": np.float32,
        }

        # a dictionary of tensors is not a tensor
        if annotated_type.startswith("dict["):
            return None

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
        if name.startswith(f"{PROPERTIES_GROUP}/"):
            name = name.split("/", maxsplit=1)[1]
            # pylint: disable-next=protected-access
            return structures._properties_of([name])[name].cpu()

        tensor = getattr(structures, name).cpu()

        if name == "ptr":
            tensor = tensor[:-1] + atoms_offset

        return tensor

    @classmethod
    def from_structures(
        cls, hdf5_file: str | os.PathLike, structures: Structures | Iterable[Structures]
    ):
        """Save structures to HDF5 file.

        The properties of Structures.properties saved are the ones of the
        first structures. They are NaN for the next ones without them.

        Raises:
            ValueError: If structures have a property the first ones do not.
        """
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

        names = list(first.properties)
        for name in names:
            tensor = first.properties[name].cpu()
            dataset = data_file.create_dataset(
                f"{PROPERTIES_GROUP}/{name}",
                tensor.shape,
                maxshape=(None,),
                dtype=np.float32,
            )
            dataset[...] = tensor
            datasets[f"{PROPERTIES_GROUP}/{name}"] = (dataset, tensor.shape[0])

        atoms_offset = int(first.num_atoms.sum())

        for struct in iterator:
            unknown = sorted(set(struct.properties) - set(names))
            if len(unknown) > 0:
                data_file.close()
                raise ValueError(
                    f"properties {unknown} are not properties of the first "
                    f'structures saved in "{hdf5_file}"'
                )

            for name, (dataset, offset) in datasets.items():
                tensor = cls._field_tensor(struct, name, atoms_offset)

                new_size = offset + tensor.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[offset:new_size, ...] = tensor

                datasets[name] = (dataset, new_size)

            atoms_offset += int(struct.num_atoms.sum())

        data_file.close()
