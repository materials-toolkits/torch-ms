"""Dataset of structures held in memory, on the CPU or on a GPU."""

import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batching import Structures
from .hdf5 import PROPERTIES_GROUP, HDF5Dataset


class InMemoryDataset(Dataset):
    """Structures held in memory as one batch, on the CPU or on a GPU.

    A batch is a selection of tensors already in memory: nothing is read from
    a file, and no worker is needed. On a GPU the batches are built there, and
    the transform runs there too.

    The datasets of crystals published are small enough: LeMat-Bulk, the
    largest one, is 1.6 GB for 5 million structures.

    A DataLoader batches it without sampler nor collate function, with
    num_workers=0. batches gives the same batches without the DataLoader,
    which is faster for large batches.

    Args:
        structures: All the structures of the dataset, as one batch without
            graph. They are used as they are, on their device.
        transform: Optional transform applied to each batch of structures.

    Examples:
        >>> structures = [Structures.diamond(), Structures.zincblende()] * 2
        >>> batch = Structures.collate(structures)
        >>> dataset = InMemoryDataset(batch)
        >>> len(dataset), dataset[[3, 0]].num_atoms
        (4, tensor([8, 8]))
        >>> [len(structures.num_atoms) for structures in dataset.batches(3)]
        [3, 1]
    """

    def __init__(
        self,
        structures: Structures,
        transform: Callable[[Structures], Structures] | None = None,
    ):
        if structures.without_graphs() is not structures:
            raise ValueError("the structures of a dataset have no graph")

        self.structures = structures
        self.transform = transform

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file: str | os.PathLike | HDF5Dataset,
        device: torch.device | str = "cpu",
        transform: Callable[[Structures], Structures] | None = None,
    ) -> "InMemoryDataset":
        """Read a whole HDF5 file in memory.

        Args:
            hdf5_file: Path of the HDF5 file, or the dataset of this file.
            device: Device of the structures, "cuda" for instance.
            transform: Optional transform applied to each batch of structures.
        """
        if isinstance(hdf5_file, HDF5Dataset):
            hdf5_file = hdf5_file.hdf5_file
        dataset = HDF5Dataset(hdf5_file)

        # each array of the file is read whole, which is much faster than
        # reading the structures by their indices
        def read(name: str) -> torch.Tensor:
            return torch.from_numpy(dataset.dataset(name)[...])

        num_atoms = read("num_atoms")
        ptr = F.pad(torch.cumsum(num_atoms, dim=0), (1, 0), value=0)

        if not torch.equal(read("ptr"), ptr[:-1]):
            # atoms which are not stored in the order of the structures
            return cls(dataset[:].to(device), transform=transform)

        structures = Structures(
            lattice=read("lattice"),
            x=read("x"),
            z=read("z"),
            num_atoms=num_atoms,
            ptr=ptr,
            total_energy=read("total_energy"),
            energy_above_hull=read("energy_above_hull"),
            band_gap=read("band_gap"),
            formation_energy_per_atom=read("formation_energy_per_atom"),
            pbc=read("pbc") if dataset.has_pbc() else None,
            properties={
                name: read(f"{PROPERTIES_GROUP}/{name}")
                for name in dataset.property_names()
            },
        )

        return cls(structures.to(device), transform=transform)

    @classmethod
    def concat(
        cls,
        datasets: Sequence["InMemoryDataset"],
        transform: Callable[[Structures], Structures] | None = None,
    ) -> "InMemoryDataset":
        """One dataset with the structures of several, one after the other.

        The datasets can hold crystals, molecules, or both: shuffled batches
        of the result mix them. It has every property of the datasets, NaN for
        the structures of a dataset without this property.

        Args:
            datasets: Datasets on the same device. Their transforms are not
                kept.
            transform: Optional transform applied to each batch of structures.

        Raises:
            ValueError: If there is no dataset.

        Examples:
            >>> crystals = InMemoryDataset(Structures.collate([Structures.diamond()] * 3))
            >>> molecules = InMemoryDataset(Structures.collate([Structures.water()] * 2))
            >>> dataset = InMemoryDataset.concat([crystals, molecules])
            >>> len(dataset), dataset[[4, 0]].pbc.all(dim=1)
            (5, tensor([False,  True]))
        """
        if len(datasets) == 0:
            raise ValueError("there must be at least one dataset")

        structures = Structures.collate([dataset.structures for dataset in datasets])

        return cls(structures, transform=transform)

    def device(self) -> torch.device:
        """Device of the structures, and of the batches."""
        return self.structures.device()

    def to(self, device: torch.device | str) -> "InMemoryDataset":
        """The dataset with its structures on a device."""
        return InMemoryDataset(self.structures.to(device), transform=self.transform)

    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        return len(self.structures.num_atoms)

    def __getitem__(self, idx: Any) -> Structures:
        """Get structures by index.

        Supports int, slice, tensor, or iterable of indices, in any order. The
        structures are returned as one batch, in the order of the indices.
        """
        device = self.device()

        if isinstance(idx, slice):
            index = torch.arange(*idx.indices(len(self)), device=device)
        elif isinstance(idx, torch.Tensor):
            index = idx.to(torch.int64).flatten()
        else:
            index = torch.as_tensor(idx, dtype=torch.int64, device=device).flatten()

        # indices from the end, as for a list
        index = torch.where(index < 0, index + len(self), index)

        samples = self.structures.select(index)

        if self.transform is not None:
            samples = self.transform(samples)

        return samples

    def __getitems__(self, indices: list[int]) -> list[Structures]:
        """Get the structures of a batch of a DataLoader, selected in one call."""
        return [self[indices]]

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> Iterator[Structures]:
        """Batches of one epoch, without a DataLoader.

        Args:
            batch_size: Number of structures per batch.
            shuffle: If True, the structures are drawn in a random order.
            drop_last: If True, a last batch smaller than batch_size is not given.
            generator: Random number generator of the order, on any device.

        Yields:
            Batches of structures, on the device of the dataset.
        """
        device = self.device()

        if shuffle:
            # drawn on the device of the generator, which can be another one
            drawn_on = device if generator is None else generator.device
            order = torch.randperm(len(self), device=drawn_on, generator=generator)
            order = order.to(device)
        else:
            order = torch.arange(len(self), device=device)

        for index in order.split(batch_size):
            if drop_last and len(index) < batch_size:
                return

            yield self[index]
