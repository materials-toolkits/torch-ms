"""Dataset of structures held in memory, on the CPU or on a GPU."""

import copy
import math
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batching import Structures
from .hdf5 import PROPERTIES_GROUP, HDF5Dataset, Indices

# when an epoch of datasets mixed in fixed shares ends
Epoch = Literal["shortest", "longest"]


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
            hdf5_file: Path of the HDF5 file, or the dataset of this file. Of
                a subset of a file, its structures only are read.
            device: Device of the structures, "cuda" for instance.
            transform: Optional transform applied to each batch of structures.
        """
        if isinstance(hdf5_file, HDF5Dataset):
            dataset = hdf5_file.with_transform(None)
        else:
            dataset = HDF5Dataset(hdf5_file)

        if dataset.indices is not None:
            # some of the structures of the file only
            return cls(dataset[:].to(device), transform=transform)

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
        moved = copy.copy(self)
        moved.structures = self.structures.to(device)

        return moved

    def with_transform(
        self, transform: Callable[[Structures], Structures] | None
    ) -> "InMemoryDataset":
        """The same dataset, sharing its structures, with another transform."""
        other = copy.copy(self)
        other.transform = transform

        return other

    def subset(self, indices: Indices) -> "InMemoryDataset":
        """Dataset of some of the structures, copied.

        Args:
            indices: Indices of the structures kept, in the order they will
                have, shape (n,).

        Returns:
            A dataset of these structures on the same device, with the same
            transform.

        Examples:
            >>> structures = [Structures.diamond(), Structures.water()] * 2
            >>> dataset = InMemoryDataset(Structures.collate(structures))
            >>> dataset.subset([3, 0]).structures.num_atoms
            tensor([3, 8])
        """
        index = torch.as_tensor(indices, dtype=torch.int64, device=self.device())
        index = index.flatten()
        index = torch.where(index < 0, index + len(self), index)

        return InMemoryDataset(self.structures.select(index), self.transform)

    def filter(
        self, condition: Callable[[Structures], torch.Tensor]
    ) -> "InMemoryDataset":
        """Dataset of the structures which pass a condition.

        Args:
            condition: Gives the mask of the structures of a batch to keep,
                shape (n_struct,): a torchms.data.filters.Filter, or any
                function. It is given the structures without the transform.

        Returns:
            A subset of the dataset, in the same order.

        Examples:
            >>> from torchms.data.filters import Filter
            >>> structures = [Structures.diamond(), Structures.water()] * 2
            >>> dataset = InMemoryDataset(Structures.collate(structures))
            >>> dataset.filter(Filter(max_atoms=4)).structures.num_atoms
            tensor([3, 3])
        """
        return self.subset(condition(self.structures).nonzero().squeeze(1))

    def epoch_size(self) -> int:
        """Number of structures given in an epoch: each one once."""
        return len(self)

    def order(
        self, shuffle: bool = False, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Indices of the structures of an epoch, in the order they are given.

        Args:
            shuffle: If True, the structures are drawn in a random order.
            generator: Random number generator of the order, on any device.

        Returns:
            Indices on the device of the dataset, shape (epoch_size,).
        """
        device = self.device()

        if not shuffle:
            return torch.arange(len(self), device=device)

        # drawn on the device of the generator, which can be another one
        drawn_on = device if generator is None else generator.device
        order = torch.randperm(len(self), device=drawn_on, generator=generator)

        return order.to(device)

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
        for index in self.order(shuffle, generator).split(batch_size):
            if drop_last and len(index) < batch_size:
                return

            yield self[index]


class MixedDataset(InMemoryDataset):
    """Several datasets in memory, given in fixed shares in every batch.

    The structures of the datasets are held as one batch, one dataset after
    the other. What makes the mix is the order of an epoch: its structures go
    by groups of sum(counts), counts[i] of them from the dataset i. Batches of
    this size, or of a multiple of it, hold the shares asked.

    In a group the structures are in the order of the datasets: the ones of
    the first dataset, then the ones of the second.

    Made by mix.

    Attributes:
        sizes: Number of structures of each dataset.
        counts: Number of structures of each dataset in a group.
        epoch: When an epoch ends, see mix.
    """

    def __init__(
        self,
        structures: Structures,
        sizes: Sequence[int],
        counts: Sequence[int],
        epoch: Epoch = "shortest",
        transform: Callable[[Structures], Structures] | None = None,
    ):
        super().__init__(structures, transform)

        self.sizes = tuple(int(size) for size in sizes)
        self.counts = tuple(int(count) for count in counts)
        self.epoch: Epoch = epoch

        if len(self.sizes) != len(self.counts):
            raise ValueError("there must be one count per dataset")

        if sum(self.sizes) != len(self):
            raise ValueError("the sizes must be the ones of the datasets mixed")

        if min(self.counts) < 1:
            raise ValueError("each dataset must have at least one structure per batch")

        if epoch not in ("shortest", "longest"):
            raise ValueError('epoch must be "shortest" or "longest"')

        if min(size // count for size, count in zip(self.sizes, self.counts)) < 1:
            raise ValueError("a dataset is too small to fill its share of a batch")

    def num_groups(self) -> int:
        """Number of groups of sum(counts) structures in an epoch."""
        pairs = list(zip(self.sizes, self.counts))

        if self.epoch == "shortest":
            return min(size // count for size, count in pairs)

        return max(math.ceil(size / count) for size, count in pairs)

    def epoch_size(self) -> int:
        """Number of structures given in an epoch."""
        return self.num_groups() * sum(self.counts)

    def subset(self, indices: Indices) -> InMemoryDataset:
        """Dataset of some of the structures, which is not mixed any more."""
        return InMemoryDataset(self.structures, self.transform).subset(indices)

    def order(
        self, shuffle: bool = False, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Indices of the structures of an epoch, by groups of sum(counts).

        Args:
            shuffle: If True, the structures of each dataset are drawn in a
                random order. Otherwise they are taken in their order.
            generator: Random number generator of the order, on any device.

        Returns:
            Indices on the device of the dataset, shape (epoch_size,).
        """
        device = self.device()
        drawn_on = device if generator is None else generator.device
        num_groups = self.num_groups()

        columns = []
        first = 0
        for size, count in zip(self.sizes, self.counts):
            needed = num_groups * count
            # a dataset smaller than what an epoch takes is gone through again
            passes = math.ceil(needed / size)

            if shuffle:
                drawn = [
                    torch.randperm(size, device=drawn_on, generator=generator)
                    for _ in range(passes)
                ]
                structures = torch.cat(drawn)[:needed]
            else:
                structures = torch.arange(size, device=drawn_on).repeat(passes)[:needed]

            columns.append((structures + first).view(num_groups, count))
            first += size

        return torch.cat(columns, dim=1).flatten().to(device)

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> Iterator[Structures]:
        """Batches of one epoch, each one with the shares of the datasets.

        Args:
            batch_size: Number of structures per batch, a multiple of
                sum(counts).
            shuffle: If True, the structures are drawn in a random order.
            drop_last: If True, a last batch smaller than batch_size is not given.
            generator: Random number generator of the order, on any device.

        Raises:
            ValueError: If batch_size is not a multiple of sum(counts).
        """
        if batch_size % sum(self.counts) != 0:
            raise ValueError(
                f"batches of {batch_size} structures cannot hold {self.counts} "
                "structures of each dataset: give a multiple of their sum"
            )

        return super().batches(batch_size, shuffle, drop_last, generator)


def mix(
    datasets: Sequence[InMemoryDataset],
    counts: Sequence[int],
    epoch: Epoch = "shortest",
    transform: Callable[[Structures], Structures] | None = None,
) -> MixedDataset:
    """Mix datasets, with a fixed number of structures of each one per batch.

    A large dataset then does not drown a small one: every batch of
    sum(counts) structures holds counts[i] structures of the dataset i.

    Args:
        datasets: Datasets in memory, on the same device. Their transforms
            are not kept.
        counts: Number of structures of each dataset in a batch.
        epoch: When an epoch ends. "shortest": when a dataset cannot fill its
            share of a batch any more, no structure is given twice, and with
            a random order the larger datasets give other structures at each
            epoch. "longest": when every structure of every dataset has been
            given, the smaller datasets are gone through several times.
        transform: Optional transform applied to each batch of structures.

    Returns:
        The mixed dataset, to give to auto_loader or to read with batches.

    Examples:
        One crystal and two molecules per batch of 3: three batches, until
        the crystals run out.

        >>> crystals = InMemoryDataset(Structures.collate([Structures.diamond()] * 3))
        >>> molecules = InMemoryDataset(Structures.collate([Structures.water()] * 20))
        >>> mixed = mix([crystals, molecules], [1, 2])
        >>> [batch.num_atoms.tolist() for batch in mixed.batches(3, shuffle=True)]
        [[8, 3, 3], [8, 3, 3], [8, 3, 3]]

        Until every molecule has been given: the crystals are given again.

        >>> mixed = mix([crystals, molecules], [1, 2], epoch="longest")
        >>> len(list(mixed.batches(3))), mixed.epoch_size()
        (10, 30)
    """
    together = InMemoryDataset.concat(datasets)
    sizes = [len(dataset) for dataset in datasets]

    return MixedDataset(together.structures, sizes, counts, epoch, transform)
