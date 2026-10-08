"""Batches of a loader moved to a device, and transformed on it."""

import math
from collections.abc import Callable, Iterable, Iterator, Sized

import torch

from .batching import Structures


class OnDevice:
    """Move the batches of a loader to a device, then transform them there.

    The transform of a dataset runs on the CPU, in the workers of its
    DataLoader. The transform given here runs in the training process, on the
    device, on each batch as it comes out of the loader: this is the way to
    build graphs on the GPU (torchms.graph.transforms).

    With a batch_size, each batch of the loader is transformed then split into
    batches of this size. A GPU builds the graphs of one large batch much
    faster than the ones of many small batches, and a loader gives more
    structures per second in large batches: give the loader a large batch size,
    8192 for instance, and the size of the batches of the training here. The
    batches of a shuffled loader are the same as without it.

    The loader is not modified. An OnDevice is iterated as many times as
    needed, once per epoch.

    Args:
        loader: Iterable of batches of Structures, a DataLoader for instance.
        device: Device of the batches, "cuda" for instance.
        transform: Optional transform applied to each batch once on the device.
        batch_size: Number of structures of the batches given, each batch of
            the loader being split after the transform. By default the
            batches of the loader are given as they are.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from torchms.graph.transforms import AddRadiusGraph
        >>> dataset = [Structures.diamond(), Structures.zincblende()] * 3
        >>> loader = DataLoader(dataset, batch_size=4)
        >>> loader = OnDevice(loader, "cpu", transform=AddRadiusGraph(3.0))
        >>> len(loader)
        2
        >>> [structures.graph.num_edges.tolist() for structures in loader]
        [[224, 32, 224, 32], [224, 32]]

        Graphs built on batches of 4, given in batches of 3:

        >>> loader = DataLoader(dataset, batch_size=4)
        >>> loader = OnDevice(loader, "cpu", AddRadiusGraph(3.0), batch_size=3)
        >>> [structures.graph.num_edges.tolist() for structures in loader]
        [[224, 32, 224], [32], [224, 32]]
    """

    def __init__(
        self,
        loader: Iterable[Structures],
        device: torch.device | str,
        transform: Callable[[Structures], Structures] | None = None,
        batch_size: int | None = None,
    ):
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.loader = loader
        self.device = torch.device(device)
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Structures]:
        for structures in self.loader:
            structures = structures.to(self.device)

            if self.transform is not None:
                structures = self.transform(structures)

            if self.batch_size is None:
                yield structures
            else:
                yield from structures.split(self.batch_size)

    def __len__(self) -> int:
        """Number of batches given.

        Raises:
            TypeError: If the loader has no length, or with a batch_size if
                it is not a DataLoader with a batch size.
        """
        if not isinstance(self.loader, Sized):
            raise TypeError(f"{type(self.loader).__name__} has no length")

        if self.batch_size is None:
            return len(self.loader)

        # the number of structures of each batch of the loader
        size = getattr(self.loader, "batch_size", None)
        sampler = getattr(self.loader, "sampler", None)
        if not isinstance(size, int) or not isinstance(sampler, Sized):
            raise TypeError(
                "the number of batches is only known for a DataLoader with a batch size"
            )

        full, rest = divmod(len(sampler), size)
        if getattr(self.loader, "drop_last", False):
            rest = 0

        return full * math.ceil(size / self.batch_size) + math.ceil(
            rest / self.batch_size
        )
