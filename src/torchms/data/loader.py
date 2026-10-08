"""Batches of a loader moved to a device, and transformed on it."""

from collections.abc import Callable, Iterable, Iterator, Sized

import torch

from .batching import Structures


class OnDevice:
    """Move the batches of a loader to a device, then transform them there.

    The transform of a dataset runs on the CPU, in the workers of its
    DataLoader. The transform given here runs in the training process, on the
    device, on each batch as it comes out of the loader: this is the way to
    build graphs on the GPU (torchms.graph.transforms).

    The loader is not modified. An OnDevice is iterated as many times as
    needed, once per epoch, and has the length of its loader.

    Args:
        loader: Iterable of batches of Structures, a DataLoader for instance.
        device: Device of the batches, "cuda" for instance.
        transform: Optional transform applied to each batch once on the device.

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
    """

    def __init__(
        self,
        loader: Iterable[Structures],
        device: torch.device | str,
        transform: Callable[[Structures], Structures] | None = None,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.transform = transform

    def __iter__(self) -> Iterator[Structures]:
        for structures in self.loader:
            structures = structures.to(self.device)

            if self.transform is not None:
                structures = self.transform(structures)

            yield structures

    def __len__(self) -> int:
        """Number of batches of the loader.

        Raises:
            TypeError: If the loader has no length.
        """
        if not isinstance(self.loader, Sized):
            raise TypeError(f"{type(self.loader).__name__} has no length")

        return len(self.loader)
