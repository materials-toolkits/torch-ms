"""Splits of a dataset in parts: training, validation and test for instance.

A split is the indices of the structures of each part. The parts of a dataset
are datasets of these structures, see HDF5Dataset.subset.

```python
from torchms.data import load_dataset, random_split, save_split, split_indices

dataset = load_dataset("qm9")
train, val, test = random_split(dataset, [0.8, 0.1, 0.1], seed=0)

# the same split, kept in a file
train, val, test = split_indices(len(dataset), [0.8, 0.1, 0.1], seed=0)
save_split("qm9_split.npz", {"train": train, "val": val, "test": test}, dataset)
```
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np
import torch
from jaxtyping import Int

from torchms.data.hdf5 import HDF5Dataset, Indices, as_indices
from torchms.data.memory import InMemoryDataset

AnyDataset = TypeVar("AnyDataset", HDF5Dataset, InMemoryDataset)

# name of the array of a file of a split which holds the size of its dataset
SIZE = "_num_structures"


def split_indices(
    num_structures: int, fractions: Sequence[float], seed: int = 0
) -> list[Int[torch.Tensor, "n_part"]]:
    """Draw the structures of each part of a random split.

    The structures are drawn in a random order, which is cut at the fractions
    given: every structure is in one part and only one. The last part takes
    what the rounding of the others leaves.

    The split is the same for the same number of structures, fractions and
    seed, with the same version of torch. Save it to keep it for good, see
    save_split.

    Args:
        num_structures: Number of structures of the dataset.
        fractions: Share of the structures of each part, positive numbers
            which sum to 1.
        seed: Seed of the random order.

    Returns:
        Indices of the structures of each part, on the CPU, in ascending
        order.

    Raises:
        ValueError: If the fractions are not positive or do not sum to 1.

    Examples:
        >>> train, val, test = split_indices(10, [0.8, 0.1, 0.1], seed=0)
        >>> len(train), len(val), len(test)
        (8, 1, 1)
        >>> torch.cat([train, val, test]).sort().values
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    shares = torch.tensor(list(fractions), dtype=torch.float64)

    if len(shares) == 0 or bool((shares <= 0).any()):
        raise ValueError("the fractions of a split must be positive")

    if abs(float(shares.sum()) - 1.0) > 1e-6:
        raise ValueError(f"the fractions of a split must sum to 1, not {shares.sum()}")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_structures, generator=generator)

    # where the order is cut: the last part ends with the last structure
    ends = (torch.cumsum(shares, dim=0) * num_structures + 1e-9).floor().long()
    ends[-1] = num_structures
    sizes = torch.diff(ends, prepend=torch.zeros(1, dtype=torch.int64))

    return [part.sort().values for part in order.split(sizes.tolist())]


def random_split(
    dataset: AnyDataset, fractions: Sequence[float], seed: int = 0
) -> list[AnyDataset]:
    """Split a dataset at random in parts of given sizes.

    Args:
        dataset: Dataset to split, read from a file or held in memory.
        fractions: Share of the structures of each part, positive numbers
            which sum to 1.
        seed: Seed of the random order, see split_indices.

    Returns:
        A dataset per part, of the type of the dataset given: its subsets.

    Examples:
        >>> from torchms.data.batching import Structures
        >>> dataset = InMemoryDataset(Structures.collate([Structures.diamond()] * 10))
        >>> train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
        >>> len(train), len(val), len(test)
        (8, 1, 1)
    """
    parts = split_indices(len(dataset), fractions, seed)

    return [dataset.subset(part) for part in parts]


def save_split(
    file: str | os.PathLike,
    parts: Mapping[str, Indices],
    dataset: HDF5Dataset | InMemoryDataset | int | None = None,
) -> None:
    """Save the indices of the parts of a split to a file.

    Args:
        file: File written, in the format of numpy.savez (".npz").
        parts: Indices of the structures of each part, by name.
        dataset: The dataset split, or its number of structures. It is saved
            with the split, to check later that the split is loaded for the
            same dataset.

    Examples:
        >>> import tempfile
        >>> train, test = split_indices(10, [0.7, 0.3])
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     file = os.path.join(directory, "split.npz")
        ...     save_split(file, {"train": train, "test": test}, 10)
        ...     {name: len(part) for name, part in load_split(file, 10).items()}
        {'train': 7, 'test': 3}
    """
    if SIZE in parts:
        raise ValueError(f'a part of a split cannot be named "{SIZE}"')

    arrays = {}
    for name, indices in parts.items():
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        arrays[name] = np.asarray(indices, dtype=np.int64).reshape(-1)

    if dataset is not None:
        size = dataset if isinstance(dataset, int) else len(dataset)
        arrays[SIZE] = np.array(size, dtype=np.int64)

    with open(file, "wb") as output:
        np.savez(output, **arrays)


def load_split(
    file: str | os.PathLike,
    dataset: HDF5Dataset | InMemoryDataset | int | None = None,
) -> dict[str, Int[torch.Tensor, "n_part"]]:
    """Load the indices of the parts of a split saved by save_split.

    Args:
        file: File of the split.
        dataset: The dataset to split, or its number of structures, to check
            that the split was made for it.

    Returns:
        Indices of the structures of each part, by name, on the CPU. Give
        them to the subset method of the dataset.

    Raises:
        ValueError: If the split was saved for a dataset of another size, or
            if an index is not a structure of the dataset.
    """
    with np.load(file) as arrays:
        saved = {name: arrays[name] for name in arrays.files}

    size = saved.pop(SIZE, None)

    if dataset is not None:
        expected = dataset if isinstance(dataset, int) else len(dataset)

        if size is not None and int(size) != expected:
            raise ValueError(
                f"the split of {os.fspath(file)} is the one of a dataset of "
                f"{int(size)} structures, not {expected}"
            )

        try:
            saved = {name: as_indices(part, expected) for name, part in saved.items()}
        except IndexError as error:
            raise ValueError(f"the split of {os.fspath(file)}: {error}") from error

    return {name: torch.from_numpy(part) for name, part in saved.items()}
