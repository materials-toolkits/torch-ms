"""Aliases of the preprocessed datasets published on Hugging Face.

Each dataset has a name and one HDF5 file per split:

| Name | Repository | Splits |
|---|---|---|
| perov-5 | materials-toolkits/perov-5 | train, val, test |
| carbon-24 | materials-toolkits/carbon-24 | train, val, test |
| mp-20 | materials-toolkits/mp-20 | train, val, test |
| materials-project | materials-toolkits/materials-project | all |
| oqmd | materials-toolkits/oqmd | all, unique |
| lematerial | materials-toolkits/lematerial | all, unique |

The default split of a dataset is its first one. A split can be given with the
name, as "mp-20/test".

```python
from torchms.data import load_dataset

dataset = load_dataset("mp-20", "train")
dataset = load_dataset("lematerial/unique")
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torchms.data.batching import Structures
from torchms.data.hdf5 import HDF5Dataset

HUB_URL = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class DatasetAlias:
    """Location of a published dataset.

    Attributes:
        repo_id: Hugging Face dataset repository.
        splits: Name of its splits, the first one is the default. The file of
            a split is "<split>.hdf5".
        description: Content of the dataset.
    """

    repo_id: str
    splits: tuple[str, ...]
    description: str


_CDVAE_SPLITS = ("train", "val", "test")

DATASETS: dict[str, DatasetAlias] = {
    "perov-5": DatasetAlias(
        "materials-toolkits/perov-5",
        _CDVAE_SPLITS,
        "18,928 perovskites with 5 atoms (CDVAE): formation energy, band gap",
    ),
    "carbon-24": DatasetAlias(
        "materials-toolkits/carbon-24",
        _CDVAE_SPLITS,
        "10,153 carbon structures with 6 to 24 atoms (CDVAE): total energy",
    ),
    "mp-20": DatasetAlias(
        "materials-toolkits/mp-20",
        _CDVAE_SPLITS,
        "45,229 structures of the Materials Project with at most 20 atoms (CDVAE): "
        "formation energy, energy above hull, band gap",
    ),
    "materials-project": DatasetAlias(
        "materials-toolkits/materials-project",
        ("all",),
        "271,325 structures of the Materials Project 2026.09.28: total energy, "
        "formation energy, energy above hull, band gap",
    ),
    "oqmd": DatasetAlias(
        "materials-toolkits/oqmd",
        ("all", "unique"),
        "OQMD v1.8, 1,407,269 structures (all) or 1,259,553 without its duplicates "
        "(unique): total energy, formation energy, stability, band gap",
    ),
    "lematerial": DatasetAlias(
        "materials-toolkits/lematerial",
        ("all", "unique"),
        "LeMat-Bulk compatible_pbe, 5,335,299 structures (all) or 4,927,038 with one "
        "structure per fingerprint (unique): total energy, energy above hull",
    ),
}


def available_datasets() -> dict[str, tuple[str, ...]]:
    """Names of the datasets and of their splits, the default split first.

    Examples:
        >>> available_datasets()["oqmd"]
        ('all', 'unique')
    """
    return {name: DATASETS[name].splits for name in sorted(DATASETS)}


def resolve(name: str, split: str | None = None) -> tuple[str, str]:
    """Get the repository and the file of a split of a dataset.

    Args:
        name: Name of the dataset, or "<name>/<split>". Upper case and "_"
            are accepted for "-".
        split: Split of the dataset, its first one by default.

    Returns:
        Hugging Face dataset repository and name of the file in it.

    Raises:
        KeyError: If the dataset or the split is unknown.
        ValueError: If the split is given with the name and as an argument.

    Examples:
        >>> resolve("mp-20", "test")
        ('materials-toolkits/mp-20', 'test.hdf5')
        >>> resolve("lematerial/unique")
        ('materials-toolkits/lematerial', 'unique.hdf5')
    """
    if "/" in name:
        if split is not None:
            raise ValueError(f'the split is given twice: "{name}" and "{split}"')
        name, split = name.split("/", maxsplit=1)

    key = name.lower().replace("_", "-")
    if key not in DATASETS:
        raise KeyError(f'unknown dataset "{name}", must be one of {sorted(DATASETS)}')

    alias = DATASETS[key]
    if split is None:
        split = alias.splits[0]
    if split not in alias.splits:
        raise KeyError(
            f'unknown split "{split}" of {key}, must be one of {list(alias.splits)}'
        )

    return alias.repo_id, f"{split}.hdf5"


def dataset_url(name: str, split: str | None = None, revision: str = "main") -> str:
    """URL of the file of a split of a dataset.

    Examples:
        >>> dataset_url("oqmd", "unique")
        'https://huggingface.co/datasets/materials-toolkits/oqmd/resolve/main/unique.hdf5'
    """
    repo_id, filename = resolve(name, split)
    return f"{HUB_URL}/{repo_id}/resolve/{revision}/{filename}"


def load_dataset(
    name: str,
    split: str | None = None,
    transform: Callable[[Structures], Structures] | None = None,
    **kwargs: Any,
) -> HDF5Dataset:
    """Load a published dataset by its name, downloaded on the first call.

    Requires huggingface_hub (pip install torchms[hub]).

    Args:
        name: Name of the dataset, or "<name>/<split>", see available_datasets.
        split: Split of the dataset, its first one by default.
        transform: Optional transform to apply to structures.
        **kwargs: Forwarded to huggingface_hub.hf_hub_download (revision,
            token, cache_dir, ...).

    Raises:
        KeyError: If the dataset or the split is unknown.
    """
    repo_id, filename = resolve(name, split)
    return HDF5Dataset.from_hub(repo_id, filename, transform=transform, **kwargs)
