# Loading datasets

Every dataset is published on Hugging Face in the same HDF5 format and is loaded by name. The
file is downloaded on the first call and cached by `huggingface_hub`.

```python
from torchms.data import load_dataset

dataset = load_dataset("mp-20", "train")
dataset = load_dataset("mp-20/train")     # same thing
dataset = load_dataset("lematerial")      # default split: "all"
```

## Names and splits

| Name | Splits (default first) | Structures | Page |
|---|---|---|---|
| `perov-5` | `train`, `val`, `test` | 18,928 | [Perov-5](../datasets/perov-5.md) |
| `carbon-24` | `train`, `val`, `test` | 10,153 | [Carbon-24](../datasets/carbon-24.md) |
| `mp-20` | `train`, `val`, `test` | 45,229 | [MP-20](../datasets/mp-20.md) |
| `materials-project` | `all` | 271,325 | [Materials Project](../datasets/materials-project.md) |
| `oqmd` | `all`, `unique` | 1,407,269 / 1,259,553 | [OQMD](../datasets/oqmd.md) |
| `lematerial` | `all`, `unique` | 5,335,299 / 4,927,038 | [LeMat-Bulk](../datasets/lematerial.md) |

Names are not case sensitive and `_` is accepted for `-`. An unknown name or split raises a
`KeyError` listing the valid ones.

```python
from torchms.data import available_datasets, dataset_url

available_datasets()["oqmd"]      # ('all', 'unique')
dataset_url("oqmd", "unique")
# 'https://huggingface.co/datasets/materials-toolkits/oqmd/resolve/main/unique.hdf5'
```

## Reading structures

A dataset returns batches of [`Structures`](structures.md). Indices must be in ascending
order.

```python
one = dataset[0]             # a batch of one structure
batch = dataset[10:42]       # a slice
batch = dataset[[3, 8, 20]]  # a list or a tensor of indices

len(dataset)                 # number of structures
```

## Training with a DataLoader

The dataset reads a whole batch in one call, and needs its indices in ascending order. Give
the `DataLoader` a sampler of sorted batches and disable its own batching:

```python
from torch.utils.data import BatchSampler, DataLoader, RandomSampler


class SortedBatches:
    """Random batches of indices, each one in ascending order."""

    def __init__(self, dataset, batch_size):
        self.batches = BatchSampler(RandomSampler(dataset), batch_size, drop_last=False)

    def __iter__(self):
        return (sorted(batch) for batch in self.batches)

    def __len__(self):
        return len(self.batches)


loader = DataLoader(dataset, sampler=SortedBatches(dataset, 64), batch_size=None)

for structures in loader:   # a Structures batch of 64 crystals
    ...
```

The dataset opens its HDF5 file once in each process, so it works with several workers:

```python
loader = DataLoader(
    dataset, sampler=SortedBatches(dataset, 64), batch_size=None, num_workers=4
)
```

Reading is fast (tens of thousands of structures per second from a local file): workers help
when the `transform` of the dataset is the slow part, since it runs in the workers.

## Transforms and other files

```python
def translate(structures):
    structures.x = (structures.x + 0.5) % 1.0
    return structures

dataset = load_dataset("mp-20", "train", transform=translate)
dataset = load_dataset("oqmd/unique", revision="main")   # options of hf_hub_download
```

Ready-made random transforms are described in [Data augmentation](augmentation.md).

A local file, or a file of any other repository:

```python
from torchms.data import HDF5Dataset

dataset = HDF5Dataset("my_structures.hdf5")
dataset = HDF5Dataset.from_hub("user/repository", "file.hdf5")
```

## Before you train

Datasets are **complete and unfiltered**: they keep the failed calculations and the duplicates
of their source. Read the page of the dataset you use, and see
[Preprocessing your own data](preprocessing.md) for the filters.
