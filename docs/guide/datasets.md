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

A dataset returns batches of [`Structures`](structures.md), in the order of the indices asked.

```python
one = dataset[0]             # a batch of one structure
last = dataset[-1]
batch = dataset[10:42]       # a slice
batch = dataset[[20, 3, 8]]  # a list or a tensor of indices, in any order

len(dataset)                 # number of structures
```

## Training with a DataLoader

A dataset goes to the `DataLoader` of PyTorch as it is, with no sampler and no collate
function to write:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True)

for structures in loader:   # a Structures batch of 64 crystals
    structures = structures.to("cuda")
    ...
```

Every argument of `DataLoader` works as usual (`drop_last`, `sampler`, `generator`, ...), and
so do `torch.utils.data.random_split` and `Subset`:

```python
from torch.utils.data import random_split

train, validation = random_split(dataset, [0.9, 0.1])
loader = DataLoader(train, batch_size=64, shuffle=True)
```

The structures of a batch are read from the file in one call, and the `transform` of the
dataset is applied once, to the whole batch.

The dataset opens its HDF5 file once in each process, so it works with several workers:

```python
loader = DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True
)
```

Reading alone is fast: one process is enough for most trainings. Workers pay off when the
`transform` of the dataset is the slow part, since it runs in the workers. This is the case
of the graph transforms, see [Building graphs](#building-graphs) below. How many workers to
use, for which transforms and on which hardware, is measured in
[Loader and graph construction](../design/loader.md).

!!! note "Collate function"
    The loader batches `Structures` with `Structures.collate`, which torchms registers as the
    default collate of PyTorch for this type. A `collate_fn` of your own receives a list of
    `Structures` (one batch already read for an HDF5 dataset): start it with
    `Structures.collate(batch)`. Graphs are not built there but by
    [transforms](graphs.md#graphs-as-transforms).

## Transforms

A transform is a function from `Structures` to `Structures`, applied to each batch read:

```python
def translate(structures):
    structures.x = (structures.x + 0.5) % 1.0
    return structures

dataset = load_dataset("mp-20", "train", transform=translate)
```

Ready-made random transforms are described in [Data augmentation](augmentation.md).

## Building graphs

Graphs are built by transforms too, which attach them to the structures of each batch:

```python
from torch.utils.data import DataLoader
from torchms.data import load_dataset
from torchms.graph.transforms import AddRadiusGraph

dataset = load_dataset("mp-20", "train", transform=AddRadiusGraph(5.0))
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

for structures in loader:
    structures.graph.edge_index   # (2, n_edges): atoms closer than 5 Å
    structures.graph.num_edges    # edges of each structure
```

Several steps are chained with `Compose`, after the augmentations:

```python
from torchms.data.augmentation import Compose, LatticeRotation
from torchms.graph.transforms import AddTriplets

transform = Compose([LatticeRotation(), AddRadiusGraph(5.0), AddTriplets(radius=3.0)])
```

Given to the dataset, the transforms build the graphs on the CPU, in the workers. To build
them on the GPU, give them to `OnDevice` instead, which moves each batch of the loader to the
device and transforms it there:

```python
from torchms.data import OnDevice

dataset = load_dataset("mp-20", "train")
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
loader = OnDevice(loader, "cuda", transform=AddRadiusGraph(5.0))

for structures in loader:         # on the GPU, with its graph
    ...
```

Which one to choose depends on the number of CPU cores, on the size of the batches and on
what limits the training: see
[Where to build the graphs](graphs.md#where-to-build-the-graphs) for the rule, and
[Loader and graph construction](../design/loader.md) for the measures.

[Graphs as transforms](graphs.md#graphs-as-transforms) lists the transforms (nearest
neighbours, chemical bonds, line graph, triplets, quadruplets, geometry of the edges and
angles), and [Fields attached to the structures](graphs.md#fields-attached-to-the-structures)
describes every field they add.

## Other files

```python
dataset = load_dataset("oqmd/unique", revision="main")   # options of hf_hub_download
```

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
