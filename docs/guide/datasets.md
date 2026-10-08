# Loading datasets

Every dataset is published on Hugging Face in the same HDF5 format and is loaded by name. The
file is downloaded on the first call and cached by `huggingface_hub`. `auto_loader` gives the
batches of a training:

```python
from torchms.data import auto_loader
from torchms.data.augmentation import LatticeRotation
from torchms.graph.transforms import AddRadiusGraph

loader = auto_loader(
    "mp-20", "train",
    batch_size=128,
    transform=AddRadiusGraph(5.0),   # optional: graphs, see the Graphs page
    augment=LatticeRotation(),       # optional: augmentations, before the graphs
    shuffle=True,
    device="cuda",                   # optional: the GPU by default if there is one
)

for epoch in range(10):
    for structures in loader:        # 128 crystals, already on the GPU
        structures.graph.edge_index  # with their graph, built on the GPU
        ...
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

## Training: `auto_loader`

`auto_loader` looks at the CPU cores, the RAM and the GPU of the machine and loads the dataset
the fastest way it can. How it chooses, how to change one of its choices, and the pieces to
build a loader by hand (`DataLoader`, `OnDevice`, `InMemoryDataset`) are in the design note
[Loader and graph construction](../design/loader.md).

The batches are [`Structures`](structures.md), already on the device, with their graphs if a
`transform` is given: there is nothing to move and nothing to build in the training loop. The
loader is iterated once per epoch and has a length.

Structures per second arriving on the GPU in batches of 128 with their graphs, on 300,000
structures of LeMat-Bulk, on a desktop with an RTX 4060 and on a node with an H100:

| | RTX 4060 | H100 |
|---|---:|---:|
| `auto_loader`, radius graph | 338,893 | 882,202 |
| `auto_loader`, full chain | 179,125 | 387,454 |

The radius graph is `AddRadiusGraph(5)`; the full chain adds triplets and quadruplets with
their geometry.

`print(loader.config)` says what was chosen and why:

```text
device cuda, dataset on the GPU, 0 workers
batches of 128, built by 512 on the device
- dataset on the GPU: 9 MB is at most 20% of the 7.1 GB free on it
- no worker: the batches are taken from memory
- graphs built on the GPU by batches of 512, split in batches of 128: the smallest within
  10 % of the fastest measured
```

The source is the name of a dataset, the path of an HDF5 file, or a dataset:

```python
auto_loader("mp-20/train", batch_size=128)             # name and split in one string
auto_loader("my_structures.hdf5", batch_size=128)      # a local file
auto_loader(HDF5Dataset.from_hub("user/repository", "file.hdf5"), batch_size=128)
```

| Argument | |
|---|---|
| `batch_size` | structures per batch |
| `transform` | [graph transforms](graphs.md#graphs-as-transforms), chained with `Compose` |
| `augment` | [augmentations](augmentation.md), applied before the graphs |
| `shuffle`, `drop_last`, `generator` | as for a `DataLoader` |
| `device` | device of the batches, the GPU by default if there is one |

Setting up takes about a second.


## Reading structures

`load_dataset` gives the dataset itself, to read structures by their indices:

```python
from torchms.data import load_dataset

dataset = load_dataset("mp-20", "train")
dataset = load_dataset("mp-20/train")     # same thing
dataset = load_dataset("lematerial")      # default split: "all"

one = dataset[0]             # a batch of one structure
last = dataset[-1]
batch = dataset[10:42]       # a slice
batch = dataset[[20, 3, 8]]  # a list or a tensor of indices, in any order

len(dataset)                 # number of structures
```

A dataset returns batches of [`Structures`](structures.md), in the order of the indices asked.

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
