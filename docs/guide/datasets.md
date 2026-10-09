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
| `qmof` | `all` | 20,372 | [QMOF](../datasets/qmof.md) |
| `core-mof` | `asr`, `fsr` | 12,020 / 7,061 | [CoRE MOF 2019](../datasets/core-mof.md) |
| `core-mof-2024` | `asr`, `fsr`, `ion`, `ncr` | 1,371 / 1,191 / 100 / 1,826 | [CoRE MOF 2024](../datasets/core-mof-2024.md) |
| `qm9` | `all` | 133,885 | [QM9](../datasets/qm9.md) |
| `zinc250k` | `all` | 249,427 | [ZINC250k](../datasets/zinc250k.md) |

The first six are inorganic crystals, `qmof`, `core-mof` and `core-mof-2024` metal-organic frameworks, `qm9` and
`zinc250k` organic molecules. ZINC20 in 3D and the MOFs of the Cambridge Structural Database
are built locally: see [Datasets to build yourself](../datasets/local.md).

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

## Several datasets, crystals and molecules together

Give a list of sources. The datasets are loaded together in memory and, with `shuffle=True`,
every batch mixes their structures:

```python
loader = auto_loader(
    ["mp-20/train", "qm9"],
    batch_size=128,
    transform=AddRadiusGraph(5.0),
    shuffle=True,
)

for structures in loader:
    crystal = structures.pbc.all(dim=1)       # (128,): False for the molecules
    structures.properties["homo"]             # NaN for the crystals
```

The split of each dataset is given in its name. The graph of a molecule has no periodic image:
its `edge_shift` is zero. A batch has the properties of all the datasets, `NaN` for the
structures of a dataset without them. See [Periodicity](../design/periodicity.md).

Without `auto_loader`, `InMemoryDataset.concat` makes one dataset from several:

```python
from torchms.data import InMemoryDataset

dataset = InMemoryDataset.concat(
    [InMemoryDataset.from_hdf5(load_dataset("mp-20/train")), InMemoryDataset.from_hdf5(load_dataset("qm9"))]
)
```


## Filtering a dataset

A `Filter` keeps the structures which pass its settings. It is given when a dataset is loaded,
by its name or by `auto_loader`, or applied to a dataset:

```python
from torchms.data import Filter, auto_loader, load_dataset

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
dataset = load_dataset("core-mof", filter=Filter(max_atoms=500, equal={"disorder": 0}))
loader = auto_loader("mp-20/train", batch_size=128, filter=Filter(max_energy_above_hull=0.1))

organic = dataset.filter(Filter(elements=(1, 6, 7, 8)))   # a dataset already loaded
```

| Setting | Keeps the structures |
|---|---|
| `min_atoms`, `max_atoms` | with a number of atoms in the range |
| `elements` | made only of these atomic numbers |
| `exclude_elements` | without these atomic numbers |
| `periodic` | `True`: crystals only, `False`: molecules only |
| `max_energy_above_hull` | not above this energy above the hull, in eV/atom |
| `minimum`, `maximum` | `{"band_gap": 0.5}`: a label or a named property above, below a value |
| `equal` | `{"disorder": 0}`: a label or a named property equal to a value |
| `known` | `("total_energy",)`: with a value for these labels or properties |

A structure is kept if it passes every setting given. A name is one of the four labels
(`total_energy`, `formation_energy_per_atom`, `energy_above_hull`, `band_gap`) or a named
property of the dataset; an unknown name raises a `KeyError`. A structure without value for a
label (`NaN`) does not pass a condition on it.

The result is a dataset of the structures kept, in their order: it is read, split, mixed and
given to `auto_loader` like any other. Nothing is copied for a dataset read from a file.
`dataset.filter` also takes any function from `Structures` to a mask of the structures to
keep, and `dataset.subset(indices)` the indices themselves. A list of sources of `auto_loader`
takes one filter for all of them, or a list with one per source (`None` for none).

## Splitting a dataset

`random_split` cuts a dataset at random in parts of the sizes given:

```python
from torchms.data import load_dataset, random_split

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
train, val, test = random_split(dataset, [0.8, 0.1, 0.1], seed=0)

loader = auto_loader(train, batch_size=128, shuffle=True)
```

Every structure is in one part and only one, and the same seed gives the same split. Filter
first, then split: the split is made on what the filter kept.

To keep a split for good, or to share it, save its indices:

```python
from torchms.data import load_split, save_split, split_indices

train, val, test = split_indices(len(dataset), [0.8, 0.1, 0.1], seed=0)
save_split("qm9_split.npz", {"train": train, "val": val, "test": test}, dataset)

parts = load_split("qm9_split.npz", dataset)     # a dictionary of indices
train = dataset.subset(parts["train"])
```

The file is a NumPy `.npz` with an array of indices per part. It records the number of
structures of the dataset, and `load_split` raises an error for a dataset of another size.
The indices are positions in the dataset given: a split saved after a filter is loaded after
the same filter.

## A fixed share of each dataset in every batch

With a list of sources alone, the proportions in a batch are the sizes of the datasets: 27,136
crystals of MP-20 are drowned in the 249,427 molecules of ZINC250k. `counts` gives the number of
structures of each source in every batch:

```python
loader = auto_loader(
    ["mp-20/train", train],            # names, files, datasets, parts of a split
    batch_size=128,
    counts=[64, 64],                   # 64 crystals and 64 molecules in every batch
    transform=AddRadiusGraph(5.0),
    shuffle=True,
)
```

- `counts` must sum to `batch_size`. Every batch is full.
- In a batch the structures are grouped by source, in the order of the list: here the 64
  crystals, then the 64 molecules.
- `epoch="shortest"`, the default: an epoch ends when a source cannot fill its share any more,
  after 424 batches here. No structure is given twice, and with `shuffle=True` the larger
  source gives other structures at each epoch.
- `epoch="longest"`: an epoch ends when every structure of every source has been given. The
  smaller sources are gone through several times.

Without `auto_loader`, `mix` makes the dataset and `batches` reads it:

```python
from torchms.data import InMemoryDataset, mix

mixed = mix([InMemoryDataset.from_hdf5(crystals), InMemoryDataset.from_hdf5(molecules)], [64, 64])
for structures in mixed.batches(128, shuffle=True):
    ...
```

The sources of a mix are held in memory together, as the ones of a list.

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
