# Loading datasets

Every dataset is published on Hugging Face in the same HDF5 format and is loaded by name. The
file is downloaded on the first call and cached by `huggingface_hub`.

From a name to the batches of a training, in the order of this page:

```{.python notest}
from torchms.data import Filter, auto_loader, load_dataset, random_split
from torchms.graph.transforms import AddRadiusGraph

# 1. load a dataset, 2. keep the structures you want: here without the failed calculations
dataset = load_dataset("lematerial/unique", filter=Filter(maximum={"total_energy": 0.0}))

# 3. split it
train, val, test = random_split(dataset, [0.8, 0.1, 0.1], seed=0)

# 4. batches of 128 crystals with their graph, on the GPU
loader = auto_loader(train, batch_size=128, transform=AddRadiusGraph(5.0), shuffle=True)

for epoch in range(10):
    for structures in loader:
        structures.graph.edge_index
        ...
```

On LeMat-Bulk, 4.9 million structures, the filter takes 3 seconds and setting up the loader 8,
on a desktop with an RTX 4060; the training set is then 3.9 million structures held on the GPU.

| Step | With | Section |
|---|---|---|
| Choose a dataset | its name, `"mp-20/train"` | [Names and splits](#names-and-splits) |
| Load it | `load_dataset`, `HDF5Dataset` | [Loading a dataset](#loading-a-dataset) |
| Keep some structures | `Filter` | [Filtering a dataset](#filtering-a-dataset) |
| Make a training, a validation and a test set | `random_split` | [Splitting a dataset](#splitting-a-dataset) |
| Get batches for a training | `auto_loader` | [Training: `auto_loader`](#training-auto_loader) |
| Train on several datasets | a list of sources, `counts` | [Several datasets](#several-datasets-crystals-and-molecules-together) |

Filtering and splitting are optional: a name can go straight to `auto_loader`.

## Names and splits

| Name | Kind | Splits (default first) | Structures | Page |
|---|---|---|---|---|
| `perov-5` | inorganic materials | `train`, `val`, `test` | 18,928 | [Perov-5](../datasets/perov-5.md) |
| `carbon-24` | inorganic materials | `train`, `val`, `test` | 10,153 | [Carbon-24](../datasets/carbon-24.md) |
| `mp-20` | inorganic materials | `train`, `val`, `test` | 45,229 | [MP-20](../datasets/mp-20.md) |
| `mpts-52` | inorganic materials | `train`, `val`, `test` | 40,476 | [MPTS-52](../datasets/mpts-52.md) |
| `materials-project` | inorganic materials | `all` | 271,325 | [Materials Project](../datasets/materials-project.md) |
| `oqmd` | inorganic materials | `all`, `unique` | 1,407,269 / 1,259,553 | [OQMD](../datasets/oqmd.md) |
| `lematerial` | inorganic materials | `all`, `unique` | 5,335,299 / 4,927,038 | [LeMat-Bulk](../datasets/lematerial.md) |
| `qmof` | MOFs | `all` | 20,372 | [QMOF](../datasets/qmof.md) |
| `core-mof` | MOFs | `asr`, `fsr` | 12,020 / 7,061 | [CoRE MOF 2019](../datasets/core-mof.md) |
| `core-mof-2024` | MOFs | `asr`, `fsr`, `ion`, `ncr` | 1,371 / 1,191 / 100 / 1,826 | [CoRE MOF 2024](../datasets/core-mof-2024.md) |
| `qm9` | molecules | `all` | 133,885 | [QM9](../datasets/qm9.md) |
| `zinc250k` | molecules | `all` | 249,427 | [ZINC250k](../datasets/zinc250k.md) |

- A split is given with the name, `"mp-20/train"`, or as a second argument. Without it, the
  first split of the table is taken.
- Names are not case sensitive and `_` is accepted for `-`. An unknown name or split raises a
  `KeyError` listing the valid ones.
- The splits of a dataset are the ones of its source. A dataset with only `all` has none: make
  yours with [`random_split`](#splitting-a-dataset).
- ZINC20 in 3D and the MOFs of the Cambridge Structural Database are not published: see
  [Datasets to build yourself](../datasets/local.md).

```python
from torchms.data import available_datasets, dataset_url

available_datasets()["oqmd"]      # ('all', 'unique')
dataset_url("oqmd", "unique")
# 'https://huggingface.co/datasets/materials-toolkits/oqmd/resolve/main/unique.hdf5'
```

!!! warning "Read the page of a dataset before training on it"
    Datasets are **complete and unfiltered**: they keep the failed calculations and the
    duplicates of their source. The page of each dataset says what to watch for and gives a
    recommended filter; they are summed up in
    [Recommended filters](../datasets/index.md#recommended-filters).

## Loading a dataset

`load_dataset` gives the dataset itself, to read structures by their indices:

```{.python notest}
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

### Other files

A local file, a file of any other repository, or another version of a published one:

```{.python notest}
from torchms.data import HDF5Dataset

dataset = HDF5Dataset("my_structures.hdf5")
dataset = HDF5Dataset.from_hub("user/repository", "file.hdf5")
dataset = load_dataset("oqmd/unique", revision="main")   # options of hf_hub_download
```

To make such a file from your own structures, see
[Preprocessing your own data](preprocessing.md).

## Filtering a dataset

A `Filter` keeps the structures which pass its settings. It is given when a dataset is loaded,
or applied to a dataset already loaded:

```{.python notest}
from torchms.data import Filter, load_dataset

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
dataset = load_dataset("core-mof", filter=Filter(max_atoms=500, equal={"disorder": 0}))

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

- A structure is kept if it passes **every** setting given.
- A name is one of the four labels (`total_energy`, `formation_energy_per_atom`,
  `energy_above_hull`, `band_gap`) or a named property of the dataset, listed on its page. An
  unknown name raises a `KeyError`.
- A structure without value for a label (`NaN`) does not pass a condition on it.

The result is a dataset of the structures kept, in their order. It is read, split and given to
`auto_loader` like any other.

A filter keeps the same structures wherever the dataset is, and leaves it there: a dataset
read from its file stays one (nothing is copied), a dataset in memory stays in memory, on its
CPU or its GPU.

For a condition a `Filter` cannot express, `dataset.filter` takes any function from
`Structures` to a mask of the structures to keep, and `dataset.subset` the indices themselves:

```{.python notest}
small_cells = dataset.filter(lambda structures: structures.lattice.det().abs() < 100.0)
first = dataset.subset(range(1000))
```

The filter to start from for each dataset is in
[Recommended filters](../datasets/index.md#recommended-filters).

## Splitting a dataset

`random_split` cuts a dataset at random in parts of the sizes given:

```{.python notest}
from torchms.data import Filter, load_dataset, random_split

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
train, val, test = random_split(dataset, [0.8, 0.1, 0.1], seed=0)
```

- Every structure is in one part and only one.
- The same seed gives the same split.
- Filter first, then split: the split is made on what the filter kept.
- Each part is a dataset: it is read and given to `auto_loader` like any other.

### Keeping a split

The same seed gives the same split with the same version of PyTorch. To keep a split for good,
or to share it, save its indices:

```{.python notest}
from torchms.data import load_split, save_split, split_indices

train, val, test = split_indices(len(dataset), [0.8, 0.1, 0.1], seed=0)
save_split("qm9_split.npz", {"train": train, "val": val, "test": test}, dataset)

parts = load_split("qm9_split.npz", dataset)     # a dictionary of indices
train = dataset.subset(parts["train"])
```

The file is a NumPy `.npz` with an array of indices per part. It records the number of
structures of the dataset, and `load_split` raises an error for a dataset of another size.

!!! note "A split saved after a filter"
    The indices are positions in the dataset which was split. A split saved after a filter
    must be loaded after the same filter: the size of the dataset is checked, not the filter.

## Training: `auto_loader`

`auto_loader` gives the batches of a training. It looks at the CPU cores, the RAM and the GPU
of the machine and loads the dataset the fastest way it can:

```{.python notest}
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

The batches are [`Structures`](structures.md), already on the device, with their graphs if a
`transform` is given: there is nothing to move and nothing to build in the training loop. The
loader is iterated once per epoch and has a length. Setting it up takes about a second.

### What it loads

The source is the name of a dataset, the path of an HDF5 file, or a dataset, such as the part
of a split:

```{.python notest}
auto_loader("mp-20/train", batch_size=128)             # name and split in one string
auto_loader("my_structures.hdf5", batch_size=128)      # a local file
auto_loader(train, batch_size=128)                     # a dataset, a part of a split
auto_loader("mp-20/train", batch_size=128, filter=Filter(max_energy_above_hull=0.0))
```

| Argument | |
|---|---|
| `batch_size` | structures per batch |
| `transform` | [graph transforms](graphs.md#graphs-as-transforms), chained with `Compose` |
| `augment` | [augmentations](augmentation.md), applied before the graphs |
| `filter` | a [`Filter`](#filtering-a-dataset), applied before everything else |
| `shuffle`, `drop_last`, `generator` | as for a `DataLoader` |
| `device` | device of the batches, the GPU by default if there is one |
| `counts`, `epoch` | for a list of sources, see [below](#a-fixed-share-of-each-dataset-in-every-batch) |

### What it chooses

`print(loader.config)` says what was chosen and why:

```text
device cuda, dataset on the GPU, 0 workers
batches of 128, built by 512 on the device
- dataset on the GPU: 9 MB is at most 20% of the 7.1 GB free on it
- no worker: the batches are taken from memory
- graphs built on the GPU by batches of 512, split in batches of 128: the smallest within
  10 % of the fastest measured
```

Structures per second arriving on the GPU in batches of 128 with their graphs, on 300,000
structures of LeMat-Bulk, on a desktop with an RTX 4060 and on a node with an H100:

| | RTX 4060 | H100 |
|---|---:|---:|
| `auto_loader`, radius graph | 338,893 | 882,202 |
| `auto_loader`, full chain | 179,125 | 387,454 |

The radius graph is `AddRadiusGraph(5)`; the full chain adds triplets and quadruplets with
their geometry.

How it chooses, how to change one of its choices, and the pieces to build a loader by hand
(`DataLoader`, `OnDevice`, `InMemoryDataset`) are in the design note
[Loader and graph construction](../design/loader.md).

## Several datasets, crystals and molecules together

Give a list of sources. The datasets are loaded together in memory and, with `shuffle=True`,
every batch mixes their structures:

```{.python notest}
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

- A source is a name with its split, a file, or a dataset: the part of a split for instance.
- `filter` takes one filter for all the sources, or a list with one per source (`None` for a
  source which is not filtered).
- A batch has the properties of all the datasets, `NaN` for the structures of a dataset
  without them.
- The graph of a molecule has no periodic image: its `edge_shift` is zero. See
  [Periodicity](../design/periodicity.md).
- The datasets are held in memory together: `auto_loader` raises an error if they do not fit.

### A fixed share of each dataset in every batch

With a list alone, the proportions in a batch are the sizes of the datasets: the 27,136
crystals of the training set of MP-20 are drowned in the 249,427 molecules of ZINC250k.
`counts` gives the number of structures of each source in every batch:

```{.python notest}
loader = auto_loader(
    ["mp-20/train", "zinc250k"],
    batch_size=128,
    counts=[64, 64],                   # 64 crystals and 64 molecules in every batch
    transform=AddRadiusGraph(5.0),
    shuffle=True,
)
```

- `counts` must sum to `batch_size`. Every batch is full.
- In a batch the structures are grouped by source, in the order of the list: here the 64
  crystals, then the 64 molecules.

`epoch` says when an epoch ends:

| `epoch` | An epoch ends | Here |
|---|---|---|
| `"shortest"`, the default | when a source cannot fill its share any more. No structure is given twice, and with `shuffle=True` the larger source gives other structures at each epoch. | 424 batches: every crystal, 27,136 of the molecules |
| `"longest"` | when every structure of every source has been given. The smaller sources are gone through several times. | 3,898 batches: every molecule, the crystals about nine times |

### Without `auto_loader`

`InMemoryDataset.concat` makes one dataset from several, and `mix` one with a fixed share of
each:

```{.python notest}
from torchms.data import InMemoryDataset, load_dataset, mix

crystals = InMemoryDataset.from_hdf5(load_dataset("mp-20/train"))
molecules = InMemoryDataset.from_hdf5(load_dataset("qm9"))

together = InMemoryDataset.concat([crystals, molecules])
for structures in together.batches(128, shuffle=True):
    ...

mixed = mix([crystals, molecules], [64, 64], epoch="shortest")
for structures in mixed.batches(128, shuffle=True):
    ...
```
