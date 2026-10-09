# Loader and graph construction

Giving a training its batches, with their graphs, as fast as the machine allows. The first
part explains why this is not one setting which is right everywhere, and what
[`auto_loader`](../guide/datasets.md#training-auto_loader) chooses. The second part shows how
to change its choices, or to set a loader up by hand.

## Why the setup is not trivial

A loader has to read structures, build their graphs and put the result on the device of the
model. Each of these can be done in several places, and the fastest place depends on the
hardware. The same code is 7 to 28 times faster or slower between two reasonable setups:

| Loader, batches of 128 on the GPU with a radius graph | RTX 4060 machine | H100 machine |
|---|---:|---:|
| plain `DataLoader` with 16 workers, graphs built per batch on the GPU | 49,227 | 31,023 |
| the setup `auto_loader` chooses | 338,893 | 882,202 |

All the numbers of this page are **structures per second**, on two machines:

| | RTX 4060 machine | H100 machine |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB | one NVIDIA H100 NVL, 96 GB |
| CPU | a desktop, 32 threads | 32 cores of an Intel Xeon Gold 6430 |
| One CPU thread | the faster one | about 2 times slower |
| torch | 2.11.0 | 2.14.1 |

The H100 is the more powerful GPU, behind the slower CPU: this is what makes the right setup
differ between the two. Unless said otherwise the data is 300,000 structures of LeMat-Bulk
(9 atoms per structure on average), the radius graph is `AddRadiusGraph(5)`, and the full
chain adds `AddTriplets(3)` and `AddQuadruplets(2.5)` with their geometry.

There are six choices.

| Choice | Options | What decides |
|---|---|---|
| [Where the dataset lives](#where-the-dataset-lives) | its file on disk, RAM, the memory of the GPU | the size of the dataset against the memory free |
| [Number of workers](#number-of-workers) | 0 to 16 | whether a file is read, and the number of cores |
| [Where the graphs are built](#where-the-graphs-are-built) | in the workers before the collate, or in the training process after it | whether there is a GPU |
| [On which device](#where-the-graphs-are-built) | CPU or GPU | whether there is a GPU, and the size of the batches |
| [Size of the batches the graphs are built on](#large-batches-split) | the size asked, or larger batches which are then split | the GPU and the speed of the CPU |
| [Compiling the transforms](#compiling-does-not-help) | `torch.compile` or not | nothing: it does not help |

### Where the dataset lives

The datasets are HDF5 files. A batch of random structures read from a file is scattered
reads, done by worker processes which then send the batch to the training process. Held in
memory as one large batch, a dataset gives a batch by selecting rows of tensors: no file, no
worker, no transfer between processes.

Every published dataset fits in memory:

| Dataset | Structures | In memory | Read from disk once |
|---|---:|---:|---:|
| MP-20 (train) | 27,136 | 9 MB | under 0.1 s |
| Materials Project | 271,325 | 204 MB | 0.1 s |
| LeMat-Bulk (unique) | 4,927,038 | 1.5 GB | 1.1 s |

Batches of 128 arriving on the GPU with their graphs, for the three places:

| Dataset in | RTX 4060, radius graph | RTX 4060, full chain | H100, radius graph | H100, full chain |
|---|---:|---:|---:|---:|
| the memory of the GPU | 335,158 | 179,695 | 881,897 | 391,922 |
| RAM | 297,202 | 154,732 | 737,482 | 360,055 |
| its file, 16 workers | 90,009 | 89,395 | 133,566 | 121,990 |

- Memory is 2 to 7 times faster than the file.
- The memory of the GPU is 9 to 20 % faster than RAM, which needs a copy to the GPU for each
  batch. RAM keeps the memory of the GPU for the model, and serves several GPUs with one copy.
- The file is the only option for a dataset larger than the memory, and it needs workers.

`auto_loader` puts the dataset in the memory of the GPU if it is at most 20 % of what is free
on it, else in RAM if it is at most 50 % of what is available, else leaves it in its file.
These two shares are a choice, not a measure: they leave the memory to the model.

### Number of workers

Workers only matter when the dataset is read from its file, or when there is no GPU and the
graphs are built on the CPU. Reading MP-20 from its file, without transform:

| Workers | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 4060 machine | 26,806 | 24,698 | 43,618 | 58,091 | 59,963 | 55,502 |
| H100 machine | 11,113 | 12,464 | 24,216 | 41,883 | 54,683 | 51,584 |

- One worker is about as fast as none: the batch has to be sent from the worker to the
  training process.
- The rate grows up to 8 workers and not beyond, on both machines.
- A dataset in memory needs no worker at all, and gives millions of structures per second.

`auto_loader` uses no worker for a dataset in memory with a GPU, and one worker per core but
one, 16 at most, to read a file or to build graphs without GPU. The cores are the ones of
the job under Slurm, not the ones of the node.

### Where the graphs are built

A graph transform builds its graph on the device of the structures it is given. It can be
called in two places:

- **in the workers, before the collate**: as the `transform` of the dataset. Each worker
  builds the graphs of its batch on one CPU thread, and the collate receives finished batches;
- **in the training process, after the collate**: on each batch which comes out of the loader,
  once it is on the device. With a GPU, the graphs are built on the GPU.

The collate itself never builds a graph: it only batches.

The time of the same batch of 128 structures, in milliseconds:

| One batch of 128 | RTX 4060 machine | H100 machine |
|---|---:|---:|
| radius graph, one CPU thread | 8.4 | 14.5 |
| radius graph, GPU | 0.71 | 1.41 |
| full chain, one CPU thread | 13.6 | 26.6 |
| full chain, GPU | 2.67 | 4.25 |

The GPU is 5 to 12 times faster than one CPU thread. Through a loader of batches of 128, 16
workers nearly make up for it:

| Graphs built | RTX 4060, radius graph | RTX 4060, full chain | H100, radius graph | H100, full chain |
|---|---:|---:|---:|---:|
| in 16 workers, on the CPU | 34,231 | 21,349 | 29,893 | 17,389 |
| in the training process, on the GPU | 47,904 | 31,723 | 29,222 | 17,237 |

With batches of 128 the two ways are close: the GPU is ahead by 40 to 50 % on the desktop, and
equal to the workers on the H100 machine, whose CPU is slower. The H100 gives nothing here.
The reason is in the next section, and it is what makes the GPU the right place.

`auto_loader` builds the graphs on the GPU when there is one, in the workers otherwise.

### Large batches, split

Building the graphs of a batch takes about 380 small operations of PyTorch, whatever the size
of the batch. Each one is started by a CPU thread, then runs on the GPU. For a batch of 128
structures, the GPU finishes each operation long before the next one is started: it waits.
A profile of one call of the radius graph shows it, on the two machines:

| One call | RTX 4060, batch of 128 | H100, batch of 128 | RTX 4060, batch of 8,192 | H100, batch of 8,192 |
|---|---:|---:|---:|---:|
| Time of the call (ms) | 0.75 | 1.46 | 30.3 | 4.0 |
| Share of the call the GPU is busy | 32 % | 16 % | 99 % | 79 % |

- With batches of 128 the GPU waits most of the time. The call costs the time to start its
  operations, which is longer on the cluster, where the CPU thread is slower. This is why the
  H100 is the slower of the two on small batches.
- With batches of 8,192 the GPU is busy. On the H100 the call takes 64 times the structures
  for 3 times the time.

So the graphs are built on a **large batch**, which is then **split** in batches of the size
asked (`Structures.split`). The split is cheap, 0.04 to 0.16 ms per batch, and the batches
are the ones a loader of the small size would give, with the same graphs.

Graphs built on a batch of each size then split in batches of 128, from memory:

| Size of the large batch | RTX 4060, radius graph | RTX 4060, full chain | H100, radius graph | H100, full chain |
|---:|---:|---:|---:|---:|
| 128 | 139,680 | 49,970 | 75,172 | 26,924 |
| 256 | 260,363 | 94,869 | 150,069 | 52,520 |
| 512 | 350,590 | 141,684 | 276,230 | 96,802 |
| 1,024 | 366,050 | 173,780 | 459,621 | 172,133 |
| 2,048 | 298,000 | 157,918 | 670,674 | 265,307 |
| 4,096 | 264,958 | 139,397 | 834,226 | 351,497 |
| 8,192 | 241,015 | 128,256 | 931,711 | 406,720 |

- The best size is not the same: 1,024 on the RTX 4060, which is busy from there and slows
  down with larger batches; 8,192 on the H100, which still gains.
- The gain over batches of 128 is 2.6 to 3.5 times on the RTX 4060, and 12 to 15 times on the
  H100.
- A large batch also reads faster from a file: a loader of 8,192 gives 3.5 to 9.5 times the
  structures per second of a loader of 128.

The best size depends on the GPU, on the CPU and on the transforms, so `auto_loader`
**measures** it. It builds the graphs of a sample at each size from the one asked to 8,192,
doubling, and keeps the smallest size within 10 % of the fastest, among the ones whose graphs
take at most 25 % of the free memory of the GPU. This takes most of the second `auto_loader`
needs to start. Without GPU the graphs are built on batches of the size asked: a CPU thread
does not gain from larger ones.

### Compiling does not help

`torch.compile` was tried on the function which builds the radius graph, in its three modes,
on batches of 128 on the GPU:

| | RTX 4060 | H100 | Compilation |
|---|---:|---:|---:|
| not compiled | 98,905 | 53,080 | |
| `torch.compile(f)` | 95,693 | 54,940 | 6 to 21 s |
| `torch.compile(f, dynamic=True)` | 86,843 | 50,875 | 2 to 9 s |
| `torch.compile(f, mode="reduce-overhead")` | 72,268 | 37,601 | 3 to 7 s |

The default mode is within 4 % either way, the two others are slower, and the edges are the
same. The sizes of the tensors of a graph depend on the data (the number of edges is not
known before they are found), which a compiler cannot fix in advance. `auto_loader` does not
compile anything.

The operations which make the CPU wait for a value from the GPU were also suspected. They
are a tenth of the time of a call: what costs is the number of operations, which large
batches pay once.

### What `auto_loader` does with this

| Choice | With a GPU | Without GPU |
|---|---|---|
| Dataset | the memory of the GPU if at most 20 % of what is free; else RAM if at most 50 % of what is available; else its file | RAM if at most 50 % of what is available; else its file |
| Workers | none from memory; one per core but one, 16 at most, for a file | one per core but one, 16 at most |
| Graphs | in the training process, on the GPU | in the workers, on the CPU |
| Batches the graphs are built on | measured, from the size asked to 8,192, then split | the size asked |
| Augmentations | before the graphs: on the GPU from memory, in the workers for a file | in the workers |
| Compilation | none | none |

It detects the CPU cores this process can use, the RAM available (no more than the limit of
the job, from its cgroup or from the memory asked to Slurm), and the GPU with its free memory.
The result, against each place of the dataset forced and against a plain loader:

| Loader | RTX 4060, radius graph | RTX 4060, full chain | H100, radius graph | H100, full chain |
|---|---:|---:|---:|---:|
| `auto_loader` | 338,893 | 179,125 | 882,202 | 387,454 |
| same, dataset forced in the memory of the GPU | 335,158 | 179,695 | 881,897 | 391,922 |
| same, dataset forced in RAM | 297,202 | 154,732 | 737,482 | 360,055 |
| same, dataset forced in its file, 16 workers | 90,009 | 89,395 | 133,566 | 121,990 |
| plain `DataLoader` of 128, graphs built per batch on the GPU | 49,227 | 32,166 | 31,023 | 17,647 |

On both machines it chose the memory of the GPU, with large batches of 512 to 1,024 on the
RTX 4060 and of 8,192 on the H100.

What is not measured: several GPUs, a GPU shared with a model during the measure, cells of
hundreds of atoms, and a machine where the file or the CPU alone is the right choice.

### Several datasets

A list of sources is loaded as one dataset in memory, on the GPU if it fits there by the rule
above, in RAM otherwise: a shuffled batch then mixes the structures of all of them, crystals
and molecules. A file cannot be shuffled with another one at this speed, so datasets which do
not fit together in memory are refused with an error. The rest is the table above.

With `counts`, every batch holds a fixed number of structures of each source. This is done by
the order of an epoch and by nothing else: the structures go by groups of `batch_size`,
`counts[i]` of them from the source `i`. The graphs are still built on large batches, which
are multiples of `batch_size`, and `Structures.split` cuts them in consecutive structures: the
batches given have the shares asked, at the speed of a plain loader. A sampler which drew each
batch apart would have built the graphs batch by batch.

### Subsets

A filter or the part of a split is a subset of a dataset: a list of rows. For a dataset read
from its file nothing is copied, the rows asked are mapped to rows of the file. In memory the
structures kept are copied once.

## Changing the configuration

`loader.config` says what was chosen, with the reason of each choice and what was measured:

```python
loader = auto_loader("mp-20", "train", batch_size=128, transform=graphs, shuffle=True)
print(loader.config)
```

```text
device cuda, dataset on the GPU, 0 workers
batches of 128, built by 512 on the device
- dataset on the GPU: 9 MB is at most 20% of the 7.1 GB free on it
- no worker: the batches are taken from memory
- graphs built on the GPU by batches of 512, split in batches of 128: the smallest within
  10 % of the fastest measured
- batches of 128: 133,874 structures/s, 16 MB
- batches of 256: 230,646 structures/s, 30 MB
- batches of 512: 254,415 structures/s, 59 MB
...
```

It is a `LoaderConfig`:

| Field | Values | |
|---|---|---|
| `device` | a `torch.device` | device of the batches |
| `storage` | `"gpu"`, `"ram"`, `"file"` | where the dataset lives |
| `batch_size` | | structures of the batches given |
| `loader_batch_size` | a multiple of `batch_size` | structures of the batches the graphs are built on |
| `num_workers` | | workers of the `DataLoader` |
| `graphs_on` | `"device"`, `"workers"` | where the transform runs |

To change a choice, copy the config with the field changed and give it back. Nothing is
detected nor measured when a config is given:

```python
from dataclasses import replace

# keep the memory of the GPU for the model
config = replace(loader.config, storage="ram")

# or: smaller batches for the graphs, to use less memory of the GPU
config = replace(loader.config, loader_batch_size=256)

loader = auto_loader("mp-20", "train", batch_size=128, transform=graphs, config=config)
```

A config can also be written whole, to use the same one on every machine:

```python
import torch
from torchms.data import LoaderConfig

config = LoaderConfig(
    device=torch.device("cuda"),
    storage="gpu",
    batch_size=128,
    loader_batch_size=2048,
    num_workers=0,
    graphs_on="device",
)
```

Two arguments of `auto_loader` change how the config is found, without giving one:

| Argument | |
|---|---|
| `device="cpu"` | no GPU is used: graphs built in the workers |
| `probe=False` | the size of the large batches is not measured: 8,192 on a GPU of 16 GB or more, 2,048 otherwise |

`detect_hardware()` returns what is detected, and `loader.hardware` what the loader saw.
`scripts/benchmark_auto.py` measures `auto_loader` against each place of the dataset on your
machine and your file.

## Setting a loader up by hand

`auto_loader` assembles four pieces, which can be used alone.

### A dataset in a `DataLoader`

An `HDF5Dataset` goes to the `DataLoader` of PyTorch as it is, with no sampler and no collate
function. A batch is read from the file in one call, in any order, and the `transform` of the
dataset is applied once to the whole batch, in the workers. `random_split`, `Subset` and the
other arguments of `DataLoader` work as usual.

```python
from torch.utils.data import DataLoader
from torchms.data import load_dataset

dataset = load_dataset("mp-20", "train", transform=augment)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
```

The loader batches `Structures` with `Structures.collate`, registered as the default collate
of PyTorch for this type. A `collate_fn` of your own receives a list of `Structures`: start it
with `Structures.collate(batch)`.

To build the graphs in the workers, on the CPU, put the graph transforms in the `transform` of
the dataset, after the augmentations:

```python
from torchms.data.augmentation import Compose

dataset = load_dataset("mp-20", "train", transform=Compose([augment, graphs]))
```

### `OnDevice`: the device, the graphs, the split

`OnDevice` wraps a loader. It moves each batch to a device, applies a transform there, in the
training process, and with `batch_size` splits the result in smaller batches. This is how
graphs are built on a GPU, on large batches:

```python
from torchms.data import OnDevice

loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=16)
loader = OnDevice(loader, "cuda", transform=graphs, batch_size=128)

for structures in loader:      # batches of 128, on the GPU, with their graphs
    ...
```

Without `transform` it only moves the batches, without `batch_size` it gives them as the
loader does. It is iterated once per epoch and has a length.

### `InMemoryDataset`: no file, no worker

`InMemoryDataset` holds a whole dataset as one batch, in RAM or on a GPU. It is indexed as an
`HDF5Dataset`, and `batches` gives the batches of an epoch without a `DataLoader`:

```python
from torchms.data import InMemoryDataset, OnDevice

dataset = InMemoryDataset.from_hdf5("my_structures.hdf5", device="cuda")

for epoch in range(10):
    batches = dataset.batches(8192, shuffle=True)          # the batches of one epoch
    for structures in OnDevice(batches, "cuda", transform=graphs, batch_size=128):
        ...
```

With `device="cpu"` the dataset is in RAM and `OnDevice` moves each batch. Its `transform`,
for the augmentations, runs on the device of the dataset.

### `split` and `select`

`Structures.split(batch_size)` cuts a batch in smaller batches of consecutive structures, with
their graph and what is built on it, the indices starting again at zero in each one.
`Structures.select(index)` gathers structures of a batch in any order, without graph.

```python
large = graphs(dataset[:8192].to("cuda"))     # the graphs of 8,192 structures, built once
batches = large.split(128)                    # 64 batches of 128, with their graphs
```

### Measuring a setup

| Script | Measures |
|---|---|
| `scripts/benchmark_auto.py` | `auto_loader` against each place of the dataset, and a plain loader |
| `scripts/benchmark_loader.py` | reading, each graph transform, the number of workers, workers against GPU |
| `scripts/benchmark_graph_gpu.py` | the size of the batches on a GPU, the operations of a call, `torch.compile` |
