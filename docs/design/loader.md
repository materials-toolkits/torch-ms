# Loader and graph construction: measures

How fast a dataset is read, how fast its graphs are built, and where to build them: in the
workers of the loader, on the CPU, or on the GPU. The guide gives the rule in
[Where to build the graphs](../guide/graphs.md#where-to-build-the-graphs); this page gives the
numbers behind it.

Measured on 2026-10-08 with `scripts/benchmark_loader.py`:

```bash
uv run python scripts/benchmark_loader.py datasets/mp_20/train.hdf5 --output report.md
uv run python scripts/benchmark_loader.py datasets/lematerial/unique.hdf5 --sample 10000
```

| | MP-20 | LeMat-Bulk |
|---|---|---|
| Structures | 27,136 (train) | 10,000 drawn at random from `unique` |
| Atoms per structure | 10.4 on average, 20 at most | 9.3 on average, 222 at most |
| Edges per structure at 5 Å | 310 | 209 |

- 32 CPU threads (24 used by torch), NVIDIA GeForce RTX 4060 with 8 GB;
- torch 2.11.0 with CUDA, Python 3.10;
- batches of 128 structures unless said otherwise, median of 3 runs.

All the numbers are **structures per second**. They depend on the machine, on the dataset
and on the transforms: run the script on yours, it takes a few minutes. The tables of this
page are the ones of MP-20, the ones of LeMat-Bulk are [at the end](#lemat-bulk).

Both datasets have small cells. Nothing here says how datasets with hundreds of atoms per
cell behave.

## What to do

| Hardware and task | Do | Measure |
|---|---|---|
| No graph transform | 0 to 4 workers. Reading gives 27,000 structures/s without worker, 58,000 with 4. | [DataLoader](#dataloader) |
| Graphs, 8 CPU cores or more for the loading, the model keeps the GPU busy | Build in the workers: transform given to the dataset, `num_workers=8`. 65 to 95 % of the rate of the GPU, which is left to the model. | [Workers or GPU](#workers-or-gpu) |
| Graphs, 4 CPU cores or less | Build on the GPU with `OnDevice`. 1.4 to 5 times the rate of the workers. | [Workers or GPU](#workers-or-gpu) |
| The GPU waits for batches | Build on the GPU with `OnDevice` and 2 to 4 workers to read: the fastest way measured. | [Workers or GPU](#workers-or-gpu) |
| No GPU | Build in the workers, 8 of them. 16 bring little or nothing more. | [DataLoader](#dataloader) |
| Batches of 512 structures and more | Build on the GPU: its rate grows with the batch, the one of a CPU thread decreases after 128. | [Graph construction](#graph-construction) |
| Batches of a few structures | Build on the CPU: a GPU is slower than one thread on a single structure. | [Graph construction](#graph-construction) |
| Any | `persistent_workers=True`: workers take 0.2 s (fork) to 0.9 s (spawn) to start, at each epoch. | [DataLoader](#dataloader) |
| A dataset of one structure per item (`CifDataset`) | Build on the batch with `OnDevice`, not per structure in the dataset: 5 to 7 times faster. | [Collate](#collate) |

Whatever the way, the loading gives 17,000 to 43,000 structures per second with their graphs
on this machine: the whole training set of MP-20 in about one second. It only matters when
the model trains at a comparable rate.

## Reading

Read from the file in one process, one batch per call, without transform nor loader. Sorted
batches are what the dataset needed before; shuffled batches are what a `DataLoader` with
`shuffle=True` asks now.

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 28,163 | 18,379 | 16,255 |
| 128 | 67,845 | 31,369 | 29,007 |
| 512 | 105,454 | 43,376 | 40,733 |

Reading indices in any order sorts them, reads, and puts the rows back in order: 6 to 12 %
slower than the same indices sorted by the caller. Random batches are 1.5 to 2.6 times slower
than consecutive ones, sorted or not, because HDF5 reads scattered rows one by one.

These rates changed by a factor of two between runs of the script on the same file, for a
reason which was not found (the first table of [LeMat-Bulk](#lemat-bulk) is such a run). The
ratios between the columns did not change.

## Graph construction

Batches of 128 structures already in memory, on the device of the measure. The sizes are the
edges of the graph and the largest of the tuples of edges built, per structure. `full chain` is
`AddRadiusGraph(5) + AddTriplets(3) + AddQuadruplets(2.5)`.

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 24 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 158.3 |  | 25,854 | 46,105 | 116,867 |
| `AddRadiusGraph(5)` | 309.5 |  | 14,209 | 33,375 | 96,654 |
| `AddRadiusGraph(6)` | 522.7 |  | 8,626 | 34,141 | 76,068 |
| `AddRadiusGraph(5, geometry)` | 309.5 |  | 12,862 | 38,838 | 87,284 |
| `AddAtomicRadiusGraph()` | 173.3 |  | 6,763 | 17,515 | 55,463 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 127.2 |  | 6,256 | 17,904 | 53,355 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 158.3 |  | 12,359 | 36,065 | 79,808 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 309.5 | 525.4 | 8,477 | 23,901 | 53,544 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 309.5 | 525.4 | 10,468 | 29,580 | 62,587 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 309.5 | 2,595.4 | 5,905 | 21,302 | 45,827 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 309.5 | 353.1 | 8,351 | 22,563 | 50,942 |
| `full chain` | 309.5 | 525.4 | 7,127 | 18,298 | 39,083 |
| `full chain, geometry` | 309.5 | 525.4 | 6,421 | 15,186 | 37,157 |

- The cost of a radius graph follows its number of edges, which grows fast with the radius.
- `geometry=True` costs about 10 %, on one thread and on the GPU.
- Each step added to the graph costs: 26 % for the triplets within 3 Å, 40 % for the line
  graph or the quadruplets, on one thread. Triplets within 4 Å, 2,600 per structure, cost
  more than the graph.
- The GPU is 4.5 to 9 times faster than one CPU thread, and 2 to 3 times faster than 24.
- 24 threads are 2 to 4 times faster than one, far from 24 times: a loader uses its cores
  better with one worker per core than with one process on all of them.

The same `AddRadiusGraph(5)` for different sizes of batch:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 3,148 | 1,256 |
| 32 | 13,249 | 34,437 |
| 128 | 14,382 | 103,919 |
| 512 | 10,008 | 164,229 |

A batch of one structure is slow everywhere, and slower on the GPU than on a CPU thread. On
one thread the rate is the best around 128 structures. On the GPU it keeps growing: 60 % more
at 512 than at 128.

## Collate

On one thread, for `AddRadiusGraph(5, geometry) + AddTriplets(3)`.

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 10,712 |
| graphs built per structure, then collate (`CifDataset`) | 1,958 |
| collate alone, structures with graphs | 87,229 |
| collate alone, structures without graph | 914,549 |

Building the graphs on the batch is 5.5 times faster than building them per structure and
collating. The collate itself is not the cost: with graphs it handles 87,000 structures per
second. What costs is to call the transforms 128 times on one structure instead of once on
128. `HDF5Dataset` transforms whole batches by itself. A dataset which returns one structure
per item should leave the graphs to `OnDevice`, which transforms the batch.

## DataLoader

`DataLoader(dataset, batch_size=128, shuffle=True, num_workers=...)`, the batches staying on
the CPU. With 0 workers the training process reads and transforms with all the threads of
torch; a worker uses one thread. `augmentations` is
`LatticeRotation + AtomTranslation + AtomPermutation`.

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 26,806 | 24,698 | 43,618 | 58,091 | 59,963 | 55,502 |
| augmentations | 24,562 | 23,534 | 44,263 | 64,263 | 71,973 | 56,375 |
| `AddRadiusGraph(5)` | 9,374 | 7,735 | 14,176 | 24,597 | 31,303 | 29,892 |
| full chain, geometry | 5,732 | 4,330 | 8,053 | 14,714 | 19,800 | 17,876 |
| augmentations + full chain, geometry | 8,395 | 4,332 | 8,022 | 14,275 | 18,226 | 18,410 |

- **Without transform**, one worker is not faster than none: the batch has to be sent from
  the worker to the training process. 2 workers and more are faster, up to 60,000.
- **Augmentations** cost nothing visible with workers.
- **With graph transforms**, one worker is slower than no worker, since the training process
  uses 24 threads and a worker one. The rate grows with the workers up to 8, where it is 3.3
  to 3.5 times the rate without worker.
- **16 workers are not faster than 8**, with or without transform. A radius graph in 8
  workers gives half of the rate without transform. Where this ceiling comes from has not
  been profiled.

The workers start at each epoch, unless the loader is created with `persistent_workers=True`.
Seconds before the first batch, with 8 workers and `AddRadiusGraph(5)`:

| start method | first batch (s) |
|---|---:|
| fork | 0.16 |
| spawn | 0.87 |

## Workers or GPU

Structures arriving on the GPU with their graphs, per second. Either the transform of the
dataset builds them in the workers and the batch is moved, or the dataset has no transform
and `OnDevice` builds them on the GPU with the same transforms:

```python
graphs = Compose([AddRadiusGraph(5.0), AddTriplets(3.0)])

# in the workers
dataset = load_dataset("mp-20", "train", transform=graphs)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
loader = OnDevice(loader, "cuda")

# on the GPU
dataset = load_dataset("mp-20", "train")
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
loader = OnDevice(loader, "cuda", transform=graphs)
```

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 8,342 | 11,094 |
| `AddRadiusGraph(5)` | 1 | 7,598 | 24,353 |
| `AddRadiusGraph(5)` | 2 | 13,726 | 38,862 |
| `AddRadiusGraph(5)` | 4 | 23,868 | 37,808 |
| `AddRadiusGraph(5)` | 8 | 27,916 | 43,234 |
| `AddRadiusGraph(5)` | 16 | 27,719 | 41,450 |
| full chain, geometry | 0 | 5,297 | 14,663 |
| full chain, geometry | 1 | 4,311 | 22,354 |
| full chain, geometry | 2 | 8,069 | 25,907 |
| full chain, geometry | 4 | 14,035 | 25,484 |
| full chain, geometry | 8 | 16,773 | 25,398 |
| full chain, geometry | 16 | 16,943 | 25,521 |

- Built on the GPU, the graphs arrive faster for every number of workers: 3 to 5 times with
  one worker, 1.6 to 1.8 times with 4, 1.5 times with 8 and 16.
- The GPU way reaches its rate with 2 workers reading. The workers way needs 8.
- On LeMat-Bulk the two ways are closer: with 8 and 16 workers, the workers give 80 to 95 %
  of the rate of the GPU ([tables](#lemat-bulk)).

This measures the loading alone, on a GPU which does nothing else. In a training the GPU also
runs the model, and building the graphs takes 1.3 ms of its time per batch of 128 for the
radius graph, 3.4 ms for the full chain. In the workers it takes none. Build on the GPU when
the loading limits the training, in the workers when the model does. Augmentations are cheap
on the CPU and stay in the transform of the dataset in both cases.

## What was optimised

Three changes were made while measuring. Each one gives the same edges, in the same order,
as before. Structures per second on MP-20, batches of 128:

| Transform | Device | Before | After |
|---|---|---:|---:|
| `AddRadiusGraph(5)` | CPU, 1 thread | 1,844 | 14,209 |
| `AddRadiusGraph(5)` | GPU | 21,228 | 96,654 |
| `AddAtomicRadiusGraph()` | CPU, 1 thread | 252 | 6,763 |
| `AddAtomicRadiusGraph()` | GPU | 4,525 | 55,463 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | CPU, 1 thread | 850 | 8,477 |
| full chain, geometry | CPU, 1 thread | 829 | 6,421 |
| full chain, geometry | GPU | 15,997 | 37,157 |
| full chain, geometry, loader with 8 workers | CPU | 4,722 | 19,800 |

- **Neighbour search** (`cutoff`). Every atom of the cell was tried against every atom of a
  supercell: 7,462 pairs per structure at 5 Å for 305 edges. For each pair of atoms, only
  the periodic images which can be within the radius are now listed, from the heights of the
  cell (`periodic_pairs`): 899 pairs.
- **Chemical bonds** (`atomic_radius`). The search went up to twice the largest atomic radius
  of the batch, 11.9 Å on MP-20 with the default scaling. It now stops at the sum of the
  radii of each pair. The time of a structure no longer depends on the other structures of
  its batch.
- **Reverse of an edge** (`reverse_edges`), needed by the line graph, the quadruplets and the
  chemical bonds. It compared rows with `torch.unique`, which took longer than building the
  graph. It now compares one integer per edge.

## LeMat-Bulk

The same measures on 10,000 structures drawn at random from LeMat-Bulk (`--sample 10000`).

Reading:

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 13,686 | 9,573 | 8,685 |
| 128 | 34,636 | 19,534 | 18,163 |
| 512 | 56,686 | 50,447 | 48,064 |

Graph construction:

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 24 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 113.7 |  | 34,118 | 54,280 | 124,970 |
| `AddRadiusGraph(5)` | 208.8 |  | 20,426 | 42,783 | 109,869 |
| `AddRadiusGraph(6)` | 365.2 |  | 12,716 | 38,563 | 92,327 |
| `AddRadiusGraph(5, geometry)` | 208.8 |  | 18,523 | 39,770 | 103,132 |
| `AddAtomicRadiusGraph()` | 164.0 |  | 7,352 | 17,861 | 58,310 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 111.1 |  | 9,139 | 21,962 | 67,595 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 113.7 |  | 17,104 | 35,224 | 85,700 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 208.8 | 268.8 | 12,533 | 25,432 | 60,579 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 208.8 | 268.8 | 15,724 | 31,867 | 72,692 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 208.8 | 1,541.0 | 9,961 | 25,412 | 57,580 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 208.8 | 157.2 | 13,043 | 25,117 | 53,066 |
| `full chain` | 208.8 | 268.8 | 10,968 | 20,962 | 41,216 |
| `full chain, geometry` | 208.8 | 268.8 | 10,229 | 18,192 | 41,034 |

`AddRadiusGraph(5)` for different sizes of batch:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 3,268 | 1,249 |
| 32 | 16,855 | 35,127 |
| 128 | 18,954 | 114,879 |
| 512 | 12,405 | 225,227 |

Collate:

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 15,987 |
| graphs built per structure, then collate (`CifDataset`) | 2,152 |
| collate alone, structures with graphs | 101,865 |
| collate alone, structures without graph | 919,823 |

DataLoader:

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 30,944 | 27,321 | 39,408 | 52,464 | 65,601 | 60,844 |
| augmentations | 16,834 | 26,775 | 40,274 | 50,620 | 59,483 | 47,135 |
| `AddRadiusGraph(5)` | 16,400 | 10,267 | 18,445 | 29,233 | 37,125 | 34,093 |
| full chain, geometry | 10,273 | 6,124 | 11,119 | 20,086 | 27,226 | 28,219 |
| augmentations + full chain, geometry | 10,002 | 6,175 | 11,129 | 18,992 | 27,509 | 31,720 |

| start method | first batch (s) |
|---|---:|
| fork | 0.17 |
| spawn | 0.91 |

Workers or GPU:

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 10,129 | 12,580 |
| `AddRadiusGraph(5)` | 1 | 10,262 | 27,464 |
| `AddRadiusGraph(5)` | 2 | 17,516 | 37,318 |
| `AddRadiusGraph(5)` | 4 | 31,601 | 43,996 |
| `AddRadiusGraph(5)` | 8 | 35,909 | 37,859 |
| `AddRadiusGraph(5)` | 16 | 34,765 | 40,542 |
| full chain, geometry | 0 | 6,427 | 16,283 |
| full chain, geometry | 1 | 6,112 | 23,831 |
| full chain, geometry | 2 | 11,475 | 26,267 |
| full chain, geometry | 4 | 18,890 | 26,368 |
| full chain, geometry | 8 | 21,006 | 26,169 |
| full chain, geometry | 16 | 23,492 | 24,840 |

The graphs have fewer edges than on MP-20 (209 against 310 at 5 Å) and are built faster.
The order of the transforms and the effect of the workers are the same. The difference is
the last table: with 8 workers and more, building in the workers is nearly as fast as
building on the GPU.
