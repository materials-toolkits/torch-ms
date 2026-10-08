# Loader and graph construction: measures

How fast a dataset is read, how fast its graphs are built, and where to build them: in the
workers of the loader, on the CPU, or on the GPU. The guide gives the rule in
[Where to build the graphs](../guide/graphs.md#where-to-build-the-graphs); this page gives the
numbers behind it, on two machines.

Measured on 2026-10-08 with `scripts/benchmark_loader.py`, at commit `99032bc`:

```bash
uv run python scripts/benchmark_loader.py datasets/mp_20/train.hdf5 --output report.md
uv run python scripts/benchmark_loader.py datasets/lematerial/unique.hdf5 --sample 10000
```

| | RTX 4060 machine | H100 machine |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB | one NVIDIA H100 NVL, 96 GB |
| CPU | 32 threads of a desktop, 24 used by torch | 32 cores of an Intel Xeon Gold 6430 (64 in the node) |
| One CPU thread, `AddRadiusGraph(5)` on MP-20 | 14,209 structures/s | 6,134 structures/s |
| torch | 2.11.0 with CUDA | 2.14.1 with CUDA |
| Where | local | CRIL cluster, partition `h100`, one job with 1 GPU and 32 CPUs |

| | MP-20 | LeMat-Bulk |
|---|---|---|
| Structures | 27,136 (train) | 10,000 drawn at random from `unique` |
| Atoms per structure | 10.4 on average, 20 at most | 9.3 on average, 222 at most |
| Edges per structure at 5 Å | 310 | 209 |

Batches of 128 structures unless said otherwise, median of 3 runs. All the numbers are
**structures per second**. The tables of the sections are the ones of MP-20, the ones of
LeMat-Bulk are [at the end](#lemat-bulk).

What these measures do not say:

- The two machines differ by their GPU, by their CPU and by their version of torch. A
  difference between them is not the effect of the GPU alone.
- Both datasets have small cells. Nothing here is about cells of hundreds of atoms.
- The GPU does nothing else during the measures. In a training it also runs the model.
- Batches larger than 128 are measured for the construction of a graph alone, not through
  the loader.

## What to do

| Hardware and task | Do | Measure |
|---|---|---|
| A GPU, and no reason below to do otherwise | Build on the GPU: `OnDevice(loader, "cuda", transform)` with 4 workers reading. It is the fastest way or within 10 % of it on both machines, with a quarter of the cores. | [Workers or GPU](#workers-or-gpu) |
| A GPU, 4 CPU cores or less for the loading | Build on the GPU. 1.3 to 5 times the rate of the workers. | [Workers or GPU](#workers-or-gpu) |
| Large batches, 512 structures and more | Build on the GPU. Its rate grows with the batch (3.2 times from 128 to 512 on the H100), the one of a CPU thread does not. | [Graph construction](#graph-construction) |
| The model keeps the GPU busy, and 8 to 16 CPU cores are free | Build in the workers: transform given to the dataset, `num_workers` 8 to 16. Same rate as the GPU way on the H100 machine, two thirds of it or more on the RTX 4060 machine, and the GPU is left to the model. | [Workers or GPU](#workers-or-gpu) |
| No GPU | Build in the workers, 8 to 16 of them. | [DataLoader](#dataloader) |
| Batches of a few structures | Build on the CPU: a GPU is 2.5 to 3 times slower than one thread on a single structure. | [Graph construction](#graph-construction) |
| No graph transform | No worker is needed below 11,000 (H100 machine) to 27,000 (RTX 4060 machine) structures/s, 4 workers give 42,000 to 58,000. | [DataLoader](#dataloader) |
| Workers, in any case | `persistent_workers=True`: workers take 0.15 s (fork) to 2.3 s (spawn) to start, at each epoch. | [DataLoader](#dataloader) |
| A dataset of one structure per item (`CifDataset`) | Build on the batch with `OnDevice`, on the CPU or the GPU, not per structure in the dataset: 4.5 to 7.5 times faster. | [Collate](#collate) |

Two results were not expected:

- **The H100 does not build the graphs of a batch of 128 faster than the RTX 4060**: 64,000
  structures per second against 97,000 for the radius graph at 5 Å. A graph is many small
  operations, and what seems to limit them is the CPU thread which starts them on the GPU,
  twice slower on the cluster. This cause was not isolated: the machines also differ by their
  version of torch. The H100 shows with large batches: 211,000 at 512 against 164,000.
- **A more powerful GPU does not make the GPU way win by more.** With 16 workers the two ways
  are equal on the H100 machine, while the GPU way stays 1.5 times ahead on the RTX 4060
  machine for MP-20. What decides is the speed of one CPU thread, the number of cores and
  the size of the batch.

Whatever the way, with 8 workers or more, the loading gives 13,000 to 43,000 structures per
second with their graphs: the whole training set of MP-20 in one or two seconds. It only
matters when the model trains at a comparable rate.

## Reading

Read from the file in one process, one batch per call, without transform nor loader. Sorted
batches are what the dataset needed before; shuffled batches are what a `DataLoader` with
`shuffle=True` asks now.

RTX 4060 machine:

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 28,163 | 18,379 | 16,255 |
| 128 | 67,845 | 31,369 | 29,007 |
| 512 | 105,454 | 43,376 | 40,733 |

H100 machine:

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 13,067 | 7,926 | 6,972 |
| 128 | 32,974 | 14,532 | 13,561 |
| 512 | 57,531 | 22,297 | 21,376 |

Reading indices in any order sorts them, reads, and puts the rows back in order: 4 to 12 %
slower than the same indices sorted by the caller. Random batches are 1.5 to 2.7 times slower
than consecutive ones, sorted or not, because HDF5 reads scattered rows one by one.

On the RTX 4060 machine these rates changed by a factor of two between runs of the script on
the same file, for a reason which was not found (its first table of
[LeMat-Bulk](#lemat-bulk) is such a run). The ratios between the columns did not change.

## Graph construction

Batches of 128 structures already in memory, on the device of the measure. The sizes are the
edges of the graph and the largest of the tuples of edges built, per structure. `full chain` is
`AddRadiusGraph(5) + AddTriplets(3) + AddQuadruplets(2.5)`.

RTX 4060 machine:

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

H100 machine:

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 32 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 158.3 |  | 11,325 | 21,065 | 67,585 |
| `AddRadiusGraph(5)` | 309.5 |  | 6,134 | 16,515 | 64,361 |
| `AddRadiusGraph(6)` | 522.7 |  | 3,795 | 14,392 | 61,868 |
| `AddRadiusGraph(5, geometry)` | 309.5 |  | 5,742 | 16,621 | 59,307 |
| `AddAtomicRadiusGraph()` | 173.3 |  | 3,140 | 7,807 | 38,146 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 127.2 |  | 2,846 | 7,866 | 41,921 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 158.3 |  | 5,364 | 14,037 | 49,346 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 309.5 | 525.4 | 3,713 | 9,628 | 34,822 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 309.5 | 525.4 | 4,344 | 11,380 | 40,373 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 309.5 | 2,595.4 | 2,606 | 8,290 | 38,115 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 309.5 | 353.1 | 3,731 | 10,749 | 30,224 |
| `full chain` | 309.5 | 525.4 | 3,234 | 8,843 | 23,560 |
| `full chain, geometry` | 309.5 | 525.4 | 3,117 | 7,986 | 23,289 |

- On one CPU thread, the cost of a radius graph follows its number of edges, which grows fast
  with the radius. On the H100 it hardly depends on the radius: 68,000 at 4 Å, 62,000 at 6 Å.
- `geometry=True` costs 6 to 10 %.
- Each step added to the graph costs. On one thread: about 25 to 30 % for the triplets within
  3 Å, 40 % for the line graph or the quadruplets. Triplets within 4 Å, 2,600 per
  structure, cost more than the graph.
- The GPU is 4.5 to 9 times faster than one CPU thread on the RTX 4060 machine, 6 to 16
  times on the H100 machine. It is 2 to 5.5 times faster than all the threads.
- All the threads are 2 to 4 times faster than one, far from the number of threads: a loader
  uses its cores better with one worker per core than with one process on all of them.

The same `AddRadiusGraph(5)` for different sizes of batch:

RTX 4060 machine:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 3,148 | 1,256 |
| 32 | 13,249 | 34,437 |
| 128 | 14,382 | 103,919 |
| 512 | 10,008 | 164,229 |

H100 machine:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 1,758 | 587 |
| 32 | 6,372 | 17,711 |
| 128 | 6,506 | 65,753 |
| 512 | 5,853 | 210,669 |

A batch of one structure is slow everywhere, and 2.5 to 3 times slower on the GPU than on a
CPU thread. On one thread the rate does not grow after 32 to 128 structures. On the GPU it
keeps growing: 1.6 times from 128 to 512 on the RTX 4060, 3.2 times on the H100, which is
then the faster of the two.

## Collate

On one thread, for `AddRadiusGraph(5, geometry) + AddTriplets(3)`.

RTX 4060 machine:

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 10,712 |
| graphs built per structure, then collate (`CifDataset`) | 1,958 |
| collate alone, structures with graphs | 87,229 |
| collate alone, structures without graph | 914,549 |

H100 machine:

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 5,056 |
| graphs built per structure, then collate (`CifDataset`) | 1,132 |
| collate alone, structures with graphs | 43,619 |
| collate alone, structures without graph | 383,671 |

Building the graphs on the batch is 4.5 to 5.5 times faster than building them per structure
and collating. The collate itself is not the cost: with graphs it handles 44,000 to 87,000
structures per second. What costs is to call the transforms 128 times on one structure
instead of once on 128. `HDF5Dataset` transforms whole batches by itself. A dataset which
returns one structure per item should leave the graphs to `OnDevice`, which transforms the
batch.

## DataLoader

`DataLoader(dataset, batch_size=128, shuffle=True, num_workers=...)`, the batches staying on
the CPU. With 0 workers the training process reads and transforms with all the threads of
torch; a worker uses one thread. `augmentations` is
`LatticeRotation + AtomTranslation + AtomPermutation`.

RTX 4060 machine:

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 26,806 | 24,698 | 43,618 | 58,091 | 59,963 | 55,502 |
| augmentations | 24,562 | 23,534 | 44,263 | 64,263 | 71,973 | 56,375 |
| `AddRadiusGraph(5)` | 9,374 | 7,735 | 14,176 | 24,597 | 31,303 | 29,892 |
| full chain, geometry | 5,732 | 4,330 | 8,053 | 14,714 | 19,800 | 17,876 |
| augmentations + full chain, geometry | 8,395 | 4,332 | 8,022 | 14,275 | 18,226 | 18,410 |

H100 machine:

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 11,113 | 12,464 | 24,216 | 41,883 | 54,683 | 51,584 |
| augmentations | 10,457 | 11,835 | 22,863 | 42,039 | 50,437 | 55,161 |
| `AddRadiusGraph(5)` | 7,135 | 4,063 | 7,481 | 13,608 | 24,205 | 31,730 |
| full chain, geometry | 4,453 | 2,307 | 4,365 | 8,119 | 14,440 | 19,076 |
| augmentations + full chain, geometry | 4,123 | 2,366 | 4,268 | 7,868 | 14,685 | 17,418 |

- **Without transform**, one worker is about as fast as none: the batch has to be sent from
  the worker to the training process. More workers are faster, up to 55,000 to 60,000 with 8.
- **Augmentations** cost nothing visible with workers.
- **With graph transforms**, one worker is slower than no worker, since the training process
  uses all its threads and a worker one. The rate then grows with the workers: up to 8 on
  the RTX 4060 machine, still at 16 on the H100 machine, whose cores are slower.
- On the RTX 4060 machine 16 workers are not faster than 8, with or without transform. Where
  this ceiling comes from has not been profiled.

The workers start at each epoch, unless the loader is created with `persistent_workers=True`.
Seconds before the first batch, with 8 workers and `AddRadiusGraph(5)`:

RTX 4060 machine:

| start method | first batch (s) |
|---|---:|
| fork | 0.16 |
| spawn | 0.87 |

H100 machine:

| start method | first batch (s) |
|---|---:|
| fork | 0.14 |
| spawn | 2.33 |

On the H100 machine, torch 2.14.1 prints "DataLoader worker is killed by signal: Aborted" when
a loader with workers is left before the end of its epoch, as the script does. It is an error
of the shutdown of the workers, reported as ignored, after the rate is measured.

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

RTX 4060 machine:

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

H100 machine:

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 6,689 | 8,956 |
| `AddRadiusGraph(5)` | 1 | 4,142 | 12,686 |
| `AddRadiusGraph(5)` | 2 | 7,503 | 21,311 |
| `AddRadiusGraph(5)` | 4 | 13,890 | 25,516 |
| `AddRadiusGraph(5)` | 8 | 23,789 | 24,844 |
| `AddRadiusGraph(5)` | 16 | 25,810 | 24,957 |
| full chain, geometry | 0 | 4,252 | 7,249 |
| full chain, geometry | 1 | 2,330 | 12,495 |
| full chain, geometry | 2 | 4,343 | 14,738 |
| full chain, geometry | 4 | 8,233 | 14,815 |
| full chain, geometry | 8 | 12,831 | 14,557 |
| full chain, geometry | 16 | 14,159 | 14,544 |

Rate of the GPU way divided by the rate of the workers way, on MP-20:

| Workers | RTX 4060 machine | H100 machine |
|---:|---|---|
| 1 | 3.2 to 5.2 | 3.1 to 5.4 |
| 2 | 2.8 to 3.2 | 2.8 to 3.4 |
| 4 | 1.6 to 1.8 | 1.8 |
| 8 | 1.5 | 1.0 to 1.1 |
| 16 | 1.5 | 1.0 |

- The GPU way reaches its rate with 2 to 4 workers reading, on both machines, and does not
  gain from more.
- The workers way gains from every worker up to 8 or 16. On the H100 machine it then equals
  the GPU way; on the RTX 4060 machine it stays at two thirds of it.
- The rates of the H100 machine are lower than the ones of the RTX 4060 machine in both
  columns. Its CPU threads are twice slower, and both ways use them: the workers to build
  the graphs, the GPU way to read and to drive the GPU.
- On LeMat-Bulk the two ways are closer still: with 8 workers and more the workers give 80 to
  115 % of the rate of the GPU ([tables](#lemat-bulk)).

This measures the loading alone, on a GPU which does nothing else. In a training the GPU also
runs the model, and building the graphs takes 1.3 to 2 ms of its time per batch of 128 for
the radius graph, 3.4 to 5.5 ms for the full chain. In the workers it takes none. Build on the
GPU when the loading limits the training, in the workers when the model does and the cores
are there. Augmentations are cheap on the CPU and stay in the transform of the dataset in
both cases.

## What was optimised

Three changes were made while measuring. Each one gives the same edges, in the same order,
as before. Structures per second on MP-20, batches of 128, RTX 4060 machine:

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

The same measures on 10,000 structures drawn at random from LeMat-Bulk (`--sample 10000`),
the same file on both machines.

### Reading

RTX 4060 machine:

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 13,686 | 9,573 | 8,685 |
| 128 | 34,636 | 19,534 | 18,163 |
| 512 | 56,686 | 50,447 | 48,064 |

H100 machine:

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 13,231 | 8,083 | 7,069 |
| 128 | 34,077 | 17,678 | 16,491 |
| 512 | 60,906 | 30,768 | 29,258 |

### Graph construction

RTX 4060 machine:

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

H100 machine:

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 32 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 113.7 |  | 14,507 | 22,859 | 68,233 |
| `AddRadiusGraph(5)` | 208.8 |  | 8,556 | 17,629 | 66,754 |
| `AddRadiusGraph(6)` | 365.2 |  | 5,900 | 19,670 | 64,524 |
| `AddRadiusGraph(5, geometry)` | 208.8 |  | 8,647 | 20,014 | 61,391 |
| `AddAtomicRadiusGraph()` | 164.0 |  | 3,501 | 8,449 | 38,776 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 111.1 |  | 3,976 | 9,028 | 43,776 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 113.7 |  | 7,935 | 17,500 | 51,012 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 208.8 | 268.8 | 5,881 | 11,691 | 35,888 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 208.8 | 268.8 | 6,842 | 13,626 | 41,848 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 208.8 | 1,541.0 | 4,131 | 10,274 | 40,209 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 208.8 | 157.2 | 5,817 | 10,627 | 31,056 |
| `full chain` | 208.8 | 268.8 | 5,018 | 9,155 | 24,330 |
| `full chain, geometry` | 208.8 | 268.8 | 4,751 | 8,456 | 23,645 |

`AddRadiusGraph(5)` for different sizes of batch:

RTX 4060 machine:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 3,268 | 1,249 |
| 32 | 16,855 | 35,127 |
| 128 | 18,954 | 114,879 |
| 512 | 12,405 | 225,227 |

H100 machine:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 1,883 | 579 |
| 32 | 7,996 | 17,476 |
| 128 | 7,934 | 66,419 |
| 512 | 7,348 | 222,296 |

### Collate

RTX 4060 machine:

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 15,987 |
| graphs built per structure, then collate (`CifDataset`) | 2,152 |
| collate alone, structures with graphs | 101,865 |
| collate alone, structures without graph | 919,823 |

H100 machine:

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 7,502 |
| graphs built per structure, then collate (`CifDataset`) | 1,233 |
| collate alone, structures with graphs | 49,506 |
| collate alone, structures without graph | 372,555 |

### DataLoader

RTX 4060 machine:

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 30,944 | 27,321 | 39,408 | 52,464 | 65,601 | 60,844 |
| augmentations | 16,834 | 26,775 | 40,274 | 50,620 | 59,483 | 47,135 |
| `AddRadiusGraph(5)` | 16,400 | 10,267 | 18,445 | 29,233 | 37,125 | 34,093 |
| full chain, geometry | 10,273 | 6,124 | 11,119 | 20,086 | 27,226 | 28,219 |
| augmentations + full chain, geometry | 10,002 | 6,175 | 11,129 | 18,992 | 27,509 | 31,720 |

H100 machine:

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 13,678 | 14,800 | 27,765 | 41,964 | 51,224 | 46,476 |
| augmentations | 12,878 | 14,350 | 27,290 | 43,536 | 46,891 | 47,643 |
| `AddRadiusGraph(5)` | 8,005 | 5,491 | 10,145 | 18,186 | 32,426 | 31,125 |
| full chain, geometry | 5,132 | 3,376 | 6,289 | 11,130 | 18,837 | 20,002 |
| augmentations + full chain, geometry | 5,040 | 3,297 | 6,170 | 11,367 | 18,613 | 20,386 |

Seconds before the first batch, with 8 workers and `AddRadiusGraph(5)`:

RTX 4060 machine:

| start method | first batch (s) |
|---|---:|
| fork | 0.17 |
| spawn | 0.91 |

H100 machine:

| start method | first batch (s) |
|---|---:|
| fork | 0.14 |
| spawn | 2.33 |

### Workers or GPU

RTX 4060 machine:

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

H100 machine:

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 7,612 | 10,324 |
| `AddRadiusGraph(5)` | 1 | 5,408 | 15,130 |
| `AddRadiusGraph(5)` | 2 | 10,116 | 23,039 |
| `AddRadiusGraph(5)` | 4 | 17,731 | 25,082 |
| `AddRadiusGraph(5)` | 8 | 26,707 | 24,002 |
| `AddRadiusGraph(5)` | 16 | 26,440 | 23,311 |
| full chain, geometry | 0 | 4,837 | 8,042 |
| full chain, geometry | 1 | 3,311 | 12,893 |
| full chain, geometry | 2 | 6,268 | 14,544 |
| full chain, geometry | 4 | 11,547 | 14,513 |
| full chain, geometry | 8 | 15,258 | 14,324 |
| full chain, geometry | 16 | 15,575 | 14,077 |

The graphs have fewer edges than on MP-20 (209 against 310 at 5 Å) and are built faster. The
order of the transforms and the effect of the workers are the same. With 8 workers and more,
building in the workers is as fast as building on the GPU on the H100 machine, and nearly as
fast on the RTX 4060 machine.
