# Loader and graph construction on the CRIL cluster: H100 and RTX 8000

2026-10-08. Benchmark of `scripts/benchmark_loader.py` on one H100 and on one RTX 8000, to
compare with the local RTX 4060 and decide where graphs should be built: in the workers of
the loader or on the GPU. The values of the RTX 4060 and of the H100 are in
`docs/design/loader.md`; this report holds the run and the values of the RTX 8000.

## Run

- **Commit**: `99032bcad2600b7889d8b58e13cad7a88e082745` (branch `dataloader`), working tree
  clean. Deployed from a `git archive` of this commit, not from the project directory, which
  holds 36 GB of datasets that `cluster deploy` would have uploaded.
- **Environment**: Python 3.10.20, torch 2.14.1+cu130. The local machine has torch
  2.11.0+cu130: `cluster deploy` resolves the versions from `pyproject.toml`, not from
  `uv.lock`.
- **Data**: `datasets/mp_20/train.hdf5` (27,136 structures) and `lematerial_10k.hdf5`, the
  10,000 structures of `datasets/lematerial/unique.hdf5` drawn by `--sample 10000`, written
  once locally and copied, so the three machines read the same file.

| Job | Partition | Node | GPU | CPU | Resources | Elapsed | State |
|---|---|---|---|---|---|---|---|
| 13158398 `bench_h100` | `h100` | nodeGcor1 | NVIDIA H100 NVL, 96 GB | Intel Xeon Gold 6430, 64 cores | 1 GPU, 32 CPUs, 64 GB | 5 min 08 s | COMPLETED |
| 13158400 `bench_rtx8000` | `quad_rtx_8000` | nodeGcal1 | Quadro RTX 8000, 46 GB | Intel Xeon Gold 6226R, 32 cores | 1 GPU, 32 CPUs, 64 GB | 6 min 44 s | COMPLETED |
| 13158399 `bench_rtx8000` | `quad_rtx_8000` | nodeGcal1 | | | | 0 s | FAILED |

Command of each job, in `~/torchms`:

```bash
env GIT_SHA=99032bc bash run_benchmark.sh h100          # 13158398
env GIT_SHA=99032bc bash run_benchmark_rtx.sh rtx8000   # 13158400
```

with, in the script (`PYTHONPATH=src`, `set -euo pipefail`):

```bash
python scripts/benchmark_loader.py datasets/mp_20/train.hdf5 --output results/${tag}_mp_20.md
python scripts/benchmark_loader.py datasets/lematerial_10k.hdf5 --output results/${tag}_lematerial.md
```

Job 13158399 failed before any measure: the script printed the CPU with
`lscpu | grep "Model name"`, which matches nothing in the French locale of the node, and
`pipefail` stopped it. The second script uses `LC_ALL=C` and `|| true`.

## Check of the logs

Both jobs reached `benchmark done`, with the five parts measured twice (10 lines
`measuring: ...` each) and the four reports written with every cell filled.

Each log holds one error, in the part `loader`:

```text
terminate called without an active exception          (6 or 7 times)
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ ...>
RuntimeError: DataLoader worker (pid ...) is killed by signal: Aborted.
```

It is raised when the workers of a loader are shut down after the script left its loop
before the end of the epoch, which it does once it has counted its batches. The rate is
computed before. It did not happen locally with torch 2.11.0, and was not investigated
further: the cause is a guess (shutdown of the workers in torch 2.14.1).

No other line matches `error|traceback|exception|out of memory|failed|nan`.

## Results

Structures per second, MP-20, batches of 128.

| | RTX 4060 (local) | H100 | RTX 8000 |
|---|---:|---:|---:|
| One CPU thread, `AddRadiusGraph(5)` | 14,209 | 6,134 | 4,352 |
| GPU, `AddRadiusGraph(5)`, batches of 128 | 96,654 | 64,361 | 50,673 |
| GPU, `AddRadiusGraph(5)`, batches of 512 | 164,229 | 210,669 | 139,639 |
| GPU, full chain with geometry | 37,157 | 23,289 | 17,889 |
| Loader, no transform, 8 workers | 59,963 | 54,683 | 30,514 |
| Radius graph on the GPU, built in 16 workers | 27,719 | 25,810 | 19,876 |
| Radius graph on the GPU, built on the GPU, 4 workers | 37,808 | 25,516 | 19,370 |
| Full chain on the GPU, built in 16 workers | 16,943 | 14,159 | 9,664 |
| Full chain on the GPU, built on the GPU, 4 workers | 25,484 | 14,815 | 10,722 |

- **The cluster GPUs do not build the graphs of a batch of 128 faster than the RTX 4060.**
  They are 1.5 (H100) and 1.9 (RTX 8000) times slower for the radius graph. The CPU thread
  is 2.3 and 3.3 times slower on these nodes, and a graph is many small operations started
  from that thread. This explanation fits the numbers but was not isolated: the version of
  torch differs too.
- **The H100 is ahead with batches of 512**: 1.3 times the RTX 4060. Its rate is multiplied
  by 3.2 from 128 to 512, against 1.6 for the RTX 4060 and 2.7 for the RTX 8000.
- **On both nodes, 16 workers building the graphs equal the GPU building them** (ratio 0.97
  to 1.15). On the local machine the GPU stays 1.5 times ahead.
- **With 4 workers or less the GPU is ahead everywhere**, 1.8 to 6 times on the nodes.
- The order of the transforms, the effect of the size of the batch and of the workers are
  the same on the three machines.

What is not measured: batches larger than 128 through the loader, a GPU shared with a
model, cells of hundreds of atoms, and the same version of torch on the three machines.

## Reports of the RTX 8000

### MP-20

#### Setting

- file: `datasets/mp_20/train.hdf5`, 27,136 structures, 10.4 atoms per structure on average (max 20)
- CPU: x86_64, 32 threads, 32 used by torch
- GPU: Quadro RTX 8000
- torch 2.14.1+cu130, Python 3.10.20
- batches of 128 structures, 16 batches per measure, median of 3 runs

#### Reading

Structures per second read from the file, one batch per call, in one process. Sorted batches are the ones the dataset needed before; shuffled batches are what a DataLoader with `shuffle=True` asks now.

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 8,275 | 5,932 | 5,510 |
| 128 | 22,059 | 12,548 | 12,871 |
| 512 | 46,963 | 17,953 | 17,392 |

#### Graph construction

Structures per second for batches of 128 structures already in memory, on the device of the measure. The sizes are the edges of the graph and the largest of the tuples of edges built (pairs, triplets or quadruplets), per structure.

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 32 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 158.3 |  | 8,158 | 17,235 | 52,891 |
| `AddRadiusGraph(5)` | 309.5 |  | 4,352 | 13,823 | 50,673 |
| `AddRadiusGraph(6)` | 522.7 |  | 2,582 | 11,291 | 48,368 |
| `AddRadiusGraph(5, geometry)` | 309.5 |  | 4,192 | 14,705 | 47,163 |
| `AddAtomicRadiusGraph()` | 173.3 |  | 2,308 | 6,979 | 30,452 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 127.2 |  | 2,036 | 6,940 | 33,177 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 158.3 |  | 3,773 | 11,608 | 39,392 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 309.5 | 525.4 | 2,976 | 9,508 | 27,732 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 309.5 | 525.4 | 3,429 | 11,677 | 32,028 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 309.5 | 2,595.4 | 1,877 | 7,784 | 28,974 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 309.5 | 353.1 | 3,062 | 8,791 | 23,506 |
| `full chain` | 309.5 | 525.4 | 2,356 | 6,365 | 18,593 |
| `full chain, geometry` | 309.5 | 525.4 | 2,231 | 6,447 | 17,889 |

Structures per second of `AddRadiusGraph(5)` for different sizes of batch:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 972 | 441 |
| 32 | 4,476 | 13,778 |
| 128 | 4,136 | 51,560 |
| 512 | 4,223 | 139,639 |

#### Collate

Structures per second on one thread, for `AddRadiusGraph(5, geometry) + AddTriplets(3)` and batches of 128 structures.

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 3,531 |
| graphs built per structure, then collate (`CifDataset`) | 641 |
| collate alone, structures with graphs | 33,447 |
| collate alone, structures without graph | 325,200 |

#### DataLoader

Structures per second of `DataLoader(dataset, batch_size=128, shuffle=True, num_workers=...)`, the batches staying on the CPU. With 0 workers the main process reads and transforms, with all the threads of torch; a worker uses one thread.

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 11,326 | 10,282 | 17,476 | 25,962 | 30,514 | 35,321 |
| augmentations | 9,597 | 9,668 | 17,193 | 27,355 | 33,463 | 36,307 |
| `AddRadiusGraph(5)` | 5,531 | 2,722 | 5,229 | 9,436 | 17,316 | 21,433 |
| full chain, geometry | 3,592 | 1,648 | 3,081 | 5,607 | 9,835 | 11,741 |
| augmentations + full chain, geometry | 3,165 | 1,563 | 3,160 | 5,358 | 9,533 | 11,621 |

Seconds before the first batch with 8 workers and `AddRadiusGraph(5)`, paid at each epoch without `persistent_workers=True`:

| start method | first batch (s) |
|---|---:|
| fork | 0.26 |
| spawn | 2.60 |

#### Workers or GPU

Structures per second arriving on the GPU with their graphs: built by the transform of the dataset in the workers then moved (`OnDevice(loader, "cuda")`), or read without transform, moved, and built on the GPU by the same transforms (`OnDevice(loader, "cuda", transform)`).

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 5,345 | 8,698 |
| `AddRadiusGraph(5)` | 1 | 2,670 | 9,898 |
| `AddRadiusGraph(5)` | 2 | 5,168 | 14,824 |
| `AddRadiusGraph(5)` | 4 | 9,437 | 19,370 |
| `AddRadiusGraph(5)` | 8 | 17,050 | 19,192 |
| `AddRadiusGraph(5)` | 16 | 19,876 | 19,928 |
| full chain, geometry | 0 | 3,327 | 6,297 |
| full chain, geometry | 1 | 1,541 | 9,684 |
| full chain, geometry | 2 | 2,886 | 11,257 |
| full chain, geometry | 4 | 5,692 | 10,722 |
| full chain, geometry | 8 | 8,382 | 10,852 |
| full chain, geometry | 16 | 9,664 | 10,972 |

### LeMat-Bulk, 10,000 structures

#### Setting

- file: `datasets/lematerial_10k.hdf5`, 10,000 structures, 9.3 atoms per structure on average (max 222)
- CPU: x86_64, 32 threads, 32 used by torch
- GPU: Quadro RTX 8000
- torch 2.14.1+cu130, Python 3.10.20
- batches of 128 structures, 16 batches per measure, median of 3 runs

#### Reading

Structures per second read from the file, one batch per call, in one process. Sorted batches are the ones the dataset needed before; shuffled batches are what a DataLoader with `shuffle=True` asks now.

| batch size | consecutive | random, sorted | random, shuffled |
|---|---:|---:|---:|
| 32 | 9,646 | 7,105 | 6,423 |
| 128 | 25,764 | 16,371 | 15,538 |
| 512 | 51,242 | 26,505 | 25,369 |

#### Graph construction

Structures per second for batches of 128 structures already in memory, on the device of the measure. The sizes are the edges of the graph and the largest of the tuples of edges built (pairs, triplets or quadruplets), per structure.

| transform | edges / structure | largest tuples / structure | CPU, 1 thread | CPU, 32 threads | GPU |
|---|---:|---:|---:|---:|---:|
| `AddRadiusGraph(4)` | 113.7 |  | 10,640 | 19,108 | 54,979 |
| `AddRadiusGraph(5)` | 208.8 |  | 5,979 | 14,231 | 54,009 |
| `AddRadiusGraph(6)` | 365.2 |  | 3,845 | 15,045 | 52,052 |
| `AddRadiusGraph(5, geometry)` | 208.8 |  | 6,059 | 16,249 | 50,034 |
| `AddAtomicRadiusGraph()` | 164.0 |  | 2,645 | 8,163 | 31,588 |
| `AddRadiusGraph(6) + KeepNearestNeighbors(12)` | 111.1 |  | 2,975 | 8,411 | 35,571 |
| `AddRadiusGraph(5) + KeepRadius(4)` | 113.7 |  | 5,628 | 13,869 | 41,501 |
| `AddRadiusGraph(5) + AddLineGraph(3)` | 208.8 | 268.8 | 4,284 | 9,603 | 29,265 |
| `AddRadiusGraph(5) + AddTriplets(3)` | 208.8 | 268.8 | 4,905 | 11,893 | 33,852 |
| `AddRadiusGraph(5) + AddTriplets(4)` | 208.8 | 1,541.0 | 2,775 | 8,546 | 32,233 |
| `AddRadiusGraph(5) + AddQuadruplets(2.5)` | 208.8 | 157.2 | 4,334 | 9,160 | 24,852 |
| `full chain` | 208.8 | 268.8 | 3,679 | 7,730 | 19,623 |
| `full chain, geometry` | 208.8 | 268.8 | 3,463 | 7,123 | 18,877 |

Structures per second of `AddRadiusGraph(5)` for different sizes of batch:

| batch size | CPU, 1 thread | GPU |
|---|---:|---:|
| 1 | 1,072 | 455 |
| 32 | 5,692 | 14,052 |
| 128 | 5,771 | 53,830 |
| 512 | 5,319 | 162,442 |

#### Collate

Structures per second on one thread, for `AddRadiusGraph(5, geometry) + AddTriplets(3)` and batches of 128 structures.

| path | structures / s |
|---|---:|
| graphs built on the batch (`HDF5Dataset`) | 5,310 |
| graphs built per structure, then collate (`CifDataset`) | 717 |
| collate alone, structures with graphs | 39,095 |
| collate alone, structures without graph | 321,793 |

#### DataLoader

Structures per second of `DataLoader(dataset, batch_size=128, shuffle=True, num_workers=...)`, the batches staying on the CPU. With 0 workers the main process reads and transforms, with all the threads of torch; a worker uses one thread.

| transform | 0 workers | 1 workers | 2 workers | 4 workers | 8 workers | 16 workers |
|---|---:|---:|---:|---:|---:|---:|
| no transform | 13,670 | 11,002 | 17,725 | 27,835 | 28,972 | 31,017 |
| augmentations | 12,470 | 11,138 | 16,566 | 27,473 | 32,879 | 36,096 |
| `AddRadiusGraph(5)` | 6,909 | 3,606 | 6,784 | 12,800 | 20,594 | 19,108 |
| full chain, geometry | 4,281 | 2,288 | 4,172 | 7,874 | 13,381 | 14,763 |
| augmentations + full chain, geometry | 4,262 | 2,228 | 4,301 | 7,651 | 11,713 | 14,289 |

Seconds before the first batch with 8 workers and `AddRadiusGraph(5)`, paid at each epoch without `persistent_workers=True`:

| start method | first batch (s) |
|---|---:|
| fork | 0.21 |
| spawn | 2.54 |

#### Workers or GPU

Structures per second arriving on the GPU with their graphs: built by the transform of the dataset in the workers then moved (`OnDevice(loader, "cuda")`), or read without transform, moved, and built on the GPU by the same transforms (`OnDevice(loader, "cuda", transform)`).

| transform | workers | built in the workers | built on the GPU |
|---|---:|---:|---:|
| `AddRadiusGraph(5)` | 0 | 6,607 | 10,359 |
| `AddRadiusGraph(5)` | 1 | 3,575 | 10,049 |
| `AddRadiusGraph(5)` | 2 | 6,467 | 15,949 |
| `AddRadiusGraph(5)` | 4 | 12,578 | 17,461 |
| `AddRadiusGraph(5)` | 8 | 18,432 | 18,022 |
| `AddRadiusGraph(5)` | 16 | 18,632 | 18,638 |
| full chain, geometry | 0 | 4,073 | 7,587 |
| full chain, geometry | 1 | 2,306 | 9,570 |
| full chain, geometry | 2 | 4,122 | 9,937 |
| full chain, geometry | 4 | 7,834 | 11,003 |
| full chain, geometry | 8 | 10,023 | 9,499 |
| full chain, geometry | 16 | 11,017 | 10,886 |


## Second run: large batches, operations of a call, torch.compile

Same day, to find why the H100 is not faster than the RTX 4060 on batches of 128.
`scripts/benchmark_graph_gpu.py` on the H100 and on the local RTX 4060.

- **Commit**: `1e701a2` for the script, copied alone to `~/torchms/scripts/`; the library on
  the cluster is still the one deployed at `99032bc`, which the script only reads. Working
  tree clean.
- **Data**: `lematerial_300k.hdf5`, 300,000 structures of `datasets/lematerial/unique.hdf5`
  drawn with the seed of `--sample 300000`, written locally and copied: 9.3 atoms per
  structure on average, 360 at most.

| Job | Partition | Node | Resources | Elapsed | State |
|---|---|---|---|---|---|
| 13158401 `gpu_h100` | `h100` | nodeGcor1 | 1 GPU, 32 CPUs, 64 GB | 3 min 48 s | COMPLETED |
| 13158402 `gpu16_h100` | `h100` | nodeGcor1 | 1 GPU, 32 CPUs, 64 GB | 1 min 54 s | COMPLETED |

```bash
# 13158401, in run_gpu.sh
python scripts/benchmark_graph_gpu.py datasets/lematerial_300k.hdf5 --output results/h100_gpu.md
# 13158402, in run_gpu16.sh
python scripts/benchmark_graph_gpu.py datasets/lematerial_300k.hdf5 \
    --parts loader --workers-gpu 16 --output results/h100_gpu_16_readers.md
```

The second job exists because the first one compared 16 workers building the graphs with 4
workers reading for the GPU, which holds the GPU back with large batches.

**Logs**: both reached `benchmark done`. No line matches
`error|traceback|exception|out of memory|failed`. The node has `/usr/bin/gcc`, and
`torch.compile` ran in its three modes.

### Results

Structures per second, `AddRadiusGraph(5)`.

| | RTX 4060 (local) | H100 |
|---|---:|---:|
| Batches of 128 on the GPU | 108,498 | 65,323 |
| Batches of 8,192 on the GPU | 188,403 | 763,922 |
| One call on a batch of one structure | 0.86 ms | 1.78 ms |
| Operations of torch per call | 469 | 466 |
| Share of the CPU time in operations which wait for the GPU | 12 % | 11 % |
| `torch.compile`, default mode, against not compiled | 95,693 / 98,905 | 54,940 / 53,080 |
| Batches of 8,192 through the loader, built in 16 workers | 47,694 | 65,438 |
| Batches of 8,192 through the loader, built on the GPU, 16 readers | 175,230 | 184,336 |

- **The fixed cost of a call explains the batches of 128.** The time of a batch on the H100
  fits `1.71 ms + 1.10 µs per structure`: 87 % of a batch of 128 is fixed. The fixed part is
  the time of a batch of one structure, twice longer on the cluster as its CPU thread.
- **With large batches the H100 is 4 times faster than the RTX 4060**, and uses 0.7 GB.
- **The round trips between CPU and GPU are a tenth of the CPU time**, not the cause that was
  expected. The cause is the number of operations, about 470 per call.
- **`torch.compile` changes nothing**: +3.5 % on the H100, -3.2 % locally with the default
  mode, slower with `dynamic=True` and `reduce-overhead`, same edges.
- **Through the loader the H100 is limited by the reading**: 184,000 structures per second
  at best, a quarter of what it builds from memory.

The line fitted on the RTX 4060 has a negative fixed cost: its rate stops growing at 512
and a line does not describe it. The time of a batch of one structure is the measure of its
fixed cost.

The column "GPU working" of the profile goes over 100 % for large batches. The times of the
profiler are not used anywhere else.

### Report of the H100

#### Setting

- file: `datasets/lematerial_300k.hdf5`, 300,000 structures, 9.3 atoms per structure on average (max 360)
- CPU: x86_64, 64 threads, 32 used by torch
- GPU: NVIDIA H100 NVL, 93 GB
- torch 2.14.1+cu130, Python 3.10.20

#### Size of the batch

Batches already on the GPU. The time of a batch is fitted by `fixed + per structure × size` over the sizes (least squares).

`AddRadiusGraph(5)`:

| batch size | structures / s | ms per batch | GPU memory (GB) |
|---|---:|---:|---:|
| 128 | 65,323 | 1.96 | 0.05 |
| 512 | 228,674 | 2.24 | 0.08 |
| 2048 | 529,672 | 3.87 | 0.21 |
| 8192 | 763,922 | 10.72 | 0.71 |

Fixed cost 1.71 ms per batch, 1.10 µs per structure: the fixed cost is 87 % of a batch of 128, 77 % of a batch of 512, 44 % of a batch of 2048, 16 % of a batch of 8192.

full chain, geometry:

| batch size | structures / s | ms per batch | GPU memory (GB) |
|---|---:|---:|---:|
| 128 | 23,272 | 5.50 | 0.05 |
| 512 | 85,067 | 6.02 | 0.08 |
| 2048 | 241,761 | 8.47 | 0.21 |
| 8192 | 408,937 | 20.03 | 0.71 |

Fixed cost 5.06 ms per batch, 1.82 µs per structure: the fixed cost is 92 % of a batch of 128, 84 % of a batch of 512, 60 % of a batch of 2048, 25 % of a batch of 8192.

#### Through the loader

Structures per second arriving on the GPU with their graphs, from `DataLoader(dataset, batch_size=..., shuffle=True)`: built in 16 workers then moved, or read by 4 workers and built on the GPU by `OnDevice`.

`AddRadiusGraph(5)`:

| batch size | built in 16 workers | built on the GPU, 4 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 29,247 | 22,616 | 0.77 |
| 512 | 49,349 | 25,560 | 0.52 |
| 2048 | 50,149 | 25,638 | 0.51 |
| 8192 | 58,181 | 40,957 | 0.70 |

full chain, geometry:

| batch size | built in 16 workers | built on the GPU, 4 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 16,998 | 15,109 | 0.89 |
| 512 | 34,702 | 26,211 | 0.76 |
| 2048 | 34,286 | 25,165 | 0.73 |
| 8192 | 41,283 | 39,361 | 0.95 |

#### What a call is made of

`AddRadiusGraph(5)` on batches on the GPU. The wall time is measured without the profiler; the counts and the time of the GPU come from `torch.profiler`. An operation waits when it gives the CPU a value or a size computed on the GPU (`item`, `nonzero`, `repeat_interleave`, `bincount`, masks).

| batch size | wall (ms) | GPU working (ms) | GPU working | operations | GPU kernels | operations which wait |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.78 | 0.61 | 34 % | 488 | 170 | 38 |
| 128 | 2.44 | 1.15 | 47 % | 466 | 191 | 38 |
| 2048 | 4.16 | 5.63 | 135 % | 466 | 212 | 38 |

Operations of a call on a batch of 1, by CPU time:

| operation | calls | share of the CPU time | waits |
|---|---:|---:|---:|
| `aten::index` | 29 | 14 % |  |
| `aten::cumsum` | 6 | 6 % |  |
| `aten::bmm` | 1 | 5 % |  |
| `aten::nonzero` | 3 | 4 % | yes |
| `aten::add` | 12 | 4 % |  |
| `aten::empty` | 24 | 3 % |  |
| `aten::constant_pad_nd` | 3 | 3 % |  |
| `aten::sub` | 10 | 3 % |  |
| `aten::repeat_interleave` | 6 | 3 % | yes |
| `aten::copy_` | 16 | 3 % |  |
| `aten::mul` | 8 | 3 % |  |
| `aten::_local_scalar_dense` | 13 | 2 % | yes |

The operations which wait take 11 % of the CPU time of the operations, under the profiler.

#### torch.compile

`cutoff(structures, 5.0)` on 8 batches of 128 structures on the GPU. The first pass compiles; the rate is measured after it, on the same batches.

| function | first pass on 8 batches (s) | structures / s after it | same edges |
|---|---:|---:|---:|
| not compiled |  | 53,080 |  |
| `torch.compile(f)` | 21.0 | 54,940 | yes |
| `torch.compile(f, dynamic=True)` | 8.9 | 50,875 | yes |
| `torch.compile(f, mode="reduce-overhead")` | 7.4 | 37,601 | yes |

### Report of the H100, 16 workers reading

#### Setting

- file: `datasets/lematerial_300k.hdf5`, 300,000 structures, 9.3 atoms per structure on average (max 360)
- CPU: x86_64, 64 threads, 32 used by torch
- GPU: NVIDIA H100 NVL, 93 GB
- torch 2.14.1+cu130, Python 3.10.20

#### Through the loader

Structures per second arriving on the GPU with their graphs, from `DataLoader(dataset, batch_size=..., shuffle=True)`: built in 16 workers then moved, or read by 16 workers and built on the GPU by `OnDevice`.

`AddRadiusGraph(5)`:

| batch size | built in 16 workers | built on the GPU, 16 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 30,246 | 27,439 | 0.91 |
| 512 | 52,520 | 88,282 | 1.68 |
| 2048 | 57,805 | 118,301 | 2.05 |
| 8192 | 65,438 | 184,336 | 2.82 |

full chain, geometry:

| batch size | built in 16 workers | built on the GPU, 16 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 17,422 | 15,434 | 0.89 |
| 512 | 36,430 | 55,020 | 1.51 |
| 2048 | 37,543 | 116,030 | 3.09 |
| 8192 | 44,023 | 178,935 | 4.06 |

### Report of the RTX 4060

#### Setting

- file: `lematerial_300k.hdf5`, 300,000 structures, 9.3 atoms per structure on average (max 360)
- CPU: x86_64, 32 threads, 24 used by torch
- GPU: NVIDIA GeForce RTX 4060, 8 GB
- torch 2.11.0+cu130, Python 3.10.18

#### Size of the batch

Batches already on the GPU. The time of a batch is fitted by `fixed + per structure × size` over the sizes (least squares).

`AddRadiusGraph(5)`:

| batch size | structures / s | ms per batch | GPU memory (GB) |
|---|---:|---:|---:|
| 128 | 108,498 | 1.18 | 0.02 |
| 512 | 216,330 | 2.37 | 0.05 |
| 2048 | 202,164 | 10.13 | 0.18 |
| 8192 | 188,403 | 43.48 | 0.69 |

Fixed cost -0.15 ms per batch, 5.31 µs per structure: the fixed cost is -12 % of a batch of 128, -6 % of a batch of 512, -1 % of a batch of 2048, -0 % of a batch of 8192.

full chain, geometry:

| batch size | structures / s | ms per batch | GPU memory (GB) |
|---|---:|---:|---:|
| 128 | 41,954 | 3.05 | 0.03 |
| 512 | 111,766 | 4.58 | 0.06 |
| 2048 | 129,761 | 15.78 | 0.19 |
| 8192 | 107,778 | 76.01 | 0.69 |

Fixed cost -0.23 ms per batch, 9.22 µs per structure: the fixed cost is -8 % of a batch of 128, -5 % of a batch of 512, -1 % of a batch of 2048, -0 % of a batch of 8192.

#### Through the loader

Structures per second arriving on the GPU with their graphs, from `DataLoader(dataset, batch_size=..., shuffle=True)`: built in 16 workers then moved, or read by 4 workers and built on the GPU by `OnDevice`.

`AddRadiusGraph(5)`:

| batch size | built in 16 workers | built on the GPU, 4 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 33,654 | 38,639 | 1.15 |
| 512 | 39,274 | 46,417 | 1.18 |
| 2048 | 39,582 | 45,094 | 1.14 |
| 8192 | 54,006 | 62,218 | 1.15 |

full chain, geometry:

| batch size | built in 16 workers | built on the GPU, 4 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 21,253 | 26,758 | 1.26 |
| 512 | 25,129 | 45,651 | 1.82 |
| 2048 | 25,624 | 44,481 | 1.74 |
| 8192 | 63,580 | 60,367 | 0.95 |

#### What a call is made of

`AddRadiusGraph(5)` on batches on the GPU. The wall time is measured without the profiler; the counts and the time of the GPU come from `torch.profiler`. An operation waits when it gives the CPU a value or a size computed on the GPU (`item`, `nonzero`, `repeat_interleave`, `bincount`, masks).

| batch size | wall (ms) | GPU working (ms) | GPU working | operations | GPU kernels | operations which wait |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.86 | 0.47 | 54 % | 491 | 171 | 38 |
| 128 | 1.33 | 1.26 | 95 % | 469 | 192 | 38 |
| 2048 | 9.49 | 18.09 | 191 % | 469 | 214 | 38 |

Operations of a call on a batch of 1, by CPU time:

| operation | calls | share of the CPU time | waits |
|---|---:|---:|---:|
| `aten::index` | 29 | 14 % |  |
| `aten::cumsum` | 6 | 6 % |  |
| `aten::add` | 12 | 4 % |  |
| `aten::empty` | 26 | 4 % |  |
| `aten::nonzero` | 3 | 4 % | yes |
| `aten::sub` | 10 | 4 % |  |
| `aten::copy_` | 16 | 3 % |  |
| `aten::repeat_interleave` | 6 | 3 % | yes |
| `aten::mul` | 8 | 3 % |  |
| `aten::_local_scalar_dense` | 13 | 3 % | yes |
| `aten::arange` | 12 | 3 % |  |
| `aten::as_strided` | 84 | 2 % |  |

The operations which wait take 12 % of the CPU time of the operations, under the profiler.

#### torch.compile

`cutoff(structures, 5.0)` on 8 batches of 128 structures on the GPU. The first pass compiles; the rate is measured after it, on the same batches.

| function | first pass on 8 batches (s) | structures / s after it | same edges |
|---|---:|---:|---:|
| not compiled |  | 98,905 |  |
| `torch.compile(f)` | 6.3 | 95,693 | yes |
| `torch.compile(f, dynamic=True)` | 1.8 | 86,843 | yes |
| `torch.compile(f, mode="reduce-overhead")` | 3.4 | 72,268 | yes |

### Report of the RTX 4060, 16 workers reading

#### Setting

- file: `lematerial_300k.hdf5`, 300,000 structures, 9.3 atoms per structure on average (max 360)
- CPU: x86_64, 32 threads, 24 used by torch
- GPU: NVIDIA GeForce RTX 4060, 8 GB
- torch 2.11.0+cu130, Python 3.10.18

#### Through the loader

Structures per second arriving on the GPU with their graphs, from `DataLoader(dataset, batch_size=..., shuffle=True)`: built in 16 workers then moved, or read by 16 workers and built on the GPU by `OnDevice`.

`AddRadiusGraph(5)`:

| batch size | built in 16 workers | built on the GPU, 16 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 35,032 | 42,941 | 1.23 |
| 512 | 39,921 | 90,089 | 2.26 |
| 2048 | 44,172 | 130,380 | 2.95 |
| 8192 | 47,694 | 175,230 | 3.67 |

full chain, geometry:

| batch size | built in 16 workers | built on the GPU, 16 workers | GPU / workers |
|---|---:|---:|---:|
| 128 | 23,183 | 27,126 | 1.17 |
| 512 | 26,317 | 66,656 | 2.53 |
| 2048 | 21,430 | 104,975 | 4.90 |
| 8192 | 57,092 | 103,806 | 1.82 |
