#!/usr/bin/env python
"""Benchmark of the construction of graphs on a GPU: what limits it.

Measures, on a preprocessed HDF5 file and a GPU:

* batch: the rate for batches of 128 to 8192 structures, and the time of a
  batch split into a fixed cost and a cost per structure;
* loader: the same sizes through a DataLoader, graphs built in the workers or
  on the GPU;
* profile: the operations of one call, the time the GPU works, the operations
  which wait for the GPU;
* compile: the same transform compiled by torch.compile.

Tables are printed in Markdown, and written to a file with --output.

    uv run python scripts/benchmark_graph_gpu.py datasets/lematerial/unique.hdf5 \
        --sample 1000000
"""

import argparse
import os
import platform
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchms.data import HDF5Dataset, OnDevice, Structures
from torchms.data.augmentation import Compose
from torchms.graph.cutoff import cutoff
from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets

PARTS = ("batch", "loader", "profile", "compile")

Transform = Callable[[Structures], Structures]

# operations which give the CPU a value or a size computed on the GPU: the CPU
# waits for the GPU to finish what it was given
WAITING = (
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::is_nonzero",
    "aten::nonzero",
    "aten::repeat_interleave",
    "aten::bincount",
    "aten::masked_select",
    "aten::unique_dim",
    "aten::_unique2",
)


def chains() -> dict[str, Transform]:
    return {
        "`AddRadiusGraph(5)`": AddRadiusGraph(5.0),
        "full chain, geometry": Compose(
            [
                AddRadiusGraph(5.0, geometry=True),
                AddTriplets(3.0, geometry=True),
                AddQuadruplets(2.5, geometry=True),
            ]
        ),
    }


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A Markdown table, numbers on the right."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]

    return "\n".join(lines)


class Benchmark:
    """Measures on a dataset and a GPU, each part giving a section of the report."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = args.file
        self.directory = None
        if args.sample is not None:
            args.file = self.sample(args.file, args.sample)
        self.dataset = HDF5Dataset(args.file)
        self.gpu = torch.device("cuda")
        self.generator = np.random.default_rng(0)

    def sample(self, file: Path, size: int) -> Path:
        """Write random structures of a file to a temporary file, and give its path."""
        dataset = HDF5Dataset(file)
        indices = np.sort(np.random.default_rng(0).choice(len(dataset), size, False))

        self.directory = tempfile.TemporaryDirectory()
        sample = Path(self.directory.name) / "sample.hdf5"
        HDF5Dataset.from_structures(
            sample, (dataset[indices[i : i + 4096]] for i in range(0, size, 4096))
        )

        return sample

    def batches(self, batch_size: int, count: int) -> list[Structures]:
        """Random batches of the dataset, on the GPU."""
        n = len(self.dataset)

        return [
            self.dataset[np.sort(self.generator.choice(n, batch_size, False))].to(
                self.gpu
            )
            for _ in range(count)
        ]

    def seconds_per_batch(
        self, chain: Callable[[Structures], object], batches: list[Structures]
    ) -> float:
        """Median time of a call on a batch already on the GPU."""
        for batch in batches[:2]:
            chain(batch)

        times = []
        for _ in range(self.args.repeats):
            for batch in batches:
                torch.cuda.synchronize()
                start = time.perf_counter()
                chain(batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        return statistics.median(times)

    def machine(self) -> str:
        num_atoms = self.dataset.dataset("num_atoms")[:]
        memory = torch.cuda.get_device_properties(self.gpu).total_memory / 2**30

        return "\n".join(
            [
                "## Setting",
                "",
                f"- file: `{self.source}`, {len(self.dataset):,} structures"
                + (" drawn at random, " if self.args.sample is not None else ", ")
                + f"{num_atoms.mean():.1f} atoms per structure on average "
                f"(max {num_atoms.max()})",
                f"- CPU: {platform.processor() or platform.machine()}, "
                f"{os.cpu_count()} threads, {torch.get_num_threads()} used by torch",
                f"- GPU: {torch.cuda.get_device_name(self.gpu)}, {memory:.0f} GB",
                f"- torch {torch.__version__}, Python {platform.python_version()}",
            ]
        )

    def batch(self) -> str:
        """Rate and time of a batch for each size, fixed cost and cost per structure."""
        sections = [
            "## Size of the batch",
            "",
            "Batches already on the GPU. The time of a batch is fitted by "
            "`fixed + per structure × size` over the sizes (least squares).",
        ]

        for name, chain in chains().items():
            rows, sizes, times = [], [], []
            for batch_size in self.args.batch_sizes:
                try:
                    batches = self.batches(batch_size, self.args.batches)
                    seconds = self.seconds_per_batch(chain, batches)
                    memory = torch.cuda.max_memory_allocated(self.gpu) / 2**30
                except torch.OutOfMemoryError:
                    rows.append([batch_size, "out of memory", "", ""])
                    torch.cuda.empty_cache()
                    break
                del batches
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.gpu)

                sizes.append(batch_size)
                times.append(seconds)
                rows.append(
                    [
                        batch_size,
                        f"{batch_size / seconds:,.0f}",
                        f"{1000 * seconds:.2f}",
                        f"{memory:.2f}",
                    ]
                )

            per_structure, fixed = np.polyfit(sizes, times, 1)
            sections += [
                "",
                f"{name}:",
                "",
                table(
                    ["batch size", "structures / s", "ms per batch", "GPU memory (GB)"],
                    rows,
                ),
                "",
                f"Fixed cost {1000 * fixed:.2f} ms per batch, "
                f"{1e6 * per_structure:.2f} µs per structure: the fixed cost is "
                + ", ".join(
                    f"{100 * fixed / t:.0f} % of a batch of {s}"
                    for s, t in zip(sizes, times)
                )
                + ".",
            ]

        return "\n".join(sections)

    def loader_rate(
        self, batch_size: int, workers: int, transform, device_transform
    ) -> float:
        """Structures per second arriving on the GPU from a DataLoader."""
        dataset = HDF5Dataset(self.args.file, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        loader = OnDevice(loader, self.gpu, device_transform)

        # the batches read while the workers start are not counted
        skipped = workers + 2
        count, start = 0, time.perf_counter()
        for i, batch in enumerate(loader):
            torch.cuda.synchronize()
            if i == skipped - 1:
                start = time.perf_counter()
            elif i >= skipped:
                count += len(batch.num_atoms)
                if time.perf_counter() - start > self.args.loader_seconds:
                    break

        seconds = time.perf_counter() - start
        del loader

        return count / seconds if count > 0 else float("nan")

    def loader(self) -> str:
        """Graphs built in the workers or on the GPU, for each size of batch."""
        w_cpu, w_gpu = self.args.workers_cpu, self.args.workers_gpu
        sections = [
            "## Through the loader",
            "",
            "Structures per second arriving on the GPU with their graphs, from "
            "`DataLoader(dataset, batch_size=..., shuffle=True)`: built in "
            f"{w_cpu} workers then moved, or read by {w_gpu} workers and built on "
            "the GPU by `OnDevice`.",
        ]

        for name, chain in chains().items():
            rows = []
            for batch_size in self.args.batch_sizes:
                needed = batch_size * (max(w_cpu, w_gpu) + 6)
                if needed > len(self.dataset):
                    rows.append([batch_size, "not enough structures", "", ""])
                    continue

                try:
                    in_workers = self.loader_rate(batch_size, w_cpu, chain, None)
                    on_gpu = self.loader_rate(batch_size, w_gpu, None, chain)
                except torch.OutOfMemoryError:
                    rows.append([batch_size, "out of memory", "", ""])
                    torch.cuda.empty_cache()
                    continue

                rows.append(
                    [
                        batch_size,
                        f"{in_workers:,.0f}",
                        f"{on_gpu:,.0f}",
                        f"{on_gpu / in_workers:.2f}",
                    ]
                )

            sections += [
                "",
                f"{name}:",
                "",
                table(
                    [
                        "batch size",
                        f"built in {w_cpu} workers",
                        f"built on the GPU, {w_gpu} workers",
                        "GPU / workers",
                    ],
                    rows,
                ),
            ]

        return "\n".join(sections)

    def profile(self) -> str:
        """Operations of a call: how many, how long the GPU works, which ones wait."""
        from torch.profiler import ProfilerActivity, profile

        chain = AddRadiusGraph(5.0)
        sections = [
            "## What a call is made of",
            "",
            "`AddRadiusGraph(5)` on batches on the GPU. The wall time is measured "
            "without the profiler; the counts and the time of the GPU come from "
            "`torch.profiler`. An operation waits when it gives the CPU a value or "
            "a size computed on the GPU (`item`, `nonzero`, `repeat_interleave`, "
            "`bincount`, masks).",
        ]

        rows, details = [], ""
        for batch_size in self.args.profile_sizes:
            batches = self.batches(batch_size, self.args.batches)
            wall = self.seconds_per_batch(chain, batches)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for batch in batches:
                    chain(batch)
                torch.cuda.synchronize()

            events = prof.key_averages()
            calls = len(batches)

            def device_time(event) -> float:
                return getattr(
                    event,
                    "self_device_time_total",
                    getattr(event, "self_cuda_time_total", 0),
                )

            operations = [e for e in events if e.key.startswith("aten::")]
            kernels = [e for e in events if str(e.device_type).endswith("CUDA")]
            waiting = [e for e in operations if e.key in WAITING]
            gpu_ms = sum(device_time(e) for e in events) / 1000 / calls

            rows.append(
                [
                    batch_size,
                    f"{1000 * wall:.2f}",
                    f"{gpu_ms:.2f}",
                    f"{100 * gpu_ms / (1000 * wall):.0f} %",
                    f"{sum(e.count for e in operations) / calls:.0f}",
                    f"{sum(e.count for e in kernels) / calls:.0f}",
                    f"{sum(e.count for e in waiting) / calls:.0f}",
                ]
            )

            if batch_size == self.args.profile_sizes[0]:
                total = sum(e.self_cpu_time_total for e in operations)
                top = sorted(operations, key=lambda e: -e.self_cpu_time_total)[:12]
                details = table(
                    ["operation", "calls", "share of the CPU time", "waits"],
                    [
                        [
                            f"`{e.key}`",
                            f"{e.count / calls:.0f}",
                            f"{100 * e.self_cpu_time_total / total:.0f} %",
                            "yes" if e.key in WAITING else "",
                        ]
                        for e in top
                    ],
                )
                share = sum(e.self_cpu_time_total for e in waiting) / total
                details += (
                    f"\n\nThe operations which wait take {100 * share:.0f} % of the "
                    "CPU time of the operations, under the profiler."
                )

        sections += [
            "",
            table(
                [
                    "batch size",
                    "wall (ms)",
                    "GPU working (ms)",
                    "GPU working",
                    "operations",
                    "GPU kernels",
                    "operations which wait",
                ],
                rows,
            ),
            "",
            f"Operations of a call on a batch of {self.args.profile_sizes[0]}, by "
            "CPU time:",
            "",
            details,
        ]

        return "\n".join(sections)

    def compile(self) -> str:
        """The same graph function compiled by torch.compile."""
        batch_size = self.args.compile_size
        batches = self.batches(batch_size, self.args.batches)

        def eager(structures: Structures):
            return cutoff(structures, 5.0)

        reference = [eager(batch) for batch in batches]
        rows = [
            [
                "not compiled",
                "",
                f"{batch_size / self.seconds_per_batch(eager, batches):,.0f}",
                "",
            ]
        ]

        options: dict[str, dict] = {
            "`torch.compile(f)`": {},
            "`torch.compile(f, dynamic=True)`": {"dynamic": True},
            '`torch.compile(f, mode="reduce-overhead")`': {"mode": "reduce-overhead"},
        }
        for name, option in options.items():
            torch._dynamo.reset()  # pylint: disable=protected-access
            try:
                compiled = torch.compile(eager, **option)

                # every batch has its own numbers of atoms: each one is a new shape
                start = time.perf_counter()
                for batch in batches:
                    compiled(batch)
                torch.cuda.synchronize()
                first = time.perf_counter() - start

                seconds = self.seconds_per_batch(compiled, batches)
                same = all(
                    torch.equal(compiled(batch).edge_index, graph.edge_index)
                    for batch, graph in zip(batches, reference)
                )
                rows.append(
                    [
                        name,
                        f"{first:.1f}",
                        f"{batch_size / seconds:,.0f}",
                        "yes" if same else "NO",
                    ]
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                message = str(error).strip().splitlines()[0][:120]
                rows.append([name, "failed", f"{type(error).__name__}: {message}", ""])

        return "\n".join(
            [
                "## torch.compile",
                "",
                f"`cutoff(structures, 5.0)` on {len(batches)} batches of "
                f"{batch_size} structures on the GPU. The first pass compiles; the "
                "rate is measured after it, on the same batches.",
                "",
                table(
                    [
                        "function",
                        f"first pass on {len(batches)} batches (s)",
                        "structures / s after it",
                        "same edges",
                    ],
                    rows,
                ),
            ]
        )

    def run(self) -> str:
        sections = [f"# Graphs on the GPU, `{self.source}`", self.machine()]
        for part in self.args.parts:
            print(f"measuring: {part} ...", flush=True)
            sections.append(getattr(self, part)())

        return "\n\n".join(sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=Path, help="preprocessed HDF5 file")
    parser.add_argument("--output", type=Path, help="Markdown file of the report")
    parser.add_argument(
        "--sample",
        type=int,
        help="measure on this number of structures drawn at random from the file",
    )
    parser.add_argument(
        "--parts", nargs="+", choices=PARTS, default=list(PARTS), help="what to measure"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 8192],
        help="sizes of batch compared",
    )
    parser.add_argument(
        "--profile-sizes",
        type=int,
        nargs="+",
        default=[1, 128, 2048],
        help="sizes of batch profiled",
    )
    parser.add_argument(
        "--compile-size", type=int, default=128, help="size of batch compiled"
    )
    parser.add_argument("--batches", type=int, default=8, help="batches per measure")
    parser.add_argument("--repeats", type=int, default=3, help="passes per measure")
    parser.add_argument(
        "--loader-seconds", type=float, default=10.0, help="seconds per DataLoader"
    )
    parser.add_argument(
        "--workers-cpu",
        type=int,
        default=16,
        help="workers of the loader which builds the graphs",
    )
    parser.add_argument(
        "--workers-gpu",
        type=int,
        default=4,
        help="workers of the loader when the graphs are built on the GPU",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = Benchmark(args).run()

    print()
    print(report)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
