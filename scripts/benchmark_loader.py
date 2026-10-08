#!/usr/bin/env python
"""Benchmark of the loading of a dataset and of the construction of its graphs.

Measures, on a preprocessed HDF5 file:

* reading: batches of sorted, shuffled and consecutive indices;
* graphs: each graph transform and typical chains, on CPU and on GPU;
* collate: graphs built per structure then batched, or built on the batch;
* loader: a DataLoader with several workers, without and with transforms;
* device: graphs built in the workers, or on the GPU in the training loop.

Tables are printed in Markdown, and written to a file with --output.

    uv run python scripts/benchmark_loader.py datasets/mp_20/train.hdf5
    uv run python scripts/benchmark_loader.py datasets/mp_20/train.hdf5 \
        --parts loader device --workers 0 4 8 --output loader.md
    uv run python scripts/benchmark_loader.py datasets/lematerial/unique.hdf5 \
        --sample 10000
"""

import argparse
import os
import platform
import statistics
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from torchms.data import HDF5Dataset, OnDevice, Structures
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeRotation,
)
from torchms.data.batching import TUPLES_FIELDS
from torchms.graph.transforms import (
    AddAtomicRadiusGraph,
    AddLineGraph,
    AddQuadruplets,
    AddRadiusGraph,
    AddTriplets,
    KeepNearestNeighbors,
    KeepRadius,
)

PARTS = ("reading", "graphs", "collate", "loader", "device")

Transform = Callable[[Structures], Structures]


class SortedBatches:
    """Random batches of indices, each one in ascending order.

    The sampler the dataset needed before it read indices in any order, kept
    here as the reference.
    """

    def __init__(self, dataset, batch_size):
        self.batches = BatchSampler(RandomSampler(dataset), batch_size, drop_last=False)

    def __iter__(self):
        return (sorted(batch) for batch in self.batches)

    def __len__(self):
        return len(self.batches)


def graph_chains() -> dict[str, Transform]:
    """Graph transforms to measure, by name."""
    return {
        "AddRadiusGraph(4)": AddRadiusGraph(4.0),
        "AddRadiusGraph(5)": AddRadiusGraph(5.0),
        "AddRadiusGraph(6)": AddRadiusGraph(6.0),
        "AddRadiusGraph(5, geometry)": AddRadiusGraph(5.0, geometry=True),
        "AddAtomicRadiusGraph()": AddAtomicRadiusGraph(),
        "AddRadiusGraph(6) + KeepNearestNeighbors(12)": Compose(
            [AddRadiusGraph(6.0), KeepNearestNeighbors(12)]
        ),
        "AddRadiusGraph(5) + KeepRadius(4)": Compose(
            [AddRadiusGraph(5.0), KeepRadius(4.0)]
        ),
        "AddRadiusGraph(5) + AddLineGraph(3)": Compose(
            [AddRadiusGraph(5.0), AddLineGraph(3.0)]
        ),
        "AddRadiusGraph(5) + AddTriplets(3)": Compose(
            [AddRadiusGraph(5.0), AddTriplets(3.0)]
        ),
        "AddRadiusGraph(5) + AddTriplets(4)": Compose(
            [AddRadiusGraph(5.0), AddTriplets(4.0)]
        ),
        "AddRadiusGraph(5) + AddQuadruplets(2.5)": Compose(
            [AddRadiusGraph(5.0), AddQuadruplets(2.5)]
        ),
        "full chain": full_chain(),
        "full chain, geometry": full_chain(geometry=True),
    }


def full_chain(geometry: bool = False) -> Transform:
    """Graph, triplets and quadruplets, as for a model with dihedral angles."""
    return Compose(
        [
            AddRadiusGraph(5.0, geometry=geometry),
            AddTriplets(3.0, geometry=geometry),
            AddQuadruplets(2.5, geometry=geometry),
        ]
    )


def augmentations() -> Transform:
    """Random description of the same crystals."""
    return Compose([LatticeRotation(), AtomTranslation(), AtomPermutation()])


def synchronize(device: torch.device) -> None:
    """Wait for the GPU, so that the time of a call is the time of its work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_time(function: Callable[[], object], repeats: int) -> float:
    """Median duration of a call in seconds, after one call to warm up."""
    function()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def rate(count: float, seconds: float) -> str:
    """A number of things per second."""
    return f"{count / seconds:,.0f}"


def per_structure(count: float, n_struct: int) -> str:
    return f"{count / n_struct:,.1f}"


def table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A Markdown table, numbers on the right."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]

    return "\n".join(lines)


def sizes(structures: Structures) -> dict[str, int]:
    """Number of edges and of tuples attached to structures."""
    count = {"edges": 0}
    if structures.graph is not None:
        count["edges"] = int(structures.graph.num_edges.sum())
    for name in TUPLES_FIELDS:
        tuples = getattr(structures, name)
        count[name] = 0 if tuples is None else int(tuples.num_tuples.sum())

    return count


class Benchmark:
    """Measures on a dataset, each part giving a section of the report."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = args.file
        self.directory = None
        if args.sample is not None:
            args.file = self.sample(args.file, args.sample)
        self.dataset = HDF5Dataset(args.file)
        self.gpu = torch.device("cuda") if torch.cuda.is_available() else None
        self.threads = torch.get_num_threads()

        # the same batches for all the measures of the graphs
        generator = np.random.default_rng(0)
        indices = generator.permutation(len(self.dataset))
        self.batches = [
            self.dataset[indices[i * args.batch_size :][: args.batch_size]]
            for i in range(args.batches)
        ]

    def sample(self, file: Path, size: int) -> Path:
        """Write random structures of a file to a temporary file, and give its path."""
        dataset = HDF5Dataset(file)
        indices = np.sort(np.random.default_rng(0).choice(len(dataset), size, False))

        self.directory = tempfile.TemporaryDirectory()
        sample = Path(self.directory.name) / "sample.hdf5"
        HDF5Dataset.from_structures(
            sample, (dataset[indices[i : i + 1024]] for i in range(0, size, 1024))
        )

        return sample

    def machine(self) -> str:
        num_atoms = self.dataset.dataset("num_atoms")[:]
        gpu = "none"
        if self.gpu is not None:
            gpu = torch.cuda.get_device_name(self.gpu)

        return "\n".join(
            [
                "## Setting",
                "",
                f"- file: `{self.source}`, {len(self.dataset):,} structures"
                + (" drawn at random, " if self.args.sample is not None else ", ")
                + f"{num_atoms.mean():.1f} atoms per structure on average "
                f"(max {num_atoms.max()})",
                f"- CPU: {platform.processor() or platform.machine()}, "
                f"{os.cpu_count()} threads, {self.threads} used by torch",
                f"- GPU: {gpu}",
                f"- torch {torch.__version__}, Python {platform.python_version()}",
                f"- batches of {self.args.batch_size} structures, "
                f"{self.args.batches} batches per measure, "
                f"median of {self.args.repeats} runs",
            ]
        )

    def reading(self) -> str:
        """Reading batches from the file, without transform nor DataLoader."""
        generator = np.random.default_rng(0)
        n = len(self.dataset)
        rows = []

        for batch_size in self.args.batch_sizes:
            count = min(self.args.batches * 4, n // batch_size)
            starts = generator.integers(0, n - batch_size, count)
            consecutive = [np.arange(s, s + batch_size) for s in starts]
            shuffled = [
                generator.choice(n, batch_size, replace=False) for _ in range(count)
            ]
            ascending = [np.sort(batch) for batch in shuffled]

            def read(batches):
                for batch in batches:
                    self.dataset[batch]

            row: list[object] = [batch_size]
            for batches in (consecutive, ascending, shuffled):
                seconds = median_time(lambda b=batches: read(b), self.args.repeats)
                row.append(rate(count * batch_size, seconds))
            rows.append(row)

        return "\n".join(
            [
                "## Reading",
                "",
                "Structures per second read from the file, one batch per call, in "
                "one process. Sorted batches are the ones the dataset needed "
                "before; shuffled batches are what a DataLoader with "
                "`shuffle=True` asks now.",
                "",
                table(
                    [
                        "batch size",
                        "consecutive",
                        "random, sorted",
                        "random, shuffled",
                    ],
                    rows,
                ),
            ]
        )

    def time_chain(
        self,
        chain: Transform,
        batches: list[Structures],
        device: torch.device,
        threads: int,
    ) -> float:
        """Seconds to apply a chain to batches already on the device."""
        torch.set_num_threads(threads)

        def run():
            for batch in batches:
                chain(batch)
            synchronize(device)

        try:
            return median_time(run, self.args.repeats)
        finally:
            torch.set_num_threads(self.threads)

    def graphs(self) -> str:
        """Each graph transform, on the batches of the benchmark."""
        n_struct = sum(len(batch.num_atoms) for batch in self.batches)
        on_gpu = None
        if self.gpu is not None:
            on_gpu = [batch.to(self.gpu) for batch in self.batches]

        rows = []
        for name, chain in graph_chains().items():
            count = {"edges": 0, **{field: 0 for field in TUPLES_FIELDS}}
            for batch in self.batches:
                for field, value in sizes(chain(batch)).items():
                    count[field] += value

            row: list[object] = [f"`{name}`"]
            row.append(per_structure(count["edges"], n_struct))
            tuples = max(count[field] for field in TUPLES_FIELDS)
            row.append(per_structure(tuples, n_struct) if tuples else "")
            row.append(
                rate(
                    n_struct,
                    self.time_chain(chain, self.batches, torch.device("cpu"), 1),
                )
            )
            row.append(
                rate(
                    n_struct,
                    self.time_chain(
                        chain, self.batches, torch.device("cpu"), self.threads
                    ),
                )
            )
            if on_gpu is not None and self.gpu is not None:
                row.append(
                    rate(
                        n_struct, self.time_chain(chain, on_gpu, self.gpu, self.threads)
                    )
                )
            rows.append(row)

        header = [
            "transform",
            "edges / structure",
            "largest tuples / structure",
            "CPU, 1 thread",
            f"CPU, {self.threads} threads",
        ]
        if on_gpu is not None:
            header.append("GPU")

        lines = [
            "## Graph construction",
            "",
            f"Structures per second for batches of {self.args.batch_size} "
            "structures already in memory, on the device of the measure. The "
            "sizes are the edges of the graph and the largest of the tuples of "
            "edges built (pairs, triplets or quadruplets), per structure.",
            "",
            table(header, rows),
            "",
            self.graphs_batch_size(),
        ]

        return "\n".join(lines)

    def graphs_batch_size(self) -> str:
        """Effect of the size of the batch on the construction of a graph."""
        structures = Structures.separate(Structures.collate(self.batches))
        structures = structures[: max(self.args.batch_sizes) * 2]
        chain = AddRadiusGraph(5.0)

        rows = []
        for batch_size in (1, *self.args.batch_sizes):
            batches = [
                Structures.collate(structures[i : i + batch_size])
                for i in range(0, len(structures), batch_size)
            ]
            row: list[object] = [batch_size]
            row.append(
                rate(
                    len(structures),
                    self.time_chain(chain, batches, torch.device("cpu"), 1),
                )
            )
            if self.gpu is not None:
                on_gpu = [batch.to(self.gpu) for batch in batches]
                row.append(
                    rate(
                        len(structures),
                        self.time_chain(chain, on_gpu, self.gpu, self.threads),
                    )
                )
            rows.append(row)

        header = ["batch size", "CPU, 1 thread"]
        if self.gpu is not None:
            header.append("GPU")

        return "\n".join(
            [
                "Structures per second of `AddRadiusGraph(5)` for different sizes "
                "of batch:",
                "",
                table(header, rows),
            ]
        )

    def collate(self) -> str:
        """Graphs built per structure then batched, or built on the batch."""
        chain = Compose([AddRadiusGraph(5.0, geometry=True), AddTriplets(3.0)])
        batches = self.batches[: max(1, self.args.batches // 4)]
        separated = [Structures.separate(batch) for batch in batches]
        n_struct = sum(len(batch.num_atoms) for batch in batches)
        torch.set_num_threads(1)

        def on_batch():
            for batch in batches:
                chain(batch)

        def per_structure_then_collate():
            for structures in separated:
                Structures.collate([chain(s) for s in structures])

        built = [[chain(s) for s in structures] for structures in separated]

        def collate_only():
            for structures in built:
                Structures.collate(structures)

        def collate_without_graph():
            for structures in separated:
                Structures.collate(structures)

        rows = [
            ["graphs built on the batch (`HDF5Dataset`)", on_batch],
            [
                "graphs built per structure, then collate (`CifDataset`)",
                per_structure_then_collate,
            ],
            ["collate alone, structures with graphs", collate_only],
            ["collate alone, structures without graph", collate_without_graph],
        ]
        try:
            rows = [
                [name, rate(n_struct, median_time(function, self.args.repeats))]
                for name, function in rows
            ]
        finally:
            torch.set_num_threads(self.threads)

        return "\n".join(
            [
                "## Collate",
                "",
                "Structures per second on one thread, for "
                "`AddRadiusGraph(5, geometry) + AddTriplets(3)` and batches of "
                f"{self.args.batch_size} structures.",
                "",
                table(["path", "structures / s"], rows),
            ]
        )

    def loader_rate(
        self,
        transform: Transform | None,
        workers: int,
        device: torch.device | None = None,
        device_transform: Transform | None = None,
        context: str | None = None,
    ) -> tuple[float, float]:
        """Structures per second of a DataLoader, and seconds before its first batch.

        With a device, the batches are moved to it by OnDevice, which applies
        device_transform there. The first batches, read while the workers
        start, are not counted in the rate.
        """
        dataset = HDF5Dataset(self.args.file, transform=transform)
        loader: Iterable[Structures] = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=workers,
            multiprocessing_context=context if workers > 0 else None,
        )
        if device is not None:
            loader = OnDevice(loader, device, device_transform)
        skipped = max(2, 2 * workers)

        start = time.perf_counter()
        first = None
        count = 0
        measured = start
        for i, batch in enumerate(loader):
            if device is not None:
                synchronize(device)
            if i == 0:
                first = time.perf_counter() - start
            if i == skipped - 1:
                measured = time.perf_counter()
            elif i >= skipped:
                count += len(batch.num_atoms)
            if i >= skipped + self.args.loader_batches - 1:
                break

        seconds = time.perf_counter() - measured
        del loader

        return count / seconds, first if first is not None else float("nan")

    def loader(self) -> str:
        """A DataLoader with several workers, without and with transforms."""
        chains: dict[str, Transform | None] = {
            "no transform": None,
            "augmentations": augmentations(),
            "`AddRadiusGraph(5)`": AddRadiusGraph(5.0),
            "full chain, geometry": full_chain(geometry=True),
            "augmentations + full chain, geometry": Compose(
                [augmentations(), full_chain(geometry=True)]
            ),
        }

        rows = []
        for name, chain in chains.items():
            row: list[object] = [name]
            for workers in self.args.workers:
                row.append(f"{self.loader_rate(chain, workers)[0]:,.0f}")
            rows.append(row)

        header = ["transform"] + [f"{workers} workers" for workers in self.args.workers]

        start = []
        workers = max(w for w in self.args.workers if w <= 8)
        if workers > 0:
            for context in ("fork", "spawn"):
                _, first = self.loader_rate(
                    AddRadiusGraph(5.0), workers, context=context
                )
                start.append([context, f"{first:.2f}"])

        lines = [
            "## DataLoader",
            "",
            "Structures per second of `DataLoader(dataset, "
            f"batch_size={self.args.batch_size}, shuffle=True, num_workers=...)`, "
            "the batches staying on the CPU. With 0 workers the main process "
            "reads and transforms, with all the threads of torch; a worker uses "
            "one thread.",
            "",
            table(header, rows),
        ]
        if start:
            lines += [
                "",
                f"Seconds before the first batch with {workers} workers and "
                "`AddRadiusGraph(5)`, paid at each epoch without "
                "`persistent_workers=True`:",
                "",
                table(["start method", "first batch (s)"], start),
            ]

        return "\n".join(lines)

    def device(self) -> str:
        """Graphs built in the workers, or on the GPU in the training loop."""
        if self.gpu is None:
            return "## Workers or GPU\n\nNo GPU on this machine."

        gpu = self.gpu
        chains = {
            "`AddRadiusGraph(5)`": AddRadiusGraph(5.0),
            "full chain, geometry": full_chain(geometry=True),
        }

        rows = []
        for name, chain in chains.items():
            for workers in self.args.workers:
                in_workers, _ = self.loader_rate(chain, workers, gpu)
                on_gpu, _ = self.loader_rate(None, workers, gpu, chain)
                rows.append([name, workers, f"{in_workers:,.0f}", f"{on_gpu:,.0f}"])

        return "\n".join(
            [
                "## Workers or GPU",
                "",
                "Structures per second arriving on the GPU with their graphs: "
                "built by the transform of the dataset in the workers then moved "
                '(`OnDevice(loader, "cuda")`), or read without transform, moved, '
                "and built on the GPU by the same transforms "
                '(`OnDevice(loader, "cuda", transform)`).',
                "",
                table(
                    [
                        "transform",
                        "workers",
                        "built in the workers",
                        "built on the GPU",
                    ],
                    rows,
                ),
            ]
        )

    def run(self) -> str:
        sections = [f"# Benchmark of `{self.source}`", self.machine()]
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
        "--batch-size", type=int, default=128, help="structures per batch"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        help="sizes of batch compared for the reading and the graphs",
    )
    parser.add_argument(
        "--batches", type=int, default=16, help="batches per measure of the graphs"
    )
    parser.add_argument(
        "--loader-batches", type=int, default=100, help="batches per DataLoader"
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per measure")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16],
        help="numbers of workers of the DataLoader",
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
