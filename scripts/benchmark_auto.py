#!/usr/bin/env python
"""Benchmark of auto_loader: its choice against each way of loading.

For a dataset and two chains of graph transforms, prints the configuration
auto_loader detects, then the structures per second it gives, against the same
loader with each storage forced, and against a plain DataLoader.

    uv run python scripts/benchmark_auto.py datasets/mp_20/train.hdf5
    uv run python scripts/benchmark_auto.py mp-20 --split train
"""

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from torchms.data import OnDevice, Structures, auto_loader
from torchms.data.auto import MAX_WORKERS
from torchms.data.augmentation import Compose
from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets

Transform = Callable[[Structures], Structures]


def chains() -> dict[str, Transform]:
    return {
        "AddRadiusGraph(5)": AddRadiusGraph(5.0),
        "full chain, geometry": Compose(
            [
                AddRadiusGraph(5.0, geometry=True),
                AddTriplets(3.0, geometry=True),
                AddQuadruplets(2.5, geometry=True),
            ]
        ),
    }


def rate(loader: Iterable[Structures], seconds: float) -> float:
    """Structures per second over whole epochs, after the first batches."""
    count, skipped, start = 0, 0, time.perf_counter()

    while True:
        for batch in loader:
            if skipped < 40:
                skipped += 1
                if skipped == 40:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                continue

            count += len(batch.num_atoms)
            if time.perf_counter() - start > seconds:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return count / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="HDF5 file, or name of a published dataset")
    parser.add_argument("--split", help="split of a published dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=8.0, help="per measure")
    args = parser.parse_args()

    print(f"torch {torch.__version__}")

    for name, chain in chains().items():
        start = time.perf_counter()
        loader = auto_loader(
            args.source,
            args.split,
            batch_size=args.batch_size,
            transform=chain,
            shuffle=True,
        )
        setup = time.perf_counter() - start
        config, hardware = loader.config, loader.hardware

        print(f"\n## {name}\n")
        print(f"{hardware}\n")
        print(f"auto_loader ready in {setup:.2f} s:\n\n```\n{config}\n```\n")

        rows = [("auto_loader", rate(loader, args.seconds))]
        del loader

        workers = max(min(hardware.cpus - 1, MAX_WORKERS), 0)
        forced = {
            "gpu": replace(config, storage="gpu", num_workers=0),
            "ram": replace(config, storage="ram", num_workers=0),
            "file": replace(config, storage="file", num_workers=workers),
        }
        if config.device.type != "cuda":
            del forced["gpu"]

        for storage, forced_config in forced.items():
            loader = auto_loader(
                args.source,
                args.split,
                batch_size=args.batch_size,
                transform=chain,
                shuffle=True,
                config=forced_config,
            )
            label = f"same, dataset forced to {storage}"
            if storage == config.storage:
                label += " (the choice)"
            rows.append((label, rate(loader, args.seconds)))
            del loader

        # a DataLoader of the batch size, graphs built on the device per batch
        plain = auto_loader(
            args.source,
            args.split,
            batch_size=args.batch_size,
            shuffle=True,
            config=replace(
                config,
                storage="file",
                num_workers=workers,
                loader_batch_size=args.batch_size,
            ),
        )
        assert isinstance(plain.loader, DataLoader)
        rows.append(
            (
                f"plain DataLoader of {args.batch_size}, {workers} workers, graphs "
                "built per batch",
                rate(OnDevice(plain.loader, config.device, chain), args.seconds),
            )
        )

        best = max(value for _, value in rows)
        print("| loader | structures / s | of the best |")
        print("|---|---:|---:|")
        for label, value in rows:
            print(f"| {label} | {value:,.0f} | {value / best:.0%} |")

    print("\ndone")


if __name__ == "__main__":
    main()
