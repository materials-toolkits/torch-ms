"""Preprocessing module for converting CIF files to HDF5 format."""

import argparse
import multiprocessing as mp
import os
import re
import resource
import time
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event
from typing import Callable, Optional, TextIO

import torch
import tqdm

from pycrystalgen.data.batching import Structures
from pycrystalgen.data.cif import CifParser
from pycrystalgen.data.constants import symbol_to_number
from pycrystalgen.data.filters import Filter
from pycrystalgen.data.hdf5 import HDF5Dataset


@dataclass
class Preprocessing:
    """Configuration for CIF to HDF5 preprocessing.

    Args:
        source: Path to input CIF file.
        destination: Path to output HDF5 file.
        batch_size: Number of structures per batch.
        num_workers: Number of worker processes (0 for single-process).
        prefetch: Number of batches to prefetch in multiprocessing mode.
        chunk_size: Size of chunks when reading input file.
        filter: Optional callable returning a mask of the structures to keep
            for each parsed batch (see pycrystalgen.data.filters.Filter).
    """

    source: str | os.PathLike
    destination: str | os.PathLike
    batch_size: int = 32
    num_workers: int = 0
    prefetch: int = 16
    chunk_size: int = 1 << 16
    filter: Optional[Callable[[Structures], torch.Tensor]] = None
    queue_raw_batch: Optional[mp.Queue] = field(default=None, init=False, repr=False)
    queue_structures: Optional[mp.Queue] = field(default=None, init=False, repr=False)
    finished: Optional[Event] = field(default=None, init=False, repr=False)

    def setup_multiprocessing(self) -> None:
        """Initialize multiprocessing queues and event."""
        self.queue_raw_batch = mp.Queue()
        self.queue_structures = mp.Queue()
        self.finished = mp.Event()

    def __post_init__(self):
        if self.num_workers > 0:
            self.setup_multiprocessing()

    def multiprocessing(self) -> tuple[mp.Queue, mp.Queue, Event]:
        """Get the queues of raw batches and structures, and the finished event.

        Raises:
            RuntimeError: If multiprocessing is not set up (num_workers is 0).
        """
        if (
            self.queue_raw_batch is None
            or self.queue_structures is None
            or self.finished is None
        ):
            raise RuntimeError("multiprocessing is not set up, num_workers must be > 0")

        return self.queue_raw_batch, self.queue_structures, self.finished


def _load_cif(args: Preprocessing):
    """Load CIF file and put batches into queue."""
    queue_raw_batch, _, finished = args.multiprocessing()

    try:
        with ChunkedCifLoader(
            args.source, args.batch_size, args.chunk_size
        ) as batch_loader:
            progress = tqdm.tqdm(
                total=batch_loader.total_size,
                desc=f"preprocessing {args.source}",
                unit="B",
                unit_scale=True,
            )

            for batch in batch_loader:
                progress.n = batch_loader.fetched_size
                progress.refresh()

                queue_raw_batch.put(batch)
                while queue_raw_batch.qsize() >= args.prefetch:
                    time.sleep(0.001)

                    if finished.is_set():
                        queue_raw_batch.put(None)
                        return

            queue_raw_batch.put(None)
            finished.wait()

    except Exception as e:
        queue_raw_batch.put(None)
        raise e


def _parse(parser: CifParser, batch: str, args: Preprocessing) -> Structures | None:
    """Parse a batch of CIF blocks, skip invalid blocks and apply the filter.

    Returns:
        Batched structures, or None if no structure is left.
    """
    parsed = [
        result.structures
        for result in parser.parse_with_errors(batch)
        if result.structures is not None
    ]

    if len(parsed) > 0 and args.filter is not None:
        mask = args.filter(Structures.collate(parsed))
        parsed = [struct for struct, keep in zip(parsed, mask.tolist()) if keep]

    if len(parsed) == 0:
        return None

    return Structures.collate(parsed)


def _parse_batch(args: Preprocessing):
    """Parse CIF batches from queue and put structures in another queue."""
    queue_raw_batch, queue_structures, finished = args.multiprocessing()
    parser = CifParser()

    next_batch = queue_raw_batch.get()
    while next_batch is not None:
        structures = _parse(parser, next_batch, args)

        while True:
            time.sleep(0.001)

            if finished.is_set():
                queue_structures.put(None)
                return

            if queue_structures.qsize() < args.prefetch:
                break

        if structures is not None:
            queue_structures.put(structures)

        next_batch = queue_raw_batch.get()

    queue_raw_batch.put(None)
    queue_structures.put(None)
    finished.wait()


def _queue_iterator(queue: mp.Queue, num_workers: int):
    """Yield items from queue until all workers are done."""
    count_finished = 0

    while count_finished < num_workers:
        data = queue.get()
        if data is None:
            count_finished += 1
        else:
            yield data


def _save_dataset(args: Preprocessing):
    """Save structures from queue to HDF5 file."""
    _, queue_structures, _ = args.multiprocessing()

    HDF5Dataset.from_structures(
        args.destination, _queue_iterator(queue_structures, args.num_workers)
    )


def _preprocessing_single_process(args: Preprocessing):
    """Preprocess CIF file in single-process mode."""
    with ChunkedCifLoader(
        args.source, args.batch_size, args.chunk_size
    ) as batch_loader:
        parser = CifParser()

        progress = tqdm.tqdm(
            total=batch_loader.total_size,
            desc=f"preprocessing {args.source}",
            unit="B",
            unit_scale=True,
        )

        def update_progress(batch: str) -> Structures | None:
            progress.n = batch_loader.fetched_size
            progress.refresh()
            return _parse(parser, batch, args)

        HDF5Dataset.from_structures(
            args.destination,
            (s for s in map(update_progress, batch_loader) if s is not None),
        )


def preprocessing(args: Preprocessing) -> None:
    """Convert CIF file to HDF5 format.

    Args:
        args: Preprocessing configuration object.
    """
    _, hard_limit = resource.getrlimit(resource.RLIMIT_OFILE)
    resource.setrlimit(resource.RLIMIT_OFILE, (hard_limit, hard_limit))

    if args.num_workers == 0:
        _preprocessing_single_process(args)
    else:
        _preprocessing_multiprocess(args)


def _preprocessing_multiprocess(args: Preprocessing):
    """Preprocess CIF file using multiple processes."""
    _, _, finished = args.multiprocessing()

    loader_process = mp.Process(
        target=_load_cif,
        args=(args,),
    )
    loader_process.start()

    parser_process = []
    for _ in range(args.num_workers):
        parser = mp.Process(
            target=_parse_batch,
            args=(args,),
        )
        parser.start()
        parser_process.append(parser)

    saver_process = mp.Process(target=_save_dataset, args=(args,))
    saver_process.start()

    saver_process.join()
    finished.set()
    loader_process.join()
    for parser in parser_process:
        parser.join()


class ChunkedCifLoader:
    """Iterator that loads CIF file in chunks and yields batches of structures.

    Args:
        file_name: Path to CIF file.
        batch_size: Number of structures per batch.
        chunk_size: Size of chunks when reading file.
    """

    def __init__(self, file_name: str | os.PathLike, batch_size: int, chunk_size: int):
        self._file_name = file_name
        self.header_match = re.compile(r"^data_(\S+)", re.MULTILINE)

        self.total_size = os.path.getsize(file_name)
        self.fetched_size = 0

        self._file: TextIO | None = None
        self.fetched = ""

        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __enter__(self):
        self._file = open(self._file_name, "r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
        return False

    def __iter__(self):
        return self

    def consume_until_header(
        self, consume_header: bool = True, _end_of_file: bool = False
    ) -> str | None:
        """Consume content until a CIF header is found.

        Args:
            consume_header: If True, include header in consumed content.
            _end_of_file: Internal flag for recursion.

        Returns:
            Consumed content or None if at end of file.
        """
        header = self.header_match.search(self.fetched)

        if header:
            begin, end = header.span()

            if consume_header:
                consumed = self.fetched[:end]
                self.fetched = self.fetched[end:]
            else:
                consumed = self.fetched[:begin]
                self.fetched = self.fetched[begin:]

            return consumed

        if _end_of_file:
            if len(self.fetched) > 0:
                consumed = self.fetched
                self.fetched = ""
                return consumed

            return None

        if self._file is None:
            raise RuntimeError("file is not opened, use the loader in a with block")

        next_chunk = self._file.read(self.chunk_size)
        self.fetched_size += len(next_chunk)
        end_of_file = len(next_chunk) == 0
        self.fetched += next_chunk
        return self.consume_until_header(
            consume_header=consume_header, _end_of_file=end_of_file
        )

    def consume_single_structure(self) -> str | None:
        """Consume a single CIF structure block.

        Returns:
            CIF block content or None if at end of file.
        """
        block_header = self.consume_until_header(consume_header=True)

        if block_header is None:  # handle end of file
            return None

        block_body = self.consume_until_header(consume_header=False)

        if block_body is None:  # handle last block
            block = block_header + self.fetched
            self.fetched = ""
        else:
            block = block_header + block_body

        return block

    def __next__(self):
        batch = ""

        for _ in range(self.batch_size):
            next_struct = self.consume_single_structure()
            if next_struct is None:
                break
            batch += next_struct

        if len(batch) == 0:
            raise StopIteration

        return batch


def main():
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        prog="pycrystalgen-preprocessing",
        description="Convert CIF files to HDF5 format",
    )
    parser.add_argument("source", help="input file (cif format)")
    parser.add_argument(
        "destination", nargs="?", help="output file (hdf5 format)", default=None
    )
    parser.add_argument(
        "--batch-size",
        help="size of a batch during preprocessing",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--num-workers", help="number of workers during parsing", default=0, type=int
    )
    parser.add_argument(
        "--prefetch",
        help="number of batches to prefetch in multiprocessing mode",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--chunk-size",
        help="size of a chunk when input file is loaded",
        default=1 << 16,
        type=int,
    )

    parser.add_argument(
        "--max-atoms",
        help="drop structures with more atoms",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--exclude-elements",
        help="drop structures containing one of these elements (e.g. He Ne Ar)",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--max-energy-above-hull",
        help="drop structures above this energy above the hull in eV/atom "
        "(0 for stable, 0.1 for meta-stable)",
        default=None,
        type=float,
    )

    args = parser.parse_args()

    if args.destination is None:
        args.destination = os.path.splitext(args.source)[0] + ".hdf5"

    preprocessing(
        Preprocessing(
            source=args.source,
            destination=args.destination,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch=args.prefetch,
            chunk_size=args.chunk_size,
            filter=Filter(
                max_atoms=args.max_atoms,
                exclude_elements=tuple(map(symbol_to_number, args.exclude_elements)),
                max_energy_above_hull=args.max_energy_above_hull,
            ),
        )
    )


if __name__ == "__main__":
    main()
