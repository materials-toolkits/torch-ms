"""Preprocessing module for converting CIF files to HDF5 format."""

import argparse
import multiprocessing as mp
import os
import re
import resource
import time
from dataclasses import dataclass, field
from typing import Optional, TextIO

import tqdm

from pycrystalgen.data.batching import Structures
from pycrystalgen.data.cif import CifParser
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
    """

    source: str | os.PathLike
    destination: str | os.PathLike
    batch_size: int = 32
    num_workers: int = 0
    prefetch: int = 16
    chunk_size: int = 1 << 16
    queue_raw_batch: Optional[mp.Queue] = field(default=None, init=False, repr=False)
    queue_structures: Optional[mp.Queue] = field(default=None, init=False, repr=False)
    finished: Optional[mp.Event] = field(default=None, init=False, repr=False)

    def setup_multiprocessing(self) -> None:
        """Initialize multiprocessing queues and event."""
        self.queue_raw_batch = mp.Queue()
        self.queue_structures = mp.Queue()
        self.finished = mp.Event()

    def __post_init__(self):
        if self.num_workers > 0:
            self.setup_multiprocessing()


def _load_cif(args: Preprocessing):
    """Load CIF file and put batches into queue."""
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

                args.queue_raw_batch.put(batch)
                while args.queue_raw_batch.qsize() >= args.prefetch:
                    time.sleep(0.001)

                    if args.finished.is_set():
                        args.queue_raw_batch.put(None)
                        return

            args.queue_raw_batch.put(None)
            args.finished.wait()

    except Exception as e:
        args.queue_raw_batch.put(None)
        raise e


def _parse_batch(args: Preprocessing):
    """Parse CIF batches from queue and put structures in another queue."""
    parser = CifParser()

    next_batch = args.queue_raw_batch.get()
    while next_batch is not None:
        structures = parser.parse(next_batch)

        while True:
            time.sleep(0.001)

            if args.finished.is_set():
                args.queue_structures.put(None)
                return

            if args.queue_structures.qsize() < args.prefetch:
                break

        args.queue_structures.put(structures)

        next_batch = args.queue_raw_batch.get()

    args.queue_raw_batch.put(None)
    args.queue_structures.put(None)
    args.finished.wait()


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
    HDF5Dataset.from_structures(
        args.destination, _queue_iterator(args.queue_structures, args.num_workers)
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

        def update_progress(batch: str) -> Structures:
            progress.n = batch_loader.fetched_size
            progress.refresh()
            return parser.parse(batch)

        HDF5Dataset.from_structures(
            args.destination,
            map(update_progress, batch_loader),
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

    saver_process = mp.Process(target=_save_dataset, args=(args,))
    saver_process.start()

    saver_process.join()
    args.finished.set()
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
        )
    )


if __name__ == "__main__":
    main()
