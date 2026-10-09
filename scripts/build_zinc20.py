"""Build a dataset of 3D molecules of ZINC20 in HDF5 format.

ZINC20 (Irwin et al., 2020) gives its molecules in 3D, ready to dock, in
tranches: files of molecules of the same weight, logP, reactivity,
purchasability, pH and charge. The tranches selected are downloaded one after
the other, their molecules written without any filter and every molecule is
checked against its file. Molecules have no label.

A tranche has a code of six letters, as BAAARN: weight, logP, reactivity,
purchasability, pH, charge. They are selected by a regular expression on this
code, every tranche by default. See what a selection holds before building it:

    uv run python scripts/build_zinc20.py datasets --select "B[A-C]AARN" --list
    uv run python scripts/build_zinc20.py datasets --select "B[A-C]AARN" --name small

The files are read from the copy of ZINC that its authors keep in the open
data of Amazon (the bucket zinc3d, read without an account), several at a
time. Everything is 54,000 files and 534 GB to download, and more than twice
as much once written: give a destination on a large disk. The build can be
stopped and started again: the list of the files and the files downloaded are
kept, and --download-only stops before the HDF5 file is written.

A file built from ZINC must not be published: ZINC is free to use, but
significant portions of it may not be redistributed without the written
permission of its authors.
"""

import argparse
import gzip
import http.client
import itertools
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

from _structures_writer import Entry, check, save, summary
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset

# https://registry.opendata.aws/zinc15/: the tranches of ZINC20 in 3D
BUCKET = "https://zinc3d.s3.amazonaws.com"
PREFIX = "tranches/"
KEY = re.compile(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>")
NEXT = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")
# a file of molecules of a tranche: tranches/BA/AARN/BAAARN.xaa.sdf.gz
SDF = re.compile(r"tranches/[A-Z]{2}/[A-Z]{4}/(([A-Z]{6})\.\w+\.sdf\.gz)")
# the shortest and the longest wait after a request which failed, in seconds
FIRST_WAIT = 5.0
LONGEST_WAIT = 600.0
HEADERS = {"User-Agent": "torchms-dataset-build/0.1 (scripts/build_zinc20.py)"}


def log(message: str) -> None:
    """Print a line with the time: the build runs for hours, often unattended."""
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', flush=True)


@dataclass(frozen=True)
class TrancheFile:
    """A file of molecules of a tranche, and its size."""

    tranche: str
    name: str
    size: int

    @property
    def url(self) -> str:
        """Address of the file in the bucket."""
        return f"{BUCKET}/{PREFIX}{self.tranche[:2]}/{self.tranche[2:]}/{self.name}"


def open_url(url: str, timeout: float = 120):
    """Open a URL of the bucket."""
    request = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(request, timeout=timeout)


def retry(action, what: str):
    """Result of an action on the bucket, tried again for as long as it fails."""
    wait = FIRST_WAIT

    while True:
        try:
            return action()
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                raise
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            reason = type(error).__name__

        log(f"failed ({reason}), waiting {wait:.0f} s: {what}")
        time.sleep(wait)
        wait = min(2 * wait, LONGEST_WAIT)


def bucket_files(raw: Path) -> list[TrancheFile]:
    """Every file of molecules of the bucket, listed once and kept."""
    cache = raw / "_listing.json"

    if not cache.exists():
        files, token, pages = [], None, 0

        while True:
            url = f"{BUCKET}/?list-type=2&prefix={PREFIX}&max-keys=1000"
            if token is not None:
                url += "&continuation-token=" + urllib.parse.quote(token, safe="")

            def page(url: str = url) -> str:
                with open_url(url) as response:
                    return response.read().decode()

            text = retry(page, "listing the bucket")
            pages += 1

            # the bucket also holds the tranches in the formats of docking
            for key, size in KEY.findall(text):
                match = SDF.fullmatch(key)
                if match is not None:
                    files.append([match.group(2), match.group(1), int(size)])

            following = NEXT.search(text)
            if following is None:
                break
            token = following.group(1).replace("&amp;", "&")

            if pages % 50 == 0:
                log(f"listing the bucket: {pages} pages, {len(files)} files")

        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_name(cache.name + ".part")
        partial.write_text(json.dumps(files))
        os.replace(partial, cache)

    return [TrancheFile(*row) for row in json.loads(cache.read_text())]


def tranche_files(select: str, raw: Path) -> list[TrancheFile]:
    """Files of the tranches whose code matches a regular expression, by name."""
    pattern = re.compile(select)
    files = [file for file in bucket_files(raw) if pattern.fullmatch(file.tranche)]

    return sorted(files, key=lambda file: (file.tranche, file.name))


def fetch(tranche_file: TrancheFile, raw: Path) -> int:
    """Download a file of a tranche, skipped if it is already there.

    Returns:
        Number of bytes downloaded.
    """
    destination = raw / tranche_file.tranche / tranche_file.name
    if destination.exists():
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    def download() -> None:
        with open_url(tranche_file.url) as response, open(partial, "wb") as file:
            shutil.copyfileobj(response, file, 1 << 20)

        if partial.stat().st_size != tranche_file.size:
            raise OSError(f"{partial} has not the size of its file")

    retry(download, tranche_file.name)
    os.replace(partial, destination)

    return tranche_file.size


def download_all(files: list[TrancheFile], raw: Path, streams: int) -> None:
    """Download the files of the tranches, several at a time."""
    total = sum(file.size for file in files)
    done, fetched, start = 0, 0, time.time()

    with ThreadPoolExecutor(streams) as pool:
        results = pool.map(lambda file: fetch(file, raw), files)

        for count, (tranche_file, size) in enumerate(zip(files, results), start=1):
            done += tranche_file.size
            fetched += size

            if count % 500 == 0 or count == len(files):
                rate = fetched / max(time.time() - start, 1.0)
                left = (total - done) / rate / 3600 if rate > 0 else 0.0
                log(
                    f"downloaded {count}/{len(files)} files, {done / 1e9:.1f} of "
                    f"{total / 1e9:.1f} GB, {rate / 1e6:.1f} MB/s, about {left:.0f} h left"
                )


def molecules(file_name: Path) -> Iterator[Chem.Mol]:
    """Molecules of a file of a tranche, with their hydrogens, as they are given."""
    with gzip.open(file_name) as file:
        for mol in Chem.ForwardSDMolSupplier(file, removeHs=False, sanitize=False):
            check(mol is not None, f"a molecule of {file_name} cannot be read")
            yield mol


def entries(files: list[TrancheFile], raw: Path, index_file: Path) -> Iterator[Entry]:
    """Entries of the molecules of the files, the row of each one written in the index."""
    schema = pa.schema([("zinc_id", pa.string()), ("tranche", pa.string())])

    with pq.ParquetWriter(index_file, schema) as writer:
        for count, tranche_file in enumerate(files, start=1):
            path = raw / tranche_file.tranche / tranche_file.name
            fetch(tranche_file, raw)

            names = []
            for mol in molecules(path):
                names.append(mol.GetProp("_Name"))

                yield Entry(
                    numbers=np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
                    coords=mol.GetConformer().GetPositions(),
                )

            tranche = [tranche_file.tranche] * len(names)
            writer.write_table(
                pa.table({"zinc_id": names, "tranche": tranche}, schema=schema)
            )
            if count % 500 == 0 or count == len(files):
                log(f"read {count}/{len(files)} files")


def listed_molecules(file_name: Path) -> Iterator[tuple[str, list[str], np.ndarray]]:
    """Name, symbols and positions of the molecules of a file, read from its text.

    A molecule is a block ended by "$$$$": its name, two lines, the number of
    its atoms in the first three characters of the fourth line, then one line
    per atom, three coordinates of ten characters and its symbol.
    """
    with gzip.open(file_name, "rt") as file:
        blocks = file.read().split("$$$$\n")

    for block in blocks:
        if block.strip() == "":
            continue

        lines = block.split("\n")
        n_atoms = int(lines[3][:3])
        atoms = lines[4 : 4 + n_atoms]
        positions = np.array(
            [[float(line[10 * k : 10 * k + 10]) for k in range(3)] for line in atoms]
        )

        yield lines[0].strip(), [line[31:34].strip() for line in atoms], positions


def verify(
    files: list[TrancheFile], raw: Path, hdf5_file: Path, index_file: Path
) -> None:
    """Compare every molecule of the dataset with its file, read another way."""
    dataset = HDF5Dataset(hdf5_file)
    index = pq.ParquetFile(index_file)
    check(index.metadata.num_rows == len(dataset), "rows of the index")
    names = (
        name
        for batch in index.iter_batches(columns=["zinc_id"])
        for name in batch.column("zinc_id").to_pylist()
    )
    row = 0

    for count, tranche_file in enumerate(files, start=1):
        path = raw / tranche_file.tranche / tranche_file.name
        listed = list(listed_molecules(path))
        check(row + len(listed) <= len(dataset), "dataset is shorter than its source")
        if len(listed) == 0:
            continue

        # the molecules of a file are read from the dataset in one call
        structures = dataset[row : row + len(listed)]
        where = tranche_file.name

        expected = [name for name, _, _ in listed]
        check(list(itertools.islice(names, len(listed))) == expected, f"{where}: names")
        check(not bool(structures.pbc.any()), f"{where}: not periodic")
        check(
            structures.num_atoms.tolist() == [len(symbols) for _, symbols, _ in listed],
            f"{where}: number of atoms",
        )
        numbers = [symbol_to_number(s) for _, symbols, _ in listed for s in symbols]
        check(structures.z.tolist() == numbers, f"{where}: elements")

        positions = np.concatenate([positions for _, _, positions in listed])
        diff = np.abs(structures.positions().double().numpy() - positions).max()
        check(diff < 1e-4, f"{where}: positions differ by {diff:.2e}")

        row += len(listed)
        if count % 500 == 0 or count == len(files):
            log(f"verified {count}/{len(files)} files, {row} molecules")

    check(row == len(dataset), f"{len(dataset)} molecules, {row} in the source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--select",
        default="[A-Z]{6}",
        help='regular expression on the codes, "B[A-C]AARN"; every tranche by default',
    )
    parser.add_argument("--name", default="all", help="name of the file written")
    parser.add_argument(
        "--list", action="store_true", help="list the files selected, build nothing"
    )
    parser.add_argument(
        "--streams", type=int, default=8, help="files downloaded at the same time"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download the files selected, write no HDF5 file",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    root = Path(args.destination)
    raw = root / "raw" / "zinc20"
    hdf5_file = root / "zinc20_3d" / f"{args.name}.hdf5"
    index_file = hdf5_file.with_name(f"{args.name}_index.parquet")

    files = tranche_files(args.select, raw)
    check(len(files) > 0, f'no tranche matches "{args.select}"')

    tranches = sorted({file.tranche for file in files})
    size = sum(file.size for file in files)
    log(
        f"{len(files)} files of {len(tranches)} tranches, {size / 1e9:.2f} GB "
        f"compressed: {tranches[:8]}{' ...' if len(tranches) > 8 else ''}"
    )
    if args.list:
        return

    download_all(files, raw, args.streams)
    if args.download_only:
        return

    hdf5_file.parent.mkdir(parents=True, exist_ok=True)
    save(hdf5_file, entries(files, raw, index_file), batch_size=4096)
    verify(files, raw, hdf5_file, index_file)

    print(summary(hdf5_file))
    print("every molecule verified against its source")
    print("local file: significant portions of ZINC must not be published")


if __name__ == "__main__":
    main()
