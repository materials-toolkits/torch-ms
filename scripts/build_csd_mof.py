"""Build a dataset of MOFs of the Cambridge Structural Database in HDF5 format.

The structures of the CSD are not distributed with torchms, and a dataset built
from them must not be published: the CSD MOF subset needs a licence of the
CCDC, and the CSD MOF Collection is free for academic research only. This
script converts CIF files you obtained from the CCDC to a local HDF5 file:

- the CSD MOF Collection, downloaded from the CCDC: its directory of CIF files;
- the MOF subset, exported from your installation of the CSD (ConQuest or the
  CSD Python API): a directory of CIF files, or files with several structures.

Structures with a site which is not fully occupied by one element are skipped
and listed, as the files which cannot be read. Every structure written is
checked against its file. Run inside the project environment:

    uv run python scripts/build_csd_mof.py path/to/cifs datasets/csd_mof/all.hdf5

The file is then loaded by its path: auto_loader("datasets/csd_mof/all.hdf5", ...).
"""

import argparse
import os
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from _cif_reader import in_order, read_entry, read_reference, verify_structure
from _structures_writer import Entry, check, save, summary
from torchms.data.hdf5 import HDF5Dataset

BLOCK = re.compile(r"^data_(\S+)", re.MULTILINE)


def split_blocks(file_name: Path, directory: Path) -> list[Path]:
    """Write each structure of a CIF file with several ones to its own file.

    Returns:
        The files written, named after their data blocks, or the file itself
        if it has one structure.
    """
    text = file_name.read_text(errors="replace")
    blocks = list(BLOCK.finditer(text))

    if len(blocks) <= 1:
        return [file_name]

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for block, following in zip(blocks, [*blocks[1:], None]):
        end = len(text) if following is None else following.start()
        name = re.sub(r"[^\w.-]", "_", block.group(1))
        path = directory / f"{name}.cif"
        check(path not in paths, f"two structures named {name} in {file_name}")
        path.write_text(text[block.start() : end])
        paths.append(path)

    return paths


def cif_files(source: Path, blocks: Path) -> list[Path]:
    """CIF files of one structure each, sorted by name of structure."""
    files = [source] if source.is_file() else sorted(source.rglob("*.cif"))
    check(len(files) > 0, f"no CIF file in {source}")

    paths = [path for file in files for path in split_blocks(file, blocks / file.stem)]
    names = Counter(path.stem for path in paths)
    check(
        max(names.values()) == 1,
        f"structures with the same name: {[n for n, c in names.items() if c > 1][:5]}",
    )

    return sorted(paths, key=lambda path: path.stem)


def entries(paths: list[Path], workers: int):
    """Entries of the structures, the names of the ones kept, and the skipped."""
    kept: list[Path] = []
    skipped: list[dict[str, str]] = []

    def iterate() -> Iterator[Entry]:
        for path, entry in zip(paths, in_order(read_entry, paths, workers, "reading")):
            if isinstance(entry, str):
                skipped.append({"name": path.stem, "reason": entry})
                continue

            kept.append(path)
            yield entry

    return iterate(), kept, skipped


def verify(paths: list[Path], hdf5_file: Path, workers: int) -> None:
    """Compare every structure of the dataset with its file read by pymatgen."""
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == len(paths), f"{len(dataset)} structures in the dataset")

    reader = in_order(read_reference, paths, workers, "verifying")
    for row, (path, reference) in enumerate(zip(paths, reader)):
        verify_structure(dataset[row], reference, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", help="directory of CIF files, or a CIF file")
    parser.add_argument("destination", help="HDF5 file written")
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1)
    )
    args = parser.parse_args()

    hdf5_file = Path(args.destination)
    prefix = hdf5_file.with_suffix("")
    paths = cif_files(Path(args.source), Path(f"{prefix}_blocks"))

    iterator, kept, skipped = entries(paths, args.workers)
    save(hdf5_file, iterator, batch_size=256)
    check(len(kept) > 0, "no structure could be read")

    pd.DataFrame({"name": [path.stem for path in kept]}).to_parquet(
        f"{prefix}_index.parquet", index=False
    )
    pd.DataFrame(skipped, columns=["name", "reason"]).to_csv(
        f"{prefix}_skipped.csv", index=False
    )

    verify(kept, hdf5_file, args.workers)

    reasons = Counter(row["reason"].split(":")[0] for row in skipped)
    print(summary(hdf5_file))
    print(f"skipped: {dict(reasons)}, listed in {prefix}_skipped.csv")
    print("every structure verified against its file read by pymatgen")
    print("local file: the structures of the CSD must not be published")


if __name__ == "__main__":
    main()
