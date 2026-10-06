# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "tqdm"]
# ///
"""Convert the Materials Project summary collection to a single CIF file.

The collection is downloaded from the public open data bucket of the Materials
Project, no API key is needed. Deprecated materials are skipped. The output is
the input of the preprocessing CLI:

    uv run scripts/materials_project_to_cif.py summary.parquet materials_project.cif
    uv run preprocessing materials_project.cif materials_project.hdf5

Labels: total energy is "energy_per_atom" (with the corrections of the
Materials Project) times the number of sites, formation energy, energy above
the hull and band gap are copied from the collection.
"""

import argparse
import os
import urllib.request
from typing import Any, Iterator, TextIO

import numpy as np
import pyarrow.parquet as pq
import tqdm

from _cif_writer import write_block

URL = (
    "https://materialsproject-build.s3.amazonaws.com/collections/summary/"
    "version=2026-09-28/"
    "part-00000-2cc5f2cc-2a79-4051-b013-2f9dec6a302e-c000.zstd.parquet"
)

COLUMNS = [
    "material_id",
    "deprecated",
    "nsites",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "structure.lattice.matrix",
    "structure.sites.list.element.abc",
    "structure.sites.list.element.species",
]


class DisorderedStructure(ValueError):
    """Raised when a site is not fully occupied by a single element."""


def download(url: str, destination: str | os.PathLike) -> None:
    """Download the collection, skipped if the file already exists."""
    if os.path.exists(destination):
        return

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

    with tqdm.tqdm(desc=f"downloading {url}", unit="B", unit_scale=True) as progress:

        def update(blocks: int, block_size: int, total_size: int) -> None:
            progress.total = total_size
            progress.update(min(blocks * block_size, total_size) - progress.n)

        partial = f"{destination}.part"
        urllib.request.urlretrieve(url, partial, reporthook=update)
        os.replace(partial, destination)


def iter_rows(
    source: str | os.PathLike, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Iterate over the materials of the collection, deprecated ones included."""
    parquet = pq.ParquetFile(source)

    for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
        yield from batch.to_pylist()


def site_symbols(row: dict[str, Any]) -> list[str]:
    """Get the element of each site.

    Raises:
        DisorderedStructure: If a site is not fully occupied by one element.
    """
    symbols = []

    for site in row["structure"]["sites"]:
        species = site["species"]

        if len(species) != 1 or abs(species[0]["occu"] - 1.0) > 1e-6:
            raise DisorderedStructure(
                f'{row["material_id"]}: site occupied by {species}'
            )

        symbols.append(species[0]["element"])

    return symbols


def write_row(fp: TextIO, row: dict[str, Any]) -> None:
    """Write a material of the collection as a P1 CIF data block.

    Raises:
        DisorderedStructure: If a site is not fully occupied by one element.
    """
    symbols = site_symbols(row)
    structure = row["structure"]

    total_energy = None
    if row["energy_per_atom"] is not None:
        total_energy = row["energy_per_atom"] * len(symbols)

    write_block(
        fp,
        row["material_id"],
        np.array(structure["lattice"]["matrix"], dtype=np.float64),
        np.array([site["abc"] for site in structure["sites"]], dtype=np.float64),
        symbols,
        total_energy=total_energy,
        energy_above_hull=row["energy_above_hull"],
        band_gap=row["band_gap"],
        formation_energy_per_atom=row["formation_energy_per_atom"],
    )


def convert(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    limit: int | None = None,
) -> dict[str, int]:
    """Write the materials of the collection to a single CIF file.

    Args:
        source: Summary collection (parquet format).
        destination: Output file (cif format).
        limit: Maximum number of structures written.

    Returns:
        Number of structures written, deprecated and disordered.
    """
    total = pq.ParquetFile(source).metadata.num_rows
    counts = {"written": 0, "deprecated": 0, "disordered": 0}

    with open(destination, "w", encoding="utf-8", buffering=1 << 20) as fp:
        for row in tqdm.tqdm(iter_rows(source), total=total, unit="struct"):
            if limit is not None and counts["written"] >= limit:
                break

            if row["deprecated"]:
                counts["deprecated"] += 1
                continue

            try:
                write_row(fp, row)
            except DisorderedStructure:
                counts["disordered"] += 1
                continue

            counts["written"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "source", help="summary collection (parquet format), downloaded if missing"
    )
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--limit", type=int, default=None, help="maximum number of structures"
    )
    args = parser.parse_args()

    download(args.url, args.source)
    counts = convert(args.source, args.destination, args.limit)

    print(
        f'{counts["written"]} structures written to {args.destination} '
        f'({counts["deprecated"]} deprecated and {counts["disordered"]} '
        "disordered skipped)"
    )


if __name__ == "__main__":
    main()
