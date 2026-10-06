# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tqdm"]
# ///
"""Convert a CSV with a CIF column to a single concatenated CIF file.

Works with the CDVAE datasets (MP-20, Carbon-24, Perov-5) and the MatterGen
release of Alex-MP-20. CIF must be in P1. The output is the input of the
preprocessing CLI:

    uv run scripts/csv_to_cif.py mp_20/train.csv mp_20_train.cif \\
        --hull-col e_above_hull --band-gap-col band_gap
    uv run preprocessing mp_20_train.cif mp_20_train.hdf5
"""

import argparse
import csv
import re
import sys

import tqdm

from _cif_writer import property_tags

HEADER = re.compile(r"^data_\S*[^\S\n]*\n", re.MULTILINE)


def read_float(row: dict[str, str], column: str | None) -> float | None:
    """Read an optional property, None if the column is unset or empty."""
    if column is None or row[column] == "":
        return None
    return float(row[column])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", help="input file (csv format)")
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument("--cif-col", default="cif")
    parser.add_argument("--id-col", default="material_id")
    parser.add_argument("--energy-col", default=None, help="total energy in eV")
    parser.add_argument(
        "--hull-col", default=None, help="energy above the hull in eV/atom"
    )
    parser.add_argument("--band-gap-col", default=None, help="band gap in eV")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    count = 0
    with (
        open(args.source, "r", encoding="utf-8", newline="") as src,
        open(args.destination, "w", encoding="utf-8") as dst,
    ):
        for row in tqdm.tqdm(csv.DictReader(src), desc=args.source, unit="struct"):
            cif = row[args.cif_col]
            name = re.sub(r"\s+", "_", row[args.id_col])

            tags = property_tags(
                total_energy=read_float(row, args.energy_col),
                energy_above_hull=read_float(row, args.hull_col),
                band_gap=read_float(row, args.band_gap_col),
            )

            cif, replaced = HEADER.subn(f"data_{name}\n{tags}", cif, count=1)
            if replaced != 1:
                raise ValueError(f'no data block in the CIF of "{name}"')

            dst.write(cif.rstrip("\n") + "\n\n")
            count += 1

    print(f"{count} structures written to {args.destination}")


if __name__ == "__main__":
    main()
