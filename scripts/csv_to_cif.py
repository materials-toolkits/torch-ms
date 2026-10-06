# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tqdm"]
# ///
"""Convert a CSV with a CIF column to a single concatenated CIF file.

Works with the CDVAE datasets (MP-20, Carbon-24, Perov-5) and the MatterGen
release of Alex-MP-20. CIF must be in P1. The output is the input of the
preprocessing CLI:

    uv run scripts/csv_to_cif.py mp_20/train.csv mp_20_train.cif \\
        --formation-energy-col formation_energy_per_atom \\
        --hull-col e_above_hull --band-gap-col band_gap
    uv run preprocessing mp_20_train.cif mp_20_train.hdf5
"""

import argparse
import csv
import os
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


def count_atoms(cif: str) -> int:
    """Count the rows of the atom site loop of a single block P1 CIF."""
    lines = [line.strip() for line in cif.splitlines()]
    headers = [i for i, line in enumerate(lines) if line.startswith("_atom_site_")]

    if len(headers) == 0:
        raise ValueError("no atom site loop in the CIF")

    return sum(
        1
        for line in lines[headers[-1] + 1 :]
        if line and not line.startswith(("#", "_", "loop_", "data_"))
    )


def convert(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    cif_col: str = "cif",
    id_col: str = "material_id",
    energy_col: str | None = None,
    energy_per_atom_col: str | None = None,
    formation_energy_col: str | None = None,
    hull_col: str | None = None,
    band_gap_col: str | None = None,
) -> int:
    """Write the CIF of each row of a CSV file to a single CIF file.

    Args:
        source: Input file (csv format).
        destination: Output file (cif format).
        cif_col: Column with the CIF of the structure.
        id_col: Column with the name of the structure.
        energy_col: Column with the total energy in eV.
        energy_per_atom_col: Column with the total energy in eV/atom, multiplied
            by the number of atoms to get the total energy.
        formation_energy_col: Column with the formation energy in eV/atom.
        hull_col: Column with the energy above the hull in eV/atom.
        band_gap_col: Column with the band gap in eV.

    Returns:
        Number of structures written.
    """
    if energy_col is not None and energy_per_atom_col is not None:
        raise ValueError("energy_col and energy_per_atom_col are mutually exclusive")

    csv.field_size_limit(sys.maxsize)

    count = 0
    with (
        open(source, "r", encoding="utf-8", newline="") as src,
        open(destination, "w", encoding="utf-8") as dst,
    ):
        for row in tqdm.tqdm(csv.DictReader(src), desc=str(source), unit="struct"):
            cif = row[cif_col]
            name = re.sub(r"\s+", "_", row[id_col])

            total_energy = read_float(row, energy_col)
            energy_per_atom = read_float(row, energy_per_atom_col)
            if energy_per_atom is not None:
                total_energy = energy_per_atom * count_atoms(cif)

            tags = property_tags(
                total_energy=total_energy,
                energy_above_hull=read_float(row, hull_col),
                band_gap=read_float(row, band_gap_col),
                formation_energy_per_atom=read_float(row, formation_energy_col),
            )

            cif, replaced = HEADER.subn(f"data_{name}\n{tags}", cif, count=1)
            if replaced != 1:
                raise ValueError(f'no data block in the CIF of "{name}"')

            dst.write(cif.rstrip("\n") + "\n\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", help="input file (csv format)")
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument("--cif-col", default="cif")
    parser.add_argument("--id-col", default="material_id")
    parser.add_argument("--energy-col", default=None, help="total energy in eV")
    parser.add_argument(
        "--energy-per-atom-col",
        default=None,
        help="total energy in eV/atom, converted to the total energy",
    )
    parser.add_argument(
        "--formation-energy-col", default=None, help="formation energy in eV/atom"
    )
    parser.add_argument(
        "--hull-col", default=None, help="energy above the hull in eV/atom"
    )
    parser.add_argument("--band-gap-col", default=None, help="band gap in eV")
    args = parser.parse_args()

    count = convert(
        args.source,
        args.destination,
        cif_col=args.cif_col,
        id_col=args.id_col,
        energy_col=args.energy_col,
        energy_per_atom_col=args.energy_per_atom_col,
        formation_energy_col=args.formation_energy_col,
        hull_col=args.hull_col,
        band_gap_col=args.band_gap_col,
    )

    print(f"{count} structures written to {args.destination}")


if __name__ == "__main__":
    main()
