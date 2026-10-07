"""Build the OQMD dataset in HDF5 format from the SQL dump of the database.

The dump is downloaded, its tables are extracted, one structure is selected
per entry, converted to CIF, preprocessed without any filter and every
structure is checked against the tables. Run inside the project environment:

    uv run python scripts/build_oqmd.py datasets            # all.hdf5
    uv run python scripts/build_oqmd.py datasets --unique   # unique.hdf5

With --unique, only the entry with the lowest energy is kept among the entries
marked as duplicates in the database.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from oqmd_extract import TABLES, extract
from oqmd_to_cif import LATTICE, convert
from torchms.data.constants import symbol_to_number
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.preprocessing import Preprocessing, preprocessing

URL = "https://static.oqmd.org/static/downloads/qmdb__v1_8__022026.sql.gz"


def check(condition: bool, message: str) -> None:
    """Stop the build when a verification fails."""
    if not condition:
        raise RuntimeError(f"verification failed: {message}")


def download(url: str, destination: Path) -> None:
    """Download the dump, skipped if the file already exists."""
    if destination.exists():
        return

    with tqdm.tqdm(desc=f"downloading {url}", unit="B", unit_scale=True) as progress:

        def update(blocks: int, block_size: int, total_size: int) -> None:
            progress.total = total_size
            progress.update(min(blocks * block_size, total_size) - progress.n)

        partial = destination.with_name(destination.name + ".part")
        urllib.request.urlretrieve(url, partial, reporthook=update)
        os.replace(partial, destination)


def valid_symbols(symbols: pd.Series) -> set[str]:
    """Elements of the atoms table known by torchms."""
    valid = set()
    for symbol in symbols.dropna().unique():
        try:
            symbol_to_number(symbol)
        except (KeyError, ValueError):
            continue
        valid.add(symbol)
    return valid


def verify(selected: pd.DataFrame, atoms: pd.DataFrame, hdf5_file: Path) -> None:
    """Compare every structure of the dataset with the tables of the dump."""
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == len(selected), f"{len(dataset)} != {len(selected)} entries")

    structures = dataset[:]
    num_atoms = selected["natoms"].to_numpy(dtype=np.int64)
    check(bool((structures.num_atoms.numpy() == num_atoms).all()), "number of atoms")

    # atoms of each structure, in the order of the selected entries
    order = np.argsort(atoms["structure_id"].to_numpy(), kind="stable")
    starts = np.searchsorted(
        atoms["structure_id"].to_numpy()[order], selected["structure_id"].to_numpy()
    )
    ptr = np.concatenate([[0], np.cumsum(num_atoms)])
    batch = np.repeat(np.arange(len(selected)), num_atoms)
    index = order[starts[batch] + np.arange(ptr[-1]) - ptr[batch]]

    check(
        bool(
            (
                atoms["structure_id"].to_numpy()[index]
                == selected["structure_id"].to_numpy()[batch]
            ).all()
        ),
        "atoms of the tables do not match the selected structures",
    )

    numbers = {s: symbol_to_number(s) for s in atoms["element_id"].iloc[index].unique()}
    z = atoms["element_id"].iloc[index].map(numbers).to_numpy(dtype=np.int64)
    check(bool((structures.z.numpy() == z).all()), "elements")

    frac = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)[index] % 1.0
    diff = np.abs(structures.x.double().numpy() - frac)
    diff = np.minimum(diff, 1.0 - diff)
    check(diff.max() < 1e-5, f"coordinates differ by {diff.max():.2e}")

    # the lattice is rebuilt in the standard orientation: compare the metric
    lattice = selected[LATTICE].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    gram = lattice @ lattice.transpose(0, 2, 1)
    stored = structures.lattice.double().numpy()
    error = np.abs(stored @ stored.transpose(0, 2, 1) - gram).max(axis=(1, 2))
    worst = (error / gram.max(axis=(1, 2))).max()
    check(worst < 1e-4, f"lattice differs by {worst:.2e}")

    labels = {
        "formation_energy_per_atom": "delta_e",
        "energy_above_hull": "stability",
        "band_gap": "band_gap",
    }
    for label, column in labels.items():
        check(
            torch.allclose(
                getattr(structures, label).double(),
                torch.from_numpy(selected[column].to_numpy(dtype=np.float64)),
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            label,
        )

    check(
        torch.allclose(
            structures.total_energy.double() / structures.num_atoms,
            torch.from_numpy(selected["energy_pa"].to_numpy(dtype=np.float64)),
            rtol=1e-6,
            atol=1e-5,
        ),
        "total_energy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--unique",
        action="store_true",
        help="keep the entry with the lowest energy among duplicates",
    )
    args = parser.parse_args()

    name = "unique" if args.unique else "all"

    root = Path(args.destination)
    dump = root / "raw" / "oqmd" / args.url.rsplit("/", maxsplit=1)[-1]
    tables = root / "raw" / "oqmd" / "tables"
    cif_file = root / "cif" / "oqmd" / f"{name}.cif"
    hdf5_file = root / "oqmd" / f"{name}.hdf5"

    for file_name in (dump, cif_file, hdf5_file):
        file_name.parent.mkdir(parents=True, exist_ok=True)

    download(args.url, dump)

    if not all((tables / f"{table}.parquet").exists() for table in TABLES):
        partial = tables.with_name("tables.part")
        counts = extract(dump, partial, TABLES, args.num_workers)
        os.replace(partial, tables)
        print(json.dumps(counts, indent=2))

    symbols = pd.read_parquet(tables / "atoms.parquet", columns=["element_id"])
    selected, atoms, report = convert(
        tables, cif_file, valid_symbols(symbols["element_id"]), unique=args.unique
    )
    del symbols

    preprocessing(Preprocessing(source=cif_file, destination=hdf5_file, batch_size=256))
    verify(selected, atoms, hdf5_file)

    selected[["entry_id", "calculation_id", "structure_id"]].to_parquet(
        hdf5_file.with_name(f"{name}_index.parquet"), index=False
    )
    with open(hdf5_file.with_name(f"{name}_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report, indent=2))
    print(
        f'{len(selected)} structures ({int(selected["natoms"].sum())} atoms, up to '
        f'{int(selected["natoms"].max())} per structure), '
        f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}"
    )
    print("every structure verified against the tables of the dump")


if __name__ == "__main__":
    main()
