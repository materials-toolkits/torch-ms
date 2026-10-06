# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "tqdm"]
# ///
"""Convert tables extracted from the OQMD dump to a single CIF file.

One structure is selected for each entry of the database: the output of its
converged calculation with the lowest energy, among "static" and "standard"
calculations (LDA excluded). Inconsistent entries are skipped and counted.

    uv run scripts/oqmd_extract.py qmdb__v1_8__022026.sql.gz tables/
    uv run scripts/oqmd_to_cif.py tables/ oqmd.cif
    uv run preprocessing oqmd.cif oqmd.hdf5

Labels: total energy is "calculations.energy_pa" times the number of atoms,
formation energy is "formation_energies.delta_e", energy above the hull is
"formation_energies.stability" (negative below the hull) and band gap is
"calculations.band_gap".
"""

import argparse
import json
import os
from typing import Collection, TextIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm

from _cif_writer import write_block

LATTICE = ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]
FIT = "standard"


def count_atoms(tables: str | os.PathLike) -> pd.Series:
    """Count the rows of the atoms table for each structure."""
    parquet = pq.ParquetFile(os.path.join(tables, "atoms.parquet"))
    counts = np.zeros(0, dtype=np.int64)

    for batch in parquet.iter_batches(batch_size=1 << 22, columns=["structure_id"]):
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        ids = ids[~np.isnan(ids)].astype(np.int64) if ids.dtype.kind == "f" else ids
        batch_counts = np.bincount(ids)

        if len(batch_counts) > len(counts):
            counts = np.pad(counts, (0, len(batch_counts) - len(counts)))
        counts[: len(batch_counts)] += batch_counts

    ids = np.flatnonzero(counts)
    return pd.Series(counts[ids], index=ids, name="num_rows")


def select_calculations(
    structures: pd.DataFrame,
    calculations: pd.DataFrame,
    formation_energies: pd.DataFrame,
    atom_counts: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select the structure and the labels of each entry.

    Args:
        structures: Table "structures" (id, natoms and lattice).
        calculations: Table "calculations".
        formation_energies: Table "formation_energies".
        atom_counts: Number of rows of the table "atoms" per structure id.

    Returns:
        One row per entry ordered by entry_id (entry_id, calculation_id,
        structure_id, natoms, energy_pa, band_gap, delta_e, stability and
        lattice), and the number of rows kept or rejected at each step.
    """
    report: dict[str, int] = {}

    # a structure is valid when its number of atoms matches the atoms table
    num_rows = atom_counts.reindex(structures["id"]).fillna(0).to_numpy()
    valid = structures[structures["natoms"].to_numpy() == num_rows]
    valid = valid[valid["natoms"] > 0]
    report["structures"] = len(structures)
    report["structures_with_inconsistent_natoms"] = len(structures) - len(valid)

    label = calculations["label"].fillna("")
    steps = {
        "calculations_not_converged": calculations["converged"].fillna(0) == 1,
        "calculations_not_static_or_standard": label.str.startswith(
            ("static", "standard")
        ),
        "calculations_lda": ~label.str.endswith("lda"),
        "calculations_without_energy": calculations["energy_pa"].notna(),
        "calculations_without_entry": calculations["entry_id"].notna(),
        "calculations_without_valid_output": calculations["output_id"].isin(
            valid["id"]
        ),
    }

    report["calculations"] = len(calculations)
    kept = np.ones(len(calculations), dtype=bool)
    for name, mask in steps.items():
        report[name] = int((kept & ~mask.to_numpy()).sum())
        kept &= mask.to_numpy()

    report["entries_with_calculations"] = int(calculations["entry_id"].nunique())

    # lowest energy of each entry, lowest calculation id when energies are equal
    selected = (
        calculations[kept]
        .sort_values(["entry_id", "energy_pa", "id"], kind="stable")
        .drop_duplicates("entry_id")
        .rename(columns={"id": "calculation_id", "output_id": "structure_id"})
    )
    selected = selected.astype({"entry_id": np.int64, "structure_id": np.int64})
    selected = selected[
        ["entry_id", "calculation_id", "structure_id", "energy_pa", "band_gap"]
    ]
    report["entries_selected"] = len(selected)

    # formation energy of the selected calculation, or of the entry if unique
    fit = formation_energies[formation_energies["fit_id"] == FIT]
    fit = fit.sort_values("id", kind="stable")
    columns = ["delta_e", "stability"]

    by_calculation = fit.dropna(subset=["calculation_id"]).drop_duplicates(
        ["entry_id", "calculation_id"]
    )
    selected = selected.merge(
        by_calculation[["entry_id", "calculation_id"] + columns],
        on=["entry_id", "calculation_id"],
        how="left",
    )

    unique = fit[~fit["entry_id"].duplicated(keep=False)].set_index("entry_id")
    missing = selected["delta_e"].isna() & selected["stability"].isna()
    fallback = unique.reindex(selected.loc[missing, "entry_id"])[columns].to_numpy()
    selected.loc[missing, columns] = fallback

    report["entries_with_formation_energy"] = int(selected["delta_e"].notna().sum())
    report["entries_with_stability"] = int(selected["stability"].notna().sum())
    report["entries_with_band_gap"] = int(selected["band_gap"].notna().sum())

    selected = selected.merge(
        valid[["id", "natoms"] + LATTICE].rename(columns={"id": "structure_id"}),
        on="structure_id",
        how="left",
    )

    return selected.reset_index(drop=True), report


def drop_duplicates(selected: pd.DataFrame, duplicate_of: pd.Series) -> pd.DataFrame:
    """Keep the entry with the lowest energy among duplicates.

    Entries are grouped with the entry they duplicate. When energies are
    equal, the original entry is kept, then the lowest entry id.

    Args:
        selected: One row per entry (see select_calculations).
        duplicate_of: Column "duplicate_of_id" of the table "entries" indexed
            by entry id. An entry which is not a duplicate references itself
            or nothing.

    Returns:
        Rows of selected without duplicates, ordered by entry_id.
    """
    entry_id = selected["entry_id"].to_numpy()
    original = duplicate_of.reindex(entry_id).to_numpy(dtype=np.float64)
    original = np.where(np.isnan(original), entry_id, original).astype(np.int64)

    order = (
        selected.assign(_original=original, _duplicate=original != entry_id)
        .sort_values(
            ["_original", "energy_pa", "_duplicate", "entry_id"], kind="stable"
        )
        .drop_duplicates("_original")
    )

    return selected.loc[np.sort(order.index.to_numpy())].reset_index(drop=True)


def load_atoms(tables: str | os.PathLike, structure_ids: np.ndarray) -> pd.DataFrame:
    """Load the atoms of the given structures, in the order of the dump."""
    parquet = pq.ParquetFile(os.path.join(tables, "atoms.parquet"))
    wanted = np.zeros(int(structure_ids.max()) + 2, dtype=bool)
    wanted[structure_ids] = True

    chunks = []
    for batch in tqdm.tqdm(
        parquet.iter_batches(batch_size=1 << 22),
        desc="loading atoms",
        total=-(-parquet.metadata.num_rows // (1 << 22)),
    ):
        atoms = batch.to_pandas()
        ids = atoms["structure_id"].fillna(-1).to_numpy().astype(np.int64)
        ids = np.where((ids >= 0) & (ids < len(wanted)), ids, len(wanted) - 1)
        chunks.append(atoms[wanted[ids]])

    return pd.concat(chunks, ignore_index=True)


def rejection(
    lattice: np.ndarray,
    frac: np.ndarray,
    symbols: list[str | None],
    occupancy: np.ndarray,
    valid_symbols: Collection[str] | None,
) -> str | None:
    """Get the reason why a structure cannot be written, None if it is valid."""
    if not np.isfinite(lattice).all():
        return "lattice_not_finite"

    lengths = np.linalg.norm(lattice, axis=1)
    if lengths.min() < 1e-3:
        return "lattice_degenerate"

    if abs(np.linalg.det(lattice)) < 1e-6 * lengths.prod():
        return "lattice_degenerate"

    if not np.isfinite(frac).all():
        return "coordinates_not_finite"

    if not np.allclose(occupancy, 1.0):
        return "partial_occupancy"

    if any(symbol is None for symbol in symbols):
        return "unknown_element"

    if valid_symbols is not None and not set(symbols) <= set(valid_symbols):
        return "unknown_element"

    return None


def write_structures(
    fp: TextIO,
    selected: pd.DataFrame,
    atoms: pd.DataFrame,
    valid_symbols: Collection[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Write the selected structures as P1 CIF data blocks.

    Returns:
        Mask of the written rows of selected, and the number of rejected
        structures per reason.
    """
    order = np.argsort(atoms["structure_id"].to_numpy(), kind="stable")
    structure_ids = atoms["structure_id"].to_numpy()[order]
    frac = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)[order]
    symbols = atoms["element_id"].to_numpy(dtype=object)[order]
    occupancy = atoms["occupancy"].to_numpy(dtype=np.float64)[order]

    lattices = selected[LATTICE].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    starts = np.searchsorted(structure_ids, selected["structure_id"].to_numpy())
    stops = starts + selected["natoms"].to_numpy(dtype=np.int64)

    def optional(value: float) -> float | None:
        return None if np.isnan(value) else float(value)

    written = np.zeros(len(selected), dtype=bool)
    rejected: dict[str, int] = {}

    rows = zip(
        selected["entry_id"],
        selected["energy_pa"],
        selected["band_gap"],
        selected["delta_e"],
        selected["stability"],
    )
    for i, (entry_id, energy_pa, band_gap, delta_e, stability) in enumerate(
        tqdm.tqdm(rows, total=len(selected), desc="writing", unit="struct")
    ):
        start, stop = starts[i], stops[i]
        struct_symbols = list(symbols[start:stop])

        reason = rejection(
            lattices[i],
            frac[start:stop],
            struct_symbols,
            occupancy[start:stop],
            valid_symbols,
        )
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        write_block(
            fp,
            f"oqmd-{entry_id}",
            lattices[i],
            frac[start:stop],
            struct_symbols,
            total_energy=energy_pa * (stop - start),
            energy_above_hull=optional(stability),
            band_gap=optional(band_gap),
            formation_energy_per_atom=optional(delta_e),
        )
        written[i] = True

    return written, rejected


def convert(
    tables: str | os.PathLike,
    destination: str | os.PathLike,
    valid_symbols: Collection[str] | None = None,
    unique: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Write the selected structure of each entry to a single CIF file.

    Args:
        tables: Directory of the tables extracted by oqmd_extract.py.
        destination: Output file (cif format).
        valid_symbols: Elements accepted, structures with others are skipped.
        unique: Keep only the entry with the lowest energy among the entries
            marked as duplicates in the database (see drop_duplicates).

    Returns:
        The written rows (see select_calculations) in the order of the file,
        their atoms, and the number of rows kept or rejected at each step.
    """

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(tables, f"{name}.parquet"))

    selected, report = select_calculations(
        read("structures"),
        read("calculations"),
        read("formation_energies"),
        count_atoms(tables),
    )

    duplicates = read("entries").set_index("id")["duplicate_of_id"]

    if unique:
        num_selected = len(selected)
        selected = drop_duplicates(selected, duplicates)
        report["entries_dropped_as_duplicates"] = num_selected - len(selected)

    atoms = load_atoms(tables, selected["structure_id"].to_numpy())

    with open(destination, "w", encoding="utf-8", buffering=1 << 20) as fp:
        written, rejected = write_structures(fp, selected, atoms, valid_symbols)

    for reason, count in rejected.items():
        report[f"entries_rejected_{reason}"] = count
    report["entries_written"] = int(written.sum())

    selected = selected[written].reset_index(drop=True)
    # an entry which is not a duplicate references itself, or nothing
    duplicate_of = duplicates.reindex(selected["entry_id"]).to_numpy(dtype=np.float64)
    report["entries_written_duplicate_of_another"] = int(
        (~np.isnan(duplicate_of) & (duplicate_of != selected["entry_id"])).sum()
    )

    return selected, atoms, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("tables", help="directory of the extracted tables")
    parser.add_argument("destination", help="output file (cif format)")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="keep the entry with the lowest energy among duplicates",
    )
    args = parser.parse_args()

    _, _, report = convert(args.tables, args.destination, unique=args.unique)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
