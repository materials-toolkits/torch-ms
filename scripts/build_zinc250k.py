"""Build the ZINC250k dataset in HDF5 format, with one 3D conformer per molecule.

The 249,455 molecules of ZINC250k (Gómez-Bombarelli et al., 2018) are given as
SMILES, without positions. The geometry of this dataset is not the one of a
source: hydrogens are added and one conformer per molecule is generated with
RDKit (ETKDGv3, then MMFF94), with a fixed seed. Run inside the project
environment:

    uv run python scripts/build_zinc250k.py datasets
"""

import argparse
import os
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from _structures_writer import Entry, check, download, save, summary
from torchms.data.hdf5 import HDF5Dataset

URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
    "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
MD5 = "b59078b2b04c6e9431280e3dc42048d5"
NUM_MOLECULES = 249455

SEED = 42
MAX_ITERATIONS = 2000

# columns of the source, and whether the force field converged
PROPERTIES = ("logp", "qed", "sas", "mmff_converged")


def read_source(file_name: Path) -> pd.DataFrame:
    """Table of the source: smiles, logp, qed, sas. Its SMILES end with a new line."""
    table = pd.read_csv(file_name)
    table.columns = [column.lower() for column in table.columns]
    table["smiles"] = table["smiles"].str.strip()

    check(len(table) == NUM_MOLECULES, f"{len(table)} molecules in the source")

    return table


def molecule(smiles: str) -> Chem.Mol:
    """Molecule of a SMILES with its hydrogens, in the order of RDKit."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def conformer(smiles: str) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Generate the conformer of a molecule.

    Returns:
        Atomic numbers, positions in Å and whether MMFF94 converged. None if
        no conformer could be embedded.
    """
    mol = molecule(smiles)

    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.numThreads = 1

    if AllChem.EmbedMolecule(mol, params) != 0:
        # from random coordinates: slower, succeeds on more molecules
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

    converged = False
    if AllChem.MMFFHasAllMoleculeParams(mol):
        converged = AllChem.MMFFOptimizeMolecule(mol, maxIters=MAX_ITERATIONS) == 0

    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

    return numbers, mol.GetConformer().GetPositions(), converged


def conformers(smiles: list[str], workers: int) -> Iterator[tuple | None]:
    """Conformer of each molecule, in the order of the molecules."""
    with Pool(workers) as pool:
        yield from tqdm.tqdm(
            pool.imap(conformer, smiles, chunksize=64),
            total=len(smiles),
            desc="conformers",
            unit="mol",
        )


def entries(table: pd.DataFrame, workers: int):
    """Entries of the molecules with a conformer, and the rows they come from."""
    kept: list[int] = []

    def iterate() -> Iterator[Entry]:
        rows = table.itertuples(index=True)

        for row, result in zip(rows, conformers(table["smiles"].tolist(), workers)):
            if result is None:
                continue

            numbers, positions, converged = result
            kept.append(row.Index)

            yield Entry(
                numbers=numbers,
                coords=positions,
                properties={
                    "logp": row.logp,
                    "qed": row.qed,
                    "sas": row.sas,
                    "mmff_converged": float(converged),
                },
            )

    return iterate(), kept


def verify(table: pd.DataFrame, hdf5_file: Path, sample: int = 500) -> None:
    """Check every molecule of the dataset against its SMILES.

    The atoms are the ones of the SMILES with its hydrogens, every bond is
    shorter than 2.5 Å, no two atoms are closer than 0.5 Å and the properties
    are the ones of the source. A sample is generated again: the build is
    reproducible.
    """
    dataset = HDF5Dataset(hdf5_file)
    check(len(dataset) == len(table), f"{len(dataset)} molecules in the dataset")

    structures = dataset[:]
    check(not bool(structures.pbc.any()), "molecules are not periodic")

    positions = structures.positions().double().numpy()
    z, ptr = structures.z.numpy(), structures.ptr.numpy()

    for name in ("logp", "qed", "sas"):
        stored = structures.properties[name].double().numpy()
        check(
            bool(np.allclose(stored, table[name].to_numpy(), rtol=1e-6, atol=1e-6)),
            f"property {name}",
        )

    for row, smiles in enumerate(tqdm.tqdm(table["smiles"], "verifying", unit="mol")):
        where = f"molecule {row} ({smiles})"
        mol = molecule(smiles)
        atoms = positions[ptr[row] : ptr[row + 1]]

        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        check(z[ptr[row] : ptr[row + 1]].tolist() == numbers, f"{where}: elements")

        dists = np.linalg.norm(atoms[:, None] - atoms[None, :], axis=-1)
        bonds = np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        )
        longest = dists[bonds[:, 0], bonds[:, 1]].max()
        check(longest < 2.5, f"{where}: a bond of {longest:.2f} Å")

        np.fill_diagonal(dists, np.inf)
        check(dists.min() > 0.5, f"{where}: atoms {dists.min():.2f} Å apart")

    rows = np.random.default_rng(0).choice(len(table), min(sample, len(table)), False)
    for row in tqdm.tqdm(np.sort(rows), "generating again", unit="mol"):
        result = conformer(table["smiles"].iloc[row])
        check(result is not None, f"molecule {row}: no conformer")
        assert result is not None
        diff = np.abs(positions[ptr[row] : ptr[row + 1]] - result[1]).max()
        check(diff < 1e-4, f"molecule {row}: positions differ by {diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1)
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="first molecules only, to try the build"
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    root = Path(args.destination)
    source = root / "raw" / "zinc250k" / "250k_rndm_zinc_drugs_clean_3.csv"
    name = "all" if args.limit is None else f"first_{args.limit}"
    hdf5_file = root / "zinc250k" / f"{name}.hdf5"

    download(URL, source, MD5)
    table = read_source(source)
    if args.limit is not None:
        table = table.iloc[: args.limit]

    iterator, kept = entries(table, args.workers)
    save(hdf5_file, iterator, PROPERTIES)

    failed = table.drop(index=kept)
    table = table.loc[kept].reset_index(names="source_row")
    table[["source_row", "smiles"]].to_parquet(
        hdf5_file.with_name(f"{name}_index.parquet"), index=False
    )

    verify(table, hdf5_file)

    converged = HDF5Dataset(hdf5_file).dataset("properties/mmff_converged")[...]
    print(summary(hdf5_file))
    print(f"conformers generated with RDKit {rdkit.__version__}, seed {SEED}")
    print(
        f"skipped: {len(failed)} without conformer {failed['smiles'].tolist()[:20]}; "
        f"kept without converged MMFF94: {int((converged == 0).sum())}"
    )
    print("every molecule verified against its SMILES")


if __name__ == "__main__":
    main()
