"""Fit the correction of a pretrained energy model to the energies of LeMat-Bulk.

The model predicts the energy of a sample of structures of
datasets/lematerial/all.hdf5, an energy per atom of each element is fitted on
structures from the Materials Project and Alexandria, and the corrected
energies are compared with DFT on structures which were not used to fit, among
them structures from OQMD which are in none of the two other sources. The energy
above the hull is the one of hull.hdf5 (build_lematerial.py --hull), for the
DFT energy and for the estimated one. Run inside the project environment:

    uv run python scripts/build_energy_correction.py datasets --model orb-v3-conservative-inf-mpa

Predictions are saved and reused by the next runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lematerial_to_cif import REPO, REVISION, parquet_files
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.hull import energy_above_reference, load_hull
from torchms.estimator import EnergyEstimator
from torchms.estimator.calibration import (
    apply_element_correction,
    composition_matrix,
    error_report,
    fit_element_correction,
    save_correction,
)

SUBSET = "compatible_pbe"
# DFT energies above this, in eV/atom, are failed calculations
MAX_ENERGY = 20.0
# an element is inconsistent between sources when the corrections fitted on
# each of them differ by more than this, in eV/atom
INCONSISTENT = 0.05
# an element needs this number of structures to compare the sources
MIN_STRUCTURES = 30


def load_sources(hdf5_file: Path) -> pd.DataFrame:
    """Identifier, source, fingerprint and DFT energy of each structure of the dataset."""
    files = parquet_files(REPO, f"{SUBSET}/*.parquet", REVISION)
    columns = [
        "immutable_id",
        "entalpic_fingerprint",
        "chemical_formula_reduced",
        "energy",
    ]
    table = pa.concat_tables(
        [pq.read_table(f, columns=columns) for f in files]
    ).to_pandas()

    index = pd.read_parquet(hdf5_file.with_name("all_index.parquet"))["immutable_id"]
    if not table["immutable_id"].equals(index):
        raise RuntimeError("the dataset is not in the order of LeMat-Bulk")

    table["source"] = table["immutable_id"].fillna("").str.extract(r"^([a-z]+)")[0]
    return table


def held_out(fingerprints: pd.Series, share: int = 5) -> np.ndarray:
    """Structures held out of the fit: one fingerprint out of share, by its hash."""

    def bucket(fingerprint: Any) -> int:
        if fingerprint is None:
            return 1
        return (
            int(hashlib.sha256(str(fingerprint).encode()).hexdigest()[:8], 16) % share
        )

    unique = pd.Series(fingerprints.unique())
    buckets = pd.Series(unique.map(bucket).to_numpy(), index=unique)
    return fingerprints.map(buckets).to_numpy() == 0


def make_splits(
    sources: pd.DataFrame, num_fit: int, num_test: int, seed: int
) -> dict[str, np.ndarray]:
    """Indices of the structures used to fit and to test, in ascending order."""
    rng = np.random.default_rng(seed)
    reference = sources["source"].isin(["mp", "agm"]).to_numpy()
    valid = (sources["energy"].notna()).to_numpy()

    fingerprints = sources["entalpic_fingerprint"]
    test = held_out(fingerprints)

    known = set(fingerprints[reference].dropna())
    oqmd = (sources["source"] == "oqmd").to_numpy() & fingerprints.notna().to_numpy()
    new_structure = oqmd & ~fingerprints.isin(known).to_numpy()
    formulas = set(sources["chemical_formula_reduced"][reference])
    new_composition = (
        new_structure & ~sources["chemical_formula_reduced"].isin(formulas).to_numpy()
    )

    def sample(mask: np.ndarray, size: int) -> np.ndarray:
        candidates = np.flatnonzero(mask & valid)
        return np.sort(
            rng.choice(candidates, min(size, len(candidates)), replace=False)
        )

    return {
        "fit": sample(reference & ~test, num_fit),
        "held_out": sample(reference & test, num_test),
        "oqmd_new_structure": sample(new_structure, num_test),
        "oqmd_new_composition": sample(new_composition, num_test),
    }


def predict(
    estimator: EnergyEstimator, dataset: HDF5Dataset, indices: np.ndarray, cache: Path
) -> np.ndarray:
    """Energy predicted by the model, reusing the predictions of previous runs."""
    known: dict[int, float] = {}
    if cache.exists():
        saved = np.load(cache)
        known = dict(zip(saved["indices"].tolist(), saved["energy"].tolist()))

    missing = np.array(sorted(set(indices.tolist()) - set(known)), dtype=np.int64)
    if len(missing) > 0:
        energy = estimator.predict(dataset, missing, progress=True)
        known.update(zip(missing.tolist(), energy.tolist()))
        order = np.array(sorted(known), dtype=np.int64)
        np.savez(
            cache, indices=order, energy=np.array([known[i] for i in order.tolist()])
        )

    return np.array([known[i] for i in indices.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument("--model", default="orb-v3-conservative-inf-mpa")
    parser.add_argument("--num-fit", type=int, default=300_000)
    parser.add_argument("--num-test", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--loss",
        default="absolute",
        choices=["squared", "absolute"],
        help="error minimized by the correction",
    )
    args = parser.parse_args()

    root = Path(args.destination) / "lematerial"
    hdf5_file = root / "all.hdf5"
    output = root / "energy"
    output.mkdir(parents=True, exist_ok=True)
    short = args.model.rsplit("-", maxsplit=1)[-1]

    sources = load_sources(hdf5_file)
    splits = make_splits(sources, args.num_fit, args.num_test, args.seed)
    indices = np.unique(np.concatenate(list(splits.values())))

    dataset = HDF5Dataset(hdf5_file)
    estimator = EnergyEstimator.from_pretrained(args.model, device=args.device)
    predicted_all = predict(
        estimator, dataset, indices, output / f"predictions_{short}.npz"
    )
    predicted = dict(zip(indices.tolist(), predicted_all.tolist()))

    with h5py.File(hdf5_file, "r") as data:
        num_atoms_all = data["num_atoms"][:]
        ptr = data["ptr"][:]
        z_all = data["z"][:]
        total_energy_all = data["total_energy"][:]

    # DFT energy above the hull of hull.hdf5 (Materials Project and Alexandria),
    # not the one stored in the dataset, which mixes the energies of OQMD
    above_hull_all = energy_above_reference(
        load_hull(root / "hull.hdf5"), z_all, num_atoms_all, total_energy_all, 8
    )

    def arrays(
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_atoms = num_atoms_all[rows]
        atoms = np.concatenate(
            [np.arange(a, a + n) for a, n in zip(ptr[rows], num_atoms)]
        )
        target = sources["energy"].to_numpy()[rows]
        energy = np.array([predicted[i] for i in rows.tolist()])
        return z_all[atoms], num_atoms, energy, target, above_hull_all[rows]

    # correction fitted on the Materials Project and Alexandria
    z, num_atoms, energy, target, _ = arrays(splits["fit"])
    physical = np.abs(target / num_atoms) < MAX_ENERGY
    correction, counts = fit_element_correction(
        z, num_atoms, energy, target, physical, loss=args.loss
    )
    mean_shift, _ = fit_element_correction(z, num_atoms, energy, target, physical)

    # elements whose energies in OQMD are not on the scale of the two others:
    # compared with the mean shift of each source (squared error), which also
    # sees the elements with a few large differences
    z, num_atoms, energy, target, _ = arrays(splits["oqmd_new_structure"])
    physical = np.abs(target / num_atoms) < MAX_ENERGY
    oqmd_shift, oqmd_counts = fit_element_correction(
        z, num_atoms, energy, target, physical
    )
    compared = (counts >= MIN_STRUCTURES) & (oqmd_counts >= MIN_STRUCTURES)
    difference = np.where(compared, oqmd_shift - mean_shift, 0.0)
    inconsistent = np.flatnonzero(np.abs(difference) > INCONSISTENT)
    unknown = np.flatnonzero((oqmd_counts > 0) & ~compared)

    report: dict[str, Any] = {
        "model": args.model,
        "loss": args.loss,
        "num_fit": int(len(splits["fit"])),
        "elements_fitted": int((counts > 0).sum()),
        "correction_min": float(correction[counts > 0].min()),
        "correction_max": float(correction[counts > 0].max()),
        "oqmd_inconsistent_elements": {
            int(el): float(difference[el]) for el in inconsistent
        },
        "oqmd_elements_not_compared": [int(el) for el in unknown],
        "sets": {},
    }

    excluded = np.zeros(128, dtype=bool)
    excluded[inconsistent] = True
    excluded[unknown] = True

    sets = {
        "A_held_out": (splits["held_out"], False),
        "B_oqmd_new_structure_consistent": (splits["oqmd_new_structure"], True),
        "C_oqmd_new_composition_consistent": (splits["oqmd_new_composition"], True),
        "D_oqmd_new_structure_all": (splits["oqmd_new_structure"], False),
    }
    for name, (rows, consistent_only) in sets.items():
        z, num_atoms, energy, target, above_hull = arrays(rows)
        keep = np.abs(target / num_atoms) < MAX_ENERGY
        if consistent_only:
            keep &= (
                np.asarray(
                    composition_matrix(z, num_atoms)[:, excluded].sum(axis=1)
                ).ravel()
                == 0
            )

        corrected = apply_element_correction(correction, z, num_atoms, energy)
        nan = np.where(keep, 0.0, np.nan)
        report["sets"][name] = {
            "raw": error_report(energy + nan, target, num_atoms, above_hull),
            "corrected": error_report(corrected + nan, target, num_atoms, above_hull),
        }

    held = report["sets"]["A_held_out"]["corrected"]
    save_correction(
        root / f"energy_correction_orb_{short}.hdf5",
        correction,
        counts,
        model=args.model,
        loss=args.loss,
        target="LeMat-Bulk compatible_pbe, Materials Project and Alexandria structures",
        num_fit=int(len(splits["fit"])),
        held_out_mae=held["mae"],
        held_out_median=held["median"],
    )

    pd.DataFrame(
        {
            "immutable_id": sources["immutable_id"].to_numpy()[
                np.concatenate(list(splits.values()))
            ],
            "row": np.concatenate(list(splits.values())),
            "set": np.concatenate(
                [[name] * len(rows) for name, rows in splits.items()]
            ),
        }
    ).to_parquet(output / "splits.parquet", index=False)

    with open(output / f"report_{short}.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report["oqmd_inconsistent_elements"]))
    for name, values in report["sets"].items():
        for kind in ("raw", "corrected"):
            r = values[kind]
            print(
                f"{name:36s} {kind:9s} n={r['count']:6d} MAE {r['mae']:.4f} median {r['median']:.4f} "
                f"p90 {r['p90']:.4f} p99 {r['p99']:.3f} | stable found {r['stable_found']:.1%} "
                f"false {r['stable_false']:.1%} agree@0.1 {r['metastable_agreement']:.1%}"
            )


if __name__ == "__main__":
    main()
