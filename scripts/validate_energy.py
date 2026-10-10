"""Validate the estimated energies of LeMat-Bulk structures, end to end.

The estimator (pretrained model and correction) predicts the total energy of
the test structures chosen by build_energy_correction.py, their energy above
the hull is computed from hull.hdf5, and both are compared with DFT. Run
inside the project environment:

    uv run python scripts/validate_energy.py datasets
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from torchms.data.hdf5 import HDF5Dataset
from torchms.data.hull import energy_above_reference, load_hull
from torchms.models.estimator import EnergyEstimator
from torchms.models.estimator.calibration import composition_matrix, load_correction

MAX_ENERGY = 20.0
STABLE = 1e-6
METASTABLE = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("destination", nargs="?", default="datasets")
    parser.add_argument("--correction", default="energy_correction_orb.hdf5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.destination) / "lematerial"
    correction_file = root / args.correction
    _, _, attributes = load_correction(correction_file)
    short = str(attributes["model"]).rsplit("-", maxsplit=1)[-1]

    with open(root / "energy" / f"report_{short}.json", encoding="utf-8") as fp:
        fit_report = json.load(fp)
    excluded = np.zeros(128, dtype=bool)
    excluded[[int(el) for el in fit_report["oqmd_inconsistent_elements"]]] = True
    excluded[fit_report["oqmd_elements_not_compared"]] = True

    splits = pd.read_parquet(root / "energy" / "splits.parquet")
    dataset = HDF5Dataset(root / "all.hdf5")
    estimator = EnergyEstimator.from_pretrained(
        str(attributes["model"]), correction_file, device=args.device
    )
    hull = load_hull(root / "hull.hdf5", args.num_workers)

    with h5py.File(root / "all.hdf5", "r") as data:
        num_atoms_all, ptr, z_all = data["num_atoms"][:], data["ptr"][:], data["z"][:]
        target_all = data["total_energy"][:].astype(np.float64)

    # DFT energy above the same hull, not the one stored in the dataset which
    # mixes the energies of OQMD
    above_all = energy_above_reference(
        hull, z_all, num_atoms_all, target_all, args.num_workers
    )

    sets = {
        "A. held-out, Materials Project and Alexandria": ("held_out", False),
        "B. OQMD new structures, consistent elements": ("oqmd_new_structure", True),
        "C. OQMD new compositions, consistent elements": ("oqmd_new_composition", True),
        "D. OQMD new structures, all elements": ("oqmd_new_structure", False),
    }
    predictions: dict[str, np.ndarray] = {}
    results = {}

    for name, (split, consistent_only) in sets.items():
        rows = np.sort(splits.loc[splits["set"] == split, "row"].to_numpy())
        if split not in predictions:
            predictions[split] = estimator.predict(dataset, rows, progress=True)
        energy = predictions[split]

        num_atoms = num_atoms_all[rows]
        atoms = np.concatenate(
            [np.arange(a, a + n) for a, n in zip(ptr[rows], num_atoms)]
        )
        z, target = z_all[atoms], target_all[rows]

        # the model has no energy for structures with atoms at the same position
        keep = (np.abs(target / num_atoms) < MAX_ENERGY) & ~np.isnan(energy)
        if consistent_only:
            keep &= (
                np.asarray(
                    composition_matrix(z, num_atoms)[:, excluded].sum(axis=1)
                ).ravel()
                == 0
            )

        estimated = energy_above_reference(hull, z, num_atoms, energy, args.num_workers)
        error = np.abs(energy - target)[keep] / num_atoms[keep]
        reference, estimated = above_all[rows][keep], estimated[keep]

        is_stable, found = reference <= STABLE, estimated <= STABLE
        results[name] = {
            "count": int(keep.sum()),
            "without_estimate": int(np.isnan(energy).sum()),
            "mae": float(error.mean()),
            "median": float(np.median(error)),
            "p90": float(np.quantile(error, 0.9)),
            "p99": float(np.quantile(error, 0.99)),
            "above_hull_mae": float(np.abs(estimated - reference).mean()),
            "stable_count": int(is_stable.sum()),
            "stable_found": float((is_stable & found).sum() / max(is_stable.sum(), 1)),
            "stable_false": float((found & ~is_stable).sum() / max(found.sum(), 1)),
            "metastable_agreement": float(
                ((reference <= METASTABLE) == (estimated <= METASTABLE)).mean()
            ),
            "below_hull": int((estimated < -STABLE).sum()),
        }

    with open(root / "energy" / "validation.json", "w", encoding="utf-8") as fp:
        json.dump({"model": attributes["model"], "sets": results}, fp, indent=2)

    for name, r in results.items():
        print(
            f"{name:48s} n={r['count']:6d} | energy MAE {r['mae']:.4f} median {r['median']:.4f} "
            f"p90 {r['p90']:.4f} p99 {r['p99']:.3f} | above hull MAE {r['above_hull_mae']:.4f} | "
            f"stable found {r['stable_found']:.1%} false {r['stable_false']:.1%} "
            f"agree@0.1 {r['metastable_agreement']:.1%} | below hull {r['below_hull']}"
        )


if __name__ == "__main__":
    main()
