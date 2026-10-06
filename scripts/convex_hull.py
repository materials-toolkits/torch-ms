"""Compute the energy above the convex hull of the structures of a dataset.

Command line of pycrystalgen.data.hull, run inside the project environment:

    uv run python scripts/convex_hull.py dataset.hdf5 above.npy
    uv run python scripts/convex_hull.py dataset.hdf5 above.npy --fill
    uv run python scripts/convex_hull.py dataset.hdf5 above.npy --save-hull hull.hdf5
    uv run python scripts/convex_hull.py generated.hdf5 above.npy --reference hull.hdf5

The hull is built from the total energies of the dataset itself, which must be
comparable (same level of theory). With --fill, the structures of the dataset
without energy above the hull get the computed one, the others are left
unchanged. With --save-hull, the structures on the hull are saved, and
--reference computes the energy of other structures above this saved hull
(negative for a structure below the hull).
"""

import argparse
import os

import h5py
import numpy as np

from pycrystalgen.data.hull import (
    energy_above_hull,
    energy_above_reference,
    fill_missing,
    load_hull,
    save_hull,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", help="dataset (hdf5 format)")
    parser.add_argument("destination", help="energy above the hull (npy format)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--fill",
        action="store_true",
        help="store the computed value of the structures without one in the dataset",
    )
    parser.add_argument(
        "--save-hull", default=None, help="save the structures on the hull (hdf5)"
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="saved hull (hdf5) the structures are compared to, instead of their own",
    )
    args = parser.parse_args(argv)

    workers = args.num_workers

    if args.fill:
        above, stats = fill_missing(args.source, workers, progress=True)
        print(stats)
    else:
        with h5py.File(args.source, "r") as data:
            arrays = (data["z"][:], data["num_atoms"][:], data["total_energy"][:])

        if args.reference is not None:
            hull = load_hull(args.reference, workers, progress=True)
            above = energy_above_reference(hull, *arrays, workers, progress=True)
        elif args.save_hull is not None:
            above = save_hull(args.save_hull, *arrays, workers, progress=True)
        else:
            above = energy_above_hull(*arrays, workers, progress=True)

    np.save(args.destination, above)
    print(
        f"{int((above == 0).sum())} structures on the hull, "
        f"{int((above < 0).sum())} below, "
        f"{int(np.isnan(above).sum())} without energy above the hull"
    )


if __name__ == "__main__":
    main()
