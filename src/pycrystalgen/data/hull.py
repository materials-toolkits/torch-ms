"""Energy above the convex hull of crystal structures.

The hull of a chemical system is the lower convex envelope of the energy per
atom of all the structures made of its elements only. It is computed one
chemical system at a time, from the elements up to the systems with the most
elements: the hull of a system is defined by its elements, the stable
structures of its sub-systems and its own structures.

Energies must be comparable (same level of theory) across the structures.

A saved hull (see save_hull) stores the composition and the energy of the
structures on the hull:

* "elements" (n_pairs,): atomic number of the elements of each structure.
* "counts" (n_pairs,): number of atoms of this element in the structure.
* "ptr" (n_struct+1,): index of the first element of each structure.
* "energy_per_atom" (n_struct,): energy per atom in eV.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import tqdm
from jaxtyping import Bool, Float, Int
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

# a structure is on the hull when it is closer than this, in eV/atom
STABLE_TOLERANCE = 1e-8

# shared with the workers (forked), set by _levels
_STATE: dict[str, Any] = {}


@dataclass
class Hull:
    """Convex hull of a set of structures.

    Attributes:
        reference: Energy per atom of the stable structure of each element,
            indexed by atomic number, NaN for the elements without structure.
        stable: For each chemical system (sorted atomic numbers), fractions
            (n_stable, n_elements) and formation energies (n_stable,) of its
            structures on the hull. Elements are not listed.
    """

    reference: Float[np.ndarray, "128"] = field(
        default_factory=lambda: np.full(128, np.nan)
    )
    stable: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


def compositions(
    z: Int[np.ndarray, "n_atoms"], num_atoms: Int[np.ndarray, "n_struct"]
) -> tuple[
    Int[np.ndarray, "n_pairs"],
    Int[np.ndarray, "n_pairs"],
    Int[np.ndarray, "n_struct+1"],
]:
    """Get the composition of structures stored as flat concatenated atoms.

    Args:
        z: Atomic numbers of the atoms, shape (n_atoms,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).

    Returns:
        Elements sorted by structure then atomic number, shape (n_pairs,),
        their number of atoms in the structure, shape (n_pairs,), and the
        index of the first element of each structure, shape (n_struct+1,).

    Examples:
        NaCl3 (unsorted atoms) and Fe2:

        >>> z = np.array([17, 11, 17, 17, 26, 26])
        >>> elements, counts, ptr = compositions(z, np.array([4, 2]))
        >>> elements.tolist(), counts.tolist(), ptr.tolist()
        ([11, 17, 26], [1, 3, 2], [0, 2, 3])
    """
    num_atoms = np.asarray(num_atoms, dtype=np.int64)
    batch = np.repeat(np.arange(len(num_atoms), dtype=np.int64), num_atoms)

    pairs, counts = np.unique(batch * 128 + np.asarray(z), return_counts=True)

    ptr = np.zeros(len(num_atoms) + 1, dtype=np.int64)
    np.cumsum(np.bincount(pairs // 128, minlength=len(num_atoms)), out=ptr[1:])

    return pairs % 128, counts, ptr


def fractions(
    counts: Int[np.ndarray, "n_pairs"], ptr: Int[np.ndarray, "n_struct+1"]
) -> Float[np.ndarray, "n_pairs"]:
    """Fraction of the atoms of its structure of each element.

    Examples:
        >>> fractions(np.array([1, 3, 2]), np.array([0, 2, 3])).tolist()
        [0.25, 0.75, 1.0]
    """
    batch = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    total = np.bincount(batch, weights=counts, minlength=len(ptr) - 1)
    return counts / total[batch]


def hull_energy_lp(
    points: Float[np.ndarray, "n_points dim"],
    energies: Float[np.ndarray, "n_points"],
    queries: Float[np.ndarray, "n_queries dim"],
) -> Float[np.ndarray, "n_queries"]:
    """Energy of the lower hull at the given compositions, by linear programs.

    Slow reference used when qhull fails on a degenerate system. Arguments
    are the ones of hull_energy, without constraint on the energies.

    Raises:
        RuntimeError: If a query is not a mix of the points.
    """
    constraints = np.vstack([points.T, np.ones(len(points))])
    result = np.zeros(len(queries))

    for i, query in enumerate(queries):
        solution = linprog(
            energies,
            A_eq=constraints,
            b_eq=np.append(query, 1.0),
            bounds=(0, None),
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(f"hull energy not found: {solution.message}")
        result[i] = solution.fun

    return result


def hull_energy(
    points: Float[np.ndarray, "n_points dim"],
    energies: Float[np.ndarray, "n_points"],
    queries: Float[np.ndarray, "n_queries dim"],
) -> Float[np.ndarray, "n_queries"]:
    """Energy of the lower convex hull at the given compositions.

    Args:
        points: Fractions of the first n-1 elements of the points defining
            the hull, shape (n_points, n-1). Must contain the n elements.
        energies: Formation energy of the points, shape (n_points,). Must be
            negative or zero, and zero for the elements.
        queries: Fractions of the first n-1 elements where the hull is
            evaluated, shape (n_queries, n-1).

    Returns:
        Hull energy of the queries, shape (n_queries,), negative or zero.

    Examples:
        Elements A and B and a compound AB with a formation energy of -1:

        >>> points = np.array([[1.0], [0.0], [0.5]])
        >>> energies = np.array([0.0, 0.0, -1.0])
        >>> queries = np.array([[0.5], [0.75], [1.0]])
        >>> hull_energy(points, energies, queries).round(6).tolist()
        [-1.0, -0.5, 0.0]
    """
    if len(queries) == 0 or not (energies < 0).any():
        return np.zeros(len(queries))

    dim = points.shape[1]

    try:
        hull = ConvexHull(np.column_stack([points, energies]))
    except QhullError:
        return hull_energy_lp(points, energies, queries)

    # lower facets: outward normal pointing to negative energies
    lower = hull.equations[hull.equations[:, dim] < -1e-12]
    if len(lower) == 0:
        return hull_energy_lp(points, energies, queries)

    planes = -(queries @ lower[:, :dim].T + lower[:, dim + 1]) / lower[:, dim]

    return np.minimum(planes.max(axis=1), 0.0)


def _system_hull(
    job: tuple[int, int, int],
) -> tuple[Float[np.ndarray, "n"], Bool[np.ndarray, "n"]]:
    """Energy above the hull of the structures of one chemical system.

    Returns:
        Energy above the hull of the structures of the system, and the mask of
        the ones on the hull.
    """
    level, start, stop = job
    rows = _STATE["order"][start:stop]
    elements = _STATE["elements"][rows[0]]
    system_fractions = _STATE["fractions"][rows]
    formation = _STATE["formation"][rows]
    stable = _STATE["hull"].stable
    fixed = _STATE["fixed"]

    # points defining the hull: the elements, the stable structures of the
    # sub-systems and, unless the hull is fixed, the structures of the system
    # with a negative formation energy
    points = [np.eye(level)]
    energies = [np.zeros(level)]

    for size in range(2, level + 1 if fixed else level):
        for subset in itertools.combinations(range(level), size):
            found = stable.get(tuple(elements[list(subset)].tolist()))
            if found is None:
                continue

            embedded = np.zeros((len(found[1]), level))
            embedded[:, subset] = found[0]
            points.append(embedded)
            energies.append(found[1])

    negative = formation < 0
    if not fixed:
        points.append(system_fractions[negative])
        energies.append(formation[negative])

    above = formation - hull_energy(
        np.concatenate(points)[:, : level - 1],
        np.concatenate(energies),
        system_fractions[:, : level - 1],
    )

    if fixed:
        above[np.abs(above) <= STABLE_TOLERANCE] = 0.0
    else:
        above[above <= STABLE_TOLERANCE] = 0.0

    return above, negative & (above == 0.0)


def _levels(
    elements: Int[np.ndarray, "n_pairs"],
    counts: Int[np.ndarray, "n_pairs"],
    ptr: Int[np.ndarray, "n_struct+1"],
    energy: Float[np.ndarray, "n_struct"],
    hull: Hull,
    fixed: bool,
    num_workers: int,
    progress: bool,
) -> Float[np.ndarray, "n_struct"]:
    """Energy above the hull of structures, one chemical system at a time.

    Args:
        elements: Elements of each structure, shape (n_pairs,).
        counts: Number of atoms of each element, shape (n_pairs,).
        ptr: Index of the first element of each structure, shape (n_struct+1,).
        energy: Energy per atom of each structure, shape (n_struct,).
        hull: Hull, completed with the structures unless it is fixed.
        fixed: If True the hull is not modified: structures are only compared
            to it and can be below.
        num_workers: Number of processes.
        progress: Show a progress bar.

    Returns:
        Energy above the hull in eV/atom, shape (n_struct,).
    """
    fraction = fractions(counts, ptr)
    num_elements = np.diff(ptr)

    result = np.full(len(energy), np.nan)
    known = ~np.isnan(energy)

    unary = np.flatnonzero(known & (num_elements == 1))
    if len(unary) > 0:
        unary_elements = elements[ptr[unary]]

        if not fixed:
            # reference of each element: its structure with the lowest energy
            np.fmin.at(hull.reference, unary_elements, energy[unary])

        above = energy[unary] - hull.reference[unary_elements]
        snap = np.abs(above) if fixed else above
        above[snap <= STABLE_TOLERANCE] = 0.0
        result[unary] = above

    levels = sorted(set(num_elements[known & (num_elements > 1)].tolist()))
    progress_bar = tqdm.tqdm(
        total=int((num_elements > 1).sum()), unit="struct", disable=not progress
    )

    for level in levels:
        index = np.flatnonzero(known & (num_elements == level))
        columns = ptr[index][:, None] + np.arange(level)
        level_elements = elements[columns]
        level_fractions = fraction[columns]
        formation = energy[index] - (
            level_fractions * hull.reference[level_elements]
        ).sum(axis=1)

        # structures with an element without reference have no hull
        defined = ~np.isnan(formation)
        index, level_elements = index[defined], level_elements[defined]
        level_fractions, formation = level_fractions[defined], formation[defined]

        systems, inverse = np.unique(level_elements, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(systems) + 1))

        _STATE.clear()
        _STATE.update(
            elements=level_elements,
            fractions=level_fractions,
            formation=formation,
            order=order,
            hull=hull,
            fixed=fixed,
        )

        jobs = [(level, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]

        if num_workers > 1 and len(jobs) > 1:
            # forked after the state is set: workers see the lower levels
            with mp.get_context("fork").Pool(num_workers) as pool:
                outputs = pool.map(_system_hull, jobs, chunksize=256)
        else:
            outputs = list(map(_system_hull, jobs))

        for (_, start, stop), (above, on_hull) in zip(jobs, outputs):
            rows = order[start:stop]
            result[index[rows]] = above

            if not fixed and on_hull.any():
                system = tuple(level_elements[rows[0]].tolist())
                hull.stable[system] = (
                    level_fractions[rows[on_hull]],
                    formation[rows[on_hull]],
                )

        progress_bar.update(len(index))

    progress_bar.close()
    _STATE.clear()

    return result


def energy_above_hull(
    z: Int[np.ndarray, "n_atoms"],
    num_atoms: Int[np.ndarray, "n_struct"],
    total_energy: Float[np.ndarray, "n_struct"],
    num_workers: int = 1,
    progress: bool = False,
) -> Float[np.ndarray, "n_struct"]:
    """Compute the energy above the convex hull of each structure.

    The hull is built from the structures themselves. Structures on the hull
    have an energy above the hull of exactly zero.

    Args:
        z: Atomic numbers of the atoms, shape (n_atoms,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        total_energy: Total energy of each structure in eV, shape (n_struct,).
        num_workers: Number of processes.
        progress: Show a progress bar.

    Returns:
        Energy above the hull in eV/atom, shape (n_struct,). NaN for the
        structures without energy or with an element that has no structure
        made of this element only.

    Examples:
        Elements A (-1 eV/atom) and B (-2 eV/atom), a stable compound AB
        (-2.5 eV/atom) and A3B (-1.5 eV/atom), above the line from A to AB:

        >>> z = np.array([1, 2, 1, 2, 1, 1, 1, 2])
        >>> num_atoms = np.array([1, 1, 2, 4])
        >>> total_energy = np.array([-1.0, -2.0, -5.0, -6.0])
        >>> energy_above_hull(z, num_atoms, total_energy).round(6).tolist()
        [0.0, 0.0, 0.0, 0.25]
    """
    energy = np.asarray(total_energy, dtype=np.float64) / np.asarray(num_atoms)
    elements, counts, ptr = compositions(z, num_atoms)

    return _levels(elements, counts, ptr, energy, Hull(), False, num_workers, progress)


def save_hull(
    hull_file: str | os.PathLike,
    z: Int[np.ndarray, "n_atoms"],
    num_atoms: Int[np.ndarray, "n_struct"],
    total_energy: Float[np.ndarray, "n_struct"],
    num_workers: int = 1,
    progress: bool = False,
) -> Float[np.ndarray, "n_struct"]:
    """Save the structures on the convex hull of a dataset.

    Args:
        hull_file: Output file (hdf5 format), see the format in the module.
        z: Atomic numbers of the atoms, shape (n_atoms,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        total_energy: Total energy of each structure in eV, shape (n_struct,).
        num_workers: Number of processes.
        progress: Show a progress bar.

    Returns:
        Energy above the hull of every structure in eV/atom, shape (n_struct,).
        The saved structures are the ones at zero, in the same order.
    """
    num_atoms = np.asarray(num_atoms, dtype=np.int64)
    energy = np.asarray(total_energy, dtype=np.float64) / num_atoms
    elements, counts, ptr = compositions(z, num_atoms)

    above = _levels(elements, counts, ptr, energy, Hull(), False, num_workers, progress)

    on_hull = above == 0.0
    pairs = np.repeat(on_hull, np.diff(ptr))
    hull_ptr = np.zeros(int(on_hull.sum()) + 1, dtype=np.int64)
    np.cumsum(np.diff(ptr)[on_hull], out=hull_ptr[1:])

    with h5py.File(hull_file, "w") as data:
        data.create_dataset("elements", data=elements[pairs].astype(np.int64))
        data.create_dataset("counts", data=counts[pairs].astype(np.int64))
        data.create_dataset("ptr", data=hull_ptr)
        data.create_dataset("energy_per_atom", data=energy[on_hull])

    return above


def _dataset(data: h5py.File, name: str) -> h5py.Dataset:
    """Get a dataset of a file by name."""
    dataset = data[name]

    if not isinstance(dataset, h5py.Dataset):
        raise KeyError(f'"{name}" is not a dataset of {data.filename}')

    return dataset


def load_hull(
    hull_file: str | os.PathLike, num_workers: int = 1, progress: bool = False
) -> Hull:
    """Load a hull saved by save_hull.

    Args:
        hull_file: Saved hull (hdf5 format), see the format in the module.
        num_workers: Number of processes.
        progress: Show a progress bar.
    """
    with h5py.File(hull_file, "r") as data:
        elements = _dataset(data, "elements")[:]
        counts = _dataset(data, "counts")[:]
        ptr = _dataset(data, "ptr")[:]
        energy = _dataset(data, "energy_per_atom")[:]

    hull = Hull()
    _levels(elements, counts, ptr, energy, hull, False, num_workers, progress)

    return hull


def energy_above_reference(
    hull: Hull,
    z: Int[np.ndarray, "n_atoms"],
    num_atoms: Int[np.ndarray, "n_struct"],
    total_energy: Float[np.ndarray, "n_struct"],
    num_workers: int = 1,
    progress: bool = False,
) -> Float[np.ndarray, "n_struct"]:
    """Compute the energy of structures above a hull they are not part of.

    Args:
        hull: Reference hull (see load_hull). It is not modified.
        z: Atomic numbers of the atoms, shape (n_atoms,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        total_energy: Total energy of each structure in eV, shape (n_struct,).
            Must be computed like the energies of the hull.
        num_workers: Number of processes.
        progress: Show a progress bar.

    Returns:
        Energy above the hull in eV/atom, shape (n_struct,). It is zero on the
        hull and negative for a structure more stable than the hull. NaN for
        the structures without energy or with an element absent from the hull.
    """
    energy = np.asarray(total_energy, dtype=np.float64) / np.asarray(num_atoms)
    elements, counts, ptr = compositions(z, num_atoms)

    return _levels(elements, counts, ptr, energy, hull, True, num_workers, progress)


def fill_missing(
    hdf5_file: str | os.PathLike, num_workers: int = 1, progress: bool = False
) -> tuple[Float[np.ndarray, "n_struct"], dict[str, float]]:
    """Fill the missing energies above the hull of a dataset with computed ones.

    Only the structures whose "energy_above_hull" is NaN are modified. The
    values already stored are compared with the computed ones.

    Args:
        hdf5_file: Dataset (hdf5 format, see HDF5Dataset), modified in place.
        num_workers: Number of processes.
        progress: Show a progress bar.

    Returns:
        The computed energy above the hull of every structure, and statistics:
        number of filled structures and differences with the stored values.
    """
    with h5py.File(hdf5_file, "r+") as data:
        computed = energy_above_hull(
            _dataset(data, "z")[:],
            _dataset(data, "num_atoms")[:],
            _dataset(data, "total_energy")[:],
            num_workers,
            progress,
        )
        target = _dataset(data, "energy_above_hull")
        stored = target[:].astype(np.float64)

        missing = np.isnan(stored) & ~np.isnan(computed)
        both = ~np.isnan(stored) & ~np.isnan(computed)
        difference = np.abs(computed[both] - stored[both])

        if missing.any():
            filled = target[:]
            filled[missing] = computed[missing]
            target[:] = filled

    stats = {
        "filled": int(missing.sum()),
        "still_missing": int((np.isnan(stored) & np.isnan(computed)).sum()),
        "compared": int(both.sum()),
        "max_difference": float(difference.max()) if both.any() else 0.0,
        "differences_above_1e-4": int((difference > 1e-4).sum()),
    }

    return computed, stats
