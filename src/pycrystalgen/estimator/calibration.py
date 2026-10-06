"""Correction of predicted total energies to the energies of a dataset.

Energies predicted by a pretrained model and energies of a dataset computed
with other settings mostly differ by an offset per atom of each element. The
correction is this offset, fitted by least squares:

    corrected = predicted + sum over atoms of correction[element]
"""

from __future__ import annotations

import os
from typing import Any, Literal

import h5py
import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Float, Int
from scipy.sparse.linalg import lsqr

from pycrystalgen.data.hull import compositions

NUM_ELEMENTS = 128

# fit of the mean absolute error: number of reweighted least squares and
# smallest error, in eV/atom, used as a weight
_ABSOLUTE_ITERATIONS = 30
_ABSOLUTE_EPSILON = 1e-3


def composition_matrix(
    z: Int[np.ndarray, "n_atoms"], num_atoms: Int[np.ndarray, "n_struct"]
) -> sp.csr_matrix:
    """Number of atoms of each element in each structure.

    Returns:
        Sparse matrix of shape (n_struct, 128), columns are atomic numbers.

    Example:
        >>> matrix = composition_matrix(np.array([1, 8, 1, 26]), np.array([3, 1]))
        >>> matrix[:, [1, 8, 26]].toarray().tolist()
        [[2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    elements, counts, ptr = compositions(z, num_atoms)
    structure = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))

    return sp.csr_matrix(
        (counts.astype(np.float64), (structure, elements)),
        shape=(len(ptr) - 1, NUM_ELEMENTS),
    )


def fit_element_correction(
    z: Int[np.ndarray, "n_atoms"],
    num_atoms: Int[np.ndarray, "n_struct"],
    predicted: Float[np.ndarray, "n_struct"],
    target: Float[np.ndarray, "n_struct"],
    mask: Bool[np.ndarray, "n_struct"] | None = None,
    loss: Literal["squared", "absolute"] = "squared",
) -> tuple[Float[np.ndarray, "128"], Int[np.ndarray, "128"]]:
    """Fit an energy per atom of each element added to predicted energies.

    The correction minimizes the error of the energy per atom of the
    structures.

    Args:
        z: Atomic numbers of the atoms, shape (n_atoms,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        predicted: Predicted total energy of each structure in eV.
        target: Total energy of each structure in eV.
        mask: Structures used to fit, all by default. Structures with a
            missing (NaN) energy are never used.
        loss: "squared" minimizes the mean squared error, "absolute" the mean
            absolute error, less sensitive to structures with a large error.

    Raises:
        ValueError: If the loss is unknown.

    Returns:
        Correction in eV/atom indexed by atomic number, shape (128,), zero for
        the elements of no fitted structure, and the number of fitted
        structures containing each element, shape (128,).

    Example:
        Predictions miss 1 eV per hydrogen and -0.5 eV per oxygen:

        >>> z = np.array([1, 1, 8, 1, 8, 8, 1, 1])
        >>> num_atoms = np.array([3, 3, 2])
        >>> target = np.array([-10.0, -12.0, -4.0])
        >>> predicted = target - np.array([2 * 1.0 - 0.5, 1.0 - 2 * 0.5, 2 * 1.0])
        >>> correction, counts = fit_element_correction(z, num_atoms, predicted, target)
        >>> correction[[1, 8]].round(6).tolist(), counts[[1, 8]].tolist()
        ([1.0, -0.5], [3, 2])
    """
    if loss not in ("squared", "absolute"):
        raise ValueError(f'unknown loss "{loss}", must be "squared" or "absolute"')

    num_atoms = np.asarray(num_atoms, dtype=np.int64)
    residual = (
        np.asarray(target, np.float64) - np.asarray(predicted, np.float64)
    ) / num_atoms

    used = ~np.isnan(residual)
    if mask is not None:
        used &= np.asarray(mask, dtype=bool)

    matrix = composition_matrix(z, num_atoms)[used]
    fractions = sp.diags(1.0 / num_atoms[used]) @ matrix

    counts = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.int64)
    seen = counts > 0

    correction = np.zeros(NUM_ELEMENTS)
    if not seen.any():
        return correction, counts

    design, values = fractions.tocsc()[:, seen], residual[used]

    def solve(weights: np.ndarray) -> np.ndarray:
        scale = np.sqrt(weights)
        return lsqr(sp.diags(scale) @ design, scale * values, atol=1e-12, btol=1e-12)[0]

    solution = solve(np.ones(len(values)))

    if loss == "absolute":
        # iteratively reweighted least squares: weights 1 / |error|
        for _ in range(_ABSOLUTE_ITERATIONS):
            error = np.abs(values - design @ solution)
            solution = solve(1.0 / np.maximum(error, _ABSOLUTE_EPSILON))

    correction[seen] = solution

    return correction, counts


def apply_element_correction(
    correction: Float[np.ndarray, "128"],
    z: Int[np.ndarray, "n_atoms"],
    num_atoms: Int[np.ndarray, "n_struct"],
    predicted: Float[np.ndarray, "n_struct"],
) -> Float[np.ndarray, "n_struct"]:
    """Add the correction of each atom to predicted total energies.

    Example:
        >>> correction = np.zeros(128)
        >>> correction[[1, 8]] = [1.0, -0.5]
        >>> z, num_atoms = np.array([1, 1, 8, 8]), np.array([3, 1])
        >>> apply_element_correction(correction, z, num_atoms, np.zeros(2)).tolist()
        [1.5, -0.5]
    """
    shift = composition_matrix(z, num_atoms) @ np.asarray(correction, dtype=np.float64)
    return np.asarray(predicted, dtype=np.float64) + shift


def save_correction(
    correction_file: str | os.PathLike,
    correction: Float[np.ndarray, "128"],
    counts: Int[np.ndarray, "128"],
    **attributes: Any,
) -> None:
    """Save a correction.

    Args:
        correction_file: Output file (hdf5 format) with the datasets
            "correction" (128,) in eV/atom and "counts" (128,), both indexed
            by atomic number.
        correction: Correction in eV/atom indexed by atomic number.
        counts: Number of fitted structures containing each element.
        **attributes: Description stored as attributes of the file (name of
            the model, statistics of the fit...): numbers or strings.
    """
    with h5py.File(correction_file, "w") as data:
        data.create_dataset("correction", data=np.asarray(correction, dtype=np.float64))
        data.create_dataset("counts", data=np.asarray(counts, dtype=np.int64))
        for name, value in attributes.items():
            data.attrs[name] = value


def load_correction(
    correction_file: str | os.PathLike,
) -> tuple[Float[np.ndarray, "128"], Int[np.ndarray, "128"], dict[str, Any]]:
    """Load a correction saved by save_correction.

    Returns:
        Correction in eV/atom, number of fitted structures containing each
        element, both indexed by atomic number, and the attributes of the file.

    Raises:
        KeyError: If the file is not a correction.
    """
    with h5py.File(correction_file, "r") as data:
        correction, counts = data["correction"], data["counts"]

        if not isinstance(correction, h5py.Dataset) or not isinstance(
            counts, h5py.Dataset
        ):
            raise KeyError(f"{correction_file} is not a correction file")

        attributes = {name: _python(value) for name, value in data.attrs.items()}
        return correction[:].astype(np.float64), counts[:].astype(np.int64), attributes


def _python(value: Any) -> Any:
    """Convert an attribute read from a file to a Python object."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def error_report(
    predicted: Float[np.ndarray, "n_struct"],
    target: Float[np.ndarray, "n_struct"],
    num_atoms: Int[np.ndarray, "n_struct"],
    above_hull: Float[np.ndarray, "n_struct"] | None = None,
    stable: float = 1e-6,
    metastable: float = 0.1,
) -> dict[str, float]:
    """Compare predicted total energies with reference ones.

    Args:
        predicted: Predicted total energy of each structure in eV.
        target: Reference total energy of each structure in eV.
        num_atoms: Number of atoms of each structure.
        above_hull: Reference energy above the hull of each structure in
            eV/atom. If given, the predicted one is this value plus the error
            of the energy per atom, and stability is compared.
        stable: A structure is stable below this energy above the hull.
        metastable: A structure is meta-stable below this one.

    Returns:
        Statistics of the absolute error of the energy per atom in eV/atom
        ("mae", "median", "p90", "p99", "max"), the mean error ("bias") and
        the number of structures ("count"). With above_hull: share of the
        stable structures predicted stable ("stable_found"), share of the
        structures predicted stable which are not ("stable_false"), and share
        of the structures on the same side of the meta-stable threshold
        ("metastable_agreement").
    """
    error = (
        np.asarray(predicted, np.float64) - np.asarray(target, np.float64)
    ) / num_atoms
    known = ~np.isnan(error)
    absolute = np.abs(error[known])

    report = {
        "count": int(known.sum()),
        "mae": float(absolute.mean()),
        "median": float(np.median(absolute)),
        "p90": float(np.quantile(absolute, 0.9)),
        "p99": float(np.quantile(absolute, 0.99)),
        "max": float(absolute.max()),
        "bias": float(error[known].mean()),
    }

    if above_hull is not None:
        reference = np.asarray(above_hull, dtype=np.float64)
        known &= ~np.isnan(reference)
        reference, estimated = reference[known], reference[known] + error[known]

        is_stable, found_stable = reference <= stable, estimated <= stable
        report["stable_count"] = int(is_stable.sum())
        report["stable_found"] = float(
            (is_stable & found_stable).sum() / max(is_stable.sum(), 1)
        )
        report["stable_false"] = float(
            (found_stable & ~is_stable).sum() / max(found_stable.sum(), 1)
        )
        report["metastable_agreement"] = float(
            ((reference <= metastable) == (estimated <= metastable)).mean()
        )

    return report
