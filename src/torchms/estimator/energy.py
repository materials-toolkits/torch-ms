"""Estimation of the total energy and of the energy above the hull of crystal structures."""

from __future__ import annotations

import os

import numpy as np
import torch
import tqdm
from jaxtyping import Float, Int

from torchms.data.batching import Structures
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.hull import Hull, energy_above_reference, load_hull
from torchms.estimator.calibration import load_correction
from torchms.estimator.orb import OrbEnergyModel

DEFAULT_MODEL = "orb-v3-conservative-inf-mpa"

# Estimator of the energies of LeMat-Bulk, see EnergyEstimator.from_lematerial.
# Files of the Hugging Face dataset repository materials-toolkits/lematerial,
# https://huggingface.co/datasets/materials-toolkits/lematerial
LEMATERIAL_REPOSITORY = "materials-toolkits/lematerial"
# correction fitted on the structures of LeMat-Bulk (compatible_pbe) from the
# Materials Project and Alexandria, for orb-v3-conservative-inf-mpa
LEMATERIAL_CORRECTION = "energy_correction_orb.hdf5"
# convex hull of the structures of LeMat-Bulk (compatible_pbe) from the
# Materials Project and Alexandria; the structures from OQMD are left out
LEMATERIAL_HULL = "hull.hdf5"
LEMATERIAL_HULL_SOURCE = (
    "convex hull of LeMat-Bulk compatible_pbe, structures of the Materials Project "
    "and Alexandria only (OQMD left out)"
)


def to_device(structures: Structures, device: torch.device | str) -> Structures:
    """Copy of structures with its tensors on a device, see Structures.to."""
    return structures.to(device)


def atom_batches(
    num_atoms: Int[np.ndarray, "n"], max_atoms: int
) -> list[tuple[int, int]]:
    """Split consecutive structures in batches of about max_atoms atoms.

    Returns:
        (start, stop) of each batch. A batch is closed by the structure with
        which it reaches max_atoms atoms.

    Examples:
        >>> atom_batches(np.array([4, 4, 4, 10, 1, 1]), max_atoms=8)
        [(0, 2), (2, 4), (4, 6)]
    """
    bounds = [0]
    count = 0

    for i, atoms in enumerate(np.asarray(num_atoms).tolist()):
        count += atoms
        if count >= max_atoms:
            bounds.append(i + 1)
            count = 0

    if bounds[-1] != len(num_atoms):
        bounds.append(len(num_atoms))

    return list(zip(bounds[:-1], bounds[1:]))


class EnergyEstimator:
    """Total energy and energy above the hull of crystal structures, from a
    pretrained model.

    The energy is predicted by the model for the structure as given (it is not
    relaxed), then corrected by an energy per atom of each element fitted on a
    dataset (see torchms.estimator.calibration), to be comparable with
    the energies of this dataset. With a convex hull of the same dataset (see
    torchms.data.hull), it gives the energy above this hull.

    from_lematerial loads the estimator of the energies of LeMat-Bulk: its
    correction and its hull are the ones of the structures of LeMat-Bulk from
    the Materials Project and Alexandria. They do not use the structures from
    OQMD, whose energies are shifted for some elements: for them the estimated
    energy is on the scale of the two other sources, not the energy stored in
    LeMat-Bulk.

    Args:
        model: Pretrained model.
        correction: Correction in eV/atom indexed by atomic number, shape
            (128,), or None for the energy of the model.
        hull: Convex hull the estimated energies are compared to, or None.
        hull_source: Description of where the hull comes from, shown by
            repr(estimator).

    Attributes:
        transform: Transform attaching to structures the graph the model
            reads, the one of the class of the model. The estimator applies it
            to the structures it is given without a graph; give it to a
            dataset to build the graphs when the structures are loaded.
        hull_source: Where the hull comes from: the file it was loaded from
            and, for from_lematerial, the structures it was built with.

    Energy above the convex hull of LeMat-Bulk:

    ```python
    estimator = EnergyEstimator.from_lematerial(device="cuda")
    print(estimator.hull_source)

    # batch of structures on the device of the model
    above = estimator.energy_above_hull(structures)

    # dataset
    energy, above = estimator.predict_energy_above_hull(HDF5Dataset("structures.hdf5"))
    ```
    """

    def __init__(
        self,
        model: OrbEnergyModel,
        correction: Float[np.ndarray, "128"] | None = None,
        hull: Hull | None = None,
        hull_source: str | None = None,
    ):
        self.model = model
        self.transform = model.transform()
        self.device = next(model.parameters()).device
        self.correction: torch.Tensor | None = None
        self.hull = hull
        self.hull_source = hull_source if hull is not None else None

        if correction is not None:
            self.correction = torch.tensor(
                correction, dtype=torch.float64, device=self.device
            )

    @classmethod
    def from_pretrained(
        cls,
        name: str | None = None,
        correction_file: str | os.PathLike | None = None,
        hull_file: str | os.PathLike | None = None,
        device: torch.device | str | None = None,
    ) -> EnergyEstimator:
        """Load a pretrained model and, optionally, a correction and a hull.

        Args:
            name: Name of the pretrained model. By default the model the
                correction was fitted for, or orb-v3-conservative-inf-mpa.
            correction_file: Correction saved by
                torchms.estimator.calibration.save_correction.
            hull_file: Hull saved by torchms.data.hull.save_hull.
            device: Device of the model.

        Raises:
            ValueError: If the correction was fitted for another model.
        """
        correction = None

        if correction_file is not None:
            correction, _, attributes = load_correction(correction_file)
            fitted_for = attributes.get("model")

            if name is None and fitted_for is not None:
                name = str(fitted_for)
            if fitted_for is not None and fitted_for != name:
                raise ValueError(
                    f'the correction was fitted for "{fitted_for}", not for "{name}"'
                )

        hull = None if hull_file is None else load_hull(hull_file)
        hull_source = None if hull_file is None else f"file {os.fspath(hull_file)}"
        model = OrbEnergyModel.from_pretrained(name or DEFAULT_MODEL, device=device)

        return cls(model, correction, hull, hull_source)

    @classmethod
    def from_lematerial(
        cls,
        device: torch.device | str | None = None,
        hull: bool = True,
        revision: str | None = None,
    ) -> EnergyEstimator:
        """Load the estimator of the energies of LeMat-Bulk (PBE).

        The model is orb-v3-conservative-inf-mpa. The correction and the hull
        are downloaded from the Hugging Face dataset repository
        materials-toolkits/lematerial:

        * "energy_correction_orb.hdf5": correction of the energies of the
          model, fitted on structures of LeMat-Bulk (compatible_pbe) from the
          Materials Project and Alexandria.
        * "hull.hdf5": convex hull of the structures of LeMat-Bulk
          (compatible_pbe) from the Materials Project and Alexandria, built
          from their DFT total energies by scripts/build_lematerial.py --hull.
          The structures of LeMat-Bulk from OQMD are left out: their energies
          are shifted for nine elements (Gd, Cu, Ni, Pu, Bi, Pa, Fe, Np, Re).
          It is not the hull of LeMat-Bulk-DFT-Hull-All, which mixes the three
          sources and gives the energy_above_hull stored in the dataset.

        Requires huggingface_hub (pip install torchms[hub]).

        Args:
            device: Device of the model.
            hull: If False the hull is not loaded: the estimator only gives
                total energies.
            revision: Revision of the repository, the last one by default.
        """
        from huggingface_hub import (  # pylint: disable=import-outside-toplevel
            hf_hub_download,
        )

        def fetch(filename: str) -> str:
            if revision is None:
                return hf_hub_download(
                    LEMATERIAL_REPOSITORY, filename, repo_type="dataset"
                )
            return hf_hub_download(
                LEMATERIAL_REPOSITORY, filename, repo_type="dataset", revision=revision
            )

        estimator = cls.from_pretrained(
            correction_file=fetch(LEMATERIAL_CORRECTION),
            hull_file=fetch(LEMATERIAL_HULL) if hull else None,
            device=device,
        )

        if hull:
            estimator.hull_source = (
                f"{LEMATERIAL_HULL_SOURCE}: {LEMATERIAL_HULL} of the Hugging Face "
                f"dataset {LEMATERIAL_REPOSITORY}"
                + ("" if revision is None else f", revision {revision}")
            )

        return estimator

    def __repr__(self) -> str:
        correction = "none" if self.correction is None else "per element"
        hull = "none" if self.hull is None else self.hull_source or "unknown source"
        return (
            f"EnergyEstimator(model={type(self.model).__name__}, "
            f"correction={correction}, hull={hull})"
        )

    @torch.no_grad()
    def __call__(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Estimate the total energy of each structure of a batch, in eV.

        Structures must be on the device of the model. The model reads their
        graph: structures without a graph are given to the transform of the
        estimator first, structures with a graph are expected to have the one
        of this transform.
        """
        if structures.graph is None:
            structures = self.transform(structures)

        energy = self.model(structures)

        if self.correction is not None:
            energy = energy.index_add(
                0, structures.batch, self.correction[structures.z]
            )

        return energy

    def predict(
        self,
        dataset: HDF5Dataset,
        indices: Int[np.ndarray, "n"] | None = None,
        max_atoms: int = 1500,
        progress: bool = False,
    ) -> Float[np.ndarray, "n"]:
        """Estimate the total energy of structures of a dataset, in eV.

        Args:
            dataset: Dataset of structures.
            indices: Structures to estimate, in ascending order, all by default.
            max_atoms: Number of atoms of a batch, limits the memory used.
            progress: Show a progress bar.

        Returns:
            Estimated total energy of each structure, in the order of indices.
        """
        if indices is None:
            indices = np.arange(len(dataset))
        indices = np.asarray(indices, dtype=np.int64)

        if len(indices) > 1 and (np.diff(indices) <= 0).any():
            raise ValueError("indices must be in ascending order")

        # the arrays of the file have a row per structure of the file
        num_atoms = dataset.dataset("num_atoms")[:][dataset.rows(indices)]
        energy = np.zeros(len(indices), dtype=np.float64)

        progress_bar = tqdm.tqdm(
            total=len(indices), unit="struct", disable=not progress
        )
        for start, stop in atom_batches(num_atoms, max_atoms):
            structures = to_device(dataset[indices[start:stop]], self.device)
            energy[start:stop] = self(structures).cpu().numpy()
            progress_bar.update(stop - start)
        progress_bar.close()

        return energy

    def _reference_hull(self) -> Hull:
        if self.hull is None:
            raise ValueError(
                "the estimator has no hull: load one with from_lematerial or from_pretrained"
            )
        return self.hull

    def energy_above_hull(
        self, structures: Structures, num_workers: int = 1
    ) -> Float[torch.Tensor, "n_struct"]:
        """Estimate the energy above the hull of each structure of a batch.

        Args:
            structures: Batch of structures, on the device of the model.
            num_workers: Number of processes computing the hulls.

        Returns:
            Energy above the hull in eV/atom. It is zero on the hull, negative
            for a structure estimated more stable than the hull, and NaN for a
            structure with an element absent from the hull.

        Raises:
            ValueError: If the estimator has no hull.
        """
        hull = self._reference_hull()
        energy = self(structures)

        above = energy_above_reference(
            hull,
            structures.z.cpu().numpy(),
            structures.num_atoms.cpu().numpy(),
            energy.cpu().numpy(),
            num_workers,
        )

        return torch.tensor(above, dtype=torch.float64, device=energy.device)

    def predict_energy_above_hull(
        self,
        dataset: HDF5Dataset,
        indices: Int[np.ndarray, "n"] | None = None,
        max_atoms: int = 1500,
        num_workers: int = 1,
        progress: bool = False,
    ) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
        """Estimate the energy above the hull of structures of a dataset.

        Args:
            dataset: Dataset of structures.
            indices: Structures to estimate, in ascending order, all by default.
            max_atoms: Number of atoms of a batch, limits the memory used.
            num_workers: Number of processes computing the hulls.
            progress: Show a progress bar.

        Returns:
            Estimated total energy in eV and estimated energy above the hull in
            eV/atom (see energy_above_hull) of each structure, in the order of
            indices.

        Raises:
            ValueError: If the estimator has no hull.
        """
        hull = self._reference_hull()

        if indices is None:
            indices = np.arange(len(dataset))
        indices = np.asarray(indices, dtype=np.int64)

        energy = self.predict(dataset, indices, max_atoms, progress)

        # atoms of the selected structures, in the same order
        rows = dataset.rows(indices)
        num_atoms = dataset.dataset("num_atoms")[:][rows]
        first = dataset.dataset("ptr")[:][rows]
        start = np.cumsum(num_atoms) - num_atoms
        atoms = np.repeat(first - start, num_atoms) + np.arange(int(num_atoms.sum()))
        z = dataset.dataset("z")[:][atoms]

        above = energy_above_reference(
            hull, z, num_atoms, energy, num_workers, progress
        )

        return energy, above
