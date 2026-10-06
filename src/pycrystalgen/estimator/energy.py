"""Estimation of the total energy of crystal structures."""

from __future__ import annotations

import os

import numpy as np
import torch
import tqdm
from jaxtyping import Float, Int

from pycrystalgen.data.batching import Structures
from pycrystalgen.data.hdf5 import HDF5Dataset
from pycrystalgen.estimator.calibration import load_correction
from pycrystalgen.estimator.orb import OrbEnergyModel

HUB_REPOSITORY = "materials-toolkits/lematerial"
HUB_CORRECTION = "energy_correction_orb.hdf5"


def to_device(structures: Structures, device: torch.device | str) -> Structures:
    """Copy of structures with its tensors on a device."""
    return Structures(
        lattice=structures.lattice.to(device),
        x=structures.x.to(device),
        z=structures.z.to(device),
        num_atoms=structures.num_atoms.to(device),
        batch=structures.batch.to(device),
        ptr=structures.ptr.to(device),
        total_energy=structures.total_energy.to(device),
        energy_above_hull=structures.energy_above_hull.to(device),
        band_gap=structures.band_gap.to(device),
        formation_energy_per_atom=structures.formation_energy_per_atom.to(device),
        _validate=True,
    )


def atom_batches(
    num_atoms: Int[np.ndarray, "n"], max_atoms: int
) -> list[tuple[int, int]]:
    """Split consecutive structures in batches of about max_atoms atoms.

    Returns:
        (start, stop) of each batch. A batch is closed by the structure with
        which it reaches max_atoms atoms.

    Example:
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
    """Total energy of crystal structures, from a pretrained model.

    The energy is predicted by the model for the structure as given (it is not
    relaxed), then corrected by an energy per atom of each element fitted on a
    dataset (see pycrystalgen.estimator.calibration), to be comparable with
    the energies of this dataset.

    The correction of from_hub is fitted on the structures of LeMat-Bulk from
    the Materials Project and Alexandria, the ones defining its convex hull
    (hull.hdf5). It is not fitted on the structures from OQMD, whose energies
    are shifted for some elements: for them the estimated energy is on the
    scale of the two other sources, not the energy stored in LeMat-Bulk.

    Args:
        model: Pretrained model.
        correction: Correction in eV/atom indexed by atomic number, shape
            (128,), or None for the energy of the model.

    Example:
        Energy above the convex hull of LeMat-Bulk::

            estimator = EnergyEstimator.from_hub(device="cuda")
            energy = estimator.predict(HDF5Dataset("structures.hdf5"))
    """

    def __init__(
        self, model: OrbEnergyModel, correction: Float[np.ndarray, "128"] | None = None
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.correction: torch.Tensor | None = None

        if correction is not None:
            self.correction = torch.tensor(
                correction, dtype=torch.float64, device=self.device
            )

    @classmethod
    def from_pretrained(
        cls,
        name: str = "orb-v3-conservative-inf-omat",
        correction_file: str | os.PathLike | None = None,
        device: torch.device | str | None = None,
    ) -> EnergyEstimator:
        """Load a pretrained model and, optionally, a saved correction.

        Raises:
            ValueError: If the correction was fitted for another model.
        """
        correction = None

        if correction_file is not None:
            correction, _, attributes = load_correction(correction_file)
            fitted_for = attributes.get("model")
            if fitted_for is not None and fitted_for != name:
                raise ValueError(
                    f'the correction was fitted for "{fitted_for}", not for "{name}"'
                )

        return cls(OrbEnergyModel.from_pretrained(name, device=device), correction)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = HUB_REPOSITORY,
        filename: str = HUB_CORRECTION,
        device: torch.device | str | None = None,
    ) -> EnergyEstimator:
        """Load the estimator of the energies of LeMat-Bulk (PBE).

        The correction is downloaded from a Hugging Face dataset repository
        and the model it was fitted for is loaded. Requires huggingface_hub
        (pip install pycrystalgen[hub]).
        """
        from huggingface_hub import (  # pylint: disable=import-outside-toplevel
            hf_hub_download,
        )

        correction_file = hf_hub_download(repo_id, filename, repo_type="dataset")
        _, _, attributes = load_correction(correction_file)

        return cls.from_pretrained(str(attributes["model"]), correction_file, device)

    @torch.no_grad()
    def __call__(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Estimate the total energy of each structure of a batch, in eV.

        Structures must be on the device of the model.
        """
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

        num_atoms = dataset.dataset("num_atoms")[:][indices]
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
