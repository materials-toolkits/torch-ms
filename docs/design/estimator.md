# Energy estimator

`pycrystalgen.estimator` estimates the PBE total energy of a structure on the scale of the
[hull of LeMat-Bulk](hull.md).

## Model

The model is orb-v3 conservative of Orbital Materials, checkpoint
`orb-v3-conservative-inf-mpa`
([orb-models](https://github.com/orbital-materials/orb-models), Apache-2.0,
[arXiv:2504.06231](https://arxiv.org/abs/2504.06231)).

`orb-models` needs Python 3.12 and several dependencies the project does not have. Its
prediction of the energy is therefore **rewritten** in `estimator/orb/`, for inference only,
with PyTorch alone: atom embedding, edge features (radial basis × spherical harmonics), five
attention layers, an energy head with a reference energy per element, and a repulsion between
close atoms. Forces, stress and training are not included. The official weights (102 MB) are
downloaded on the first use and checked against a pinned SHA-256.

The graph is built by `pycrystalgen.graph`: atoms closer than 6 Å, at most the 120 nearest,
periodic images of the atom itself included.

### Is the rewrite faithful

Energies of `orb-models` 0.7.0 were computed once, outside of the project, for 253 structures:

| | Difference with orb-models |
|---|---|
| Mean | 1.5e-6 eV/atom |
| Largest, cells below 120 neighbours | 2.7e-5 eV/atom |
| Largest, cells above 120 neighbours | 1.7e-4 eV/atom |

Above 120 neighbours, atoms at the same distance are not kept in the same order as
`orb-models`. A test compares 40 of these structures at each run, when the weights are in the
cache.

### Behaviours kept from orb-models

The weights were trained with them, so they are reproduced:

- the repulsion between close atoms uses the atomic number **plus one**;
- this repulsion is **averaged** over the atoms of a structure instead of summed: this small
  term does not grow with the cell;
- the network is **not exactly invariant to rotations**: the energy changes by about
  2e-4 eV/atom on average, up to 8e-3, with the orientation of the cell.

## Choice of the checkpoint

Two checkpoints were compared on the same structures of LeMat-Bulk.

| Mean error (eV/atom) | `mpa` | `omat` corrected |
|---|---|---|
| Materials Project and Alexandria, held out | 0.0097 | 0.0138 |
| OQMD, new structures | 0.0105 | 0.0141 |
| OQMD, new compositions | 0.0123 | 0.0157 |

`mpa` is trained on energies computed like the ones of the Materials Project and is already on
the scale of LeMat-Bulk; `omat` is 0.064 eV/atom off before correction. The predictions of
`omat` published by LeMaterial are reproduced by the rewrite (median difference 1e-6 eV/atom).

## Correction

An energy per atom of each element is added to the prediction, fitted on 300,000 structures of
the Materials Project and Alexandria to minimize the absolute error
(`estimator/calibration.py`). For `mpa` it is small: 0.003 eV/atom on average, at most 0.024.
It lowers the error of the held-out set from 0.0100 to 0.0097 eV/atom and the share of stable
structures found from 27% to 38%; it changes nothing on the structures of OQMD. A least squares
fit made the error slightly larger and is not used.

## Validation

Structures of LeMat-Bulk which were not used to fit, and never share a fingerprint with a
fitted one. The network is trained on data from the Materials Project and Alexandria, not
OQMD: sets B and C are structures of OQMD with a fingerprint (B) or a composition (C) absent
from both.

| Set | Structures | Mean error | Median | 90% | Same side of 0.1 eV/atom as DFT | Stable found | Wrongly stable |
|---|---|---|---|---|---|---|---|
| A. Materials Project and Alexandria | 60,000 | 0.0097 | 0.0042 | 0.018 | 98.9% | 38% | 35% |
| B. OQMD, new structures | 37,856 | 0.0105 | 0.0056 | 0.024 | 98.9% | 61% | 47% |
| C. OQMD, new compositions | 36,153 | 0.0123 | 0.0067 | 0.028 | 99.1% | 78% | 34% |
| D. OQMD, new structures, all elements | 59,994 | 0.136 | 0.012 | 0.115 | 92.7% | 55% | 61% |

Errors are in eV/atom. B and C leave out the nine elements whose energies in OQMD are shifted;
D includes them, and its error is the shift of the reference energies, not of the estimator.
Energies above the hull use `hull.hdf5` for DFT and for the estimator; a structure is stable
when it is on or below the hull.

`scripts/build_energy_correction.py` fits and `scripts/validate_energy.py` validates end to
end.

## Limits

- The error, 0.01 eV/atom, is as large as the distance between a stable phase and its close
  competitors: the estimator tells if a structure is near the hull, not if it is on it.
- The energy is the one of the structure as given, not relaxed.
- The training data of the network and the sources of LeMat-Bulk overlap: the error on
  structures unlike all of them is not measured here.
- A structure with two atoms at the same position has no energy (NaN).
- That OQMD is outside of the training data comes from the documentation of `orb-models`,
  which gives the Materials Project trajectories and Alexandria for this checkpoint.

## Other pretrained models

Other published models are rewritten the same way, to check the graph functions of
`pycrystalgen.graph` against models which use terms on three atoms and more. They predict the
energy only, load the official weights, and are not used by `EnergyEstimator`. No package of
these models is a dependency: their energies were computed once, outside of the project, by
the scripts `scripts/reference_*.py`, for the 40 structures of the test of orb.

### CHGNet

`ChgnetEnergyModel` is CHGNet 0.3.0
([chgnet](https://github.com/CederGroupHub/chgnet), modified BSD,
[arXiv:2302.14231](https://arxiv.org/abs/2302.14231)), in `estimator/chgnet/`. Atoms closer
than 6 Å are bonded, an edge and its reverse sharing the features of one bond
(`reverse_edges`); the angles are the pairs of bonds shorter than 3 Å around an atom
(`triplets` with a mask), in a Fourier basis. Magnetic moments, forces and stress are not
included.

| Against chgnet 0.4.2, 40 structures | |
|---|---|
| Edges, bonds and angles of each structure (29,222 angles) | identical |
| Largest difference of energy | 4.1e-6 eV/atom |
| Mean difference of energy | 4.3e-7 eV/atom |

Behaviours kept from chgnet: the cosine of an angle is multiplied by 1 - 1e-6 before the
arc cosine, and an atom without neighbour is not an error.

### M3GNet

`M3gnetEnergyModel` is the M3GNet potential `M3GNet-PES-MatPES-PBE-2025.2`
([matgl](https://github.com/materialsvirtuallab/matgl), BSD 3-Clause,
[arXiv:2202.02450](https://arxiv.org/abs/2202.02450)), in `estimator/m3gnet/`. Atoms closer
than 5 Å are bonded; a bond is updated from the angles it forms with the other bonds of its
atom of at most 4 Å (`triplets` with a mask), in a basis of spherical Bessel functions and
spherical harmonics. matgl builds these functions with sympy; they are computed here with
torch, and their roots with scipy. Forces and stress are not included.

| Against matgl 4.1.0, 40 structures | |
|---|---|
| Edges and angles of each structure (128,348 angles) | identical |
| Roots of the spherical Bessel functions | identical |
| Largest difference of energy | 1.6e-6 eV/atom |
| Mean difference of energy | 4.6e-7 eV/atom |

The model knows 89 elements, H to Bi and Ac to Pu: a structure with another element raises an
error.
