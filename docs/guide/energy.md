# Energy above the hull

The energy above the convex hull tells how far a crystal is from the most stable combination
of phases with the same composition: 0 for a stable crystal, usually below 0.1 eV/atom for a
crystal which can be made. There are three ways to get it.

## 1. Read it from a dataset

Most datasets store the value of their source in `energy_above_hull`. Check the page of the
dataset first: the value of LeMat-Bulk mixes sources which are not comparable for some
elements, and the one of OQMD is negative for the most stable phases.

## 2. Estimate it for any structure

`EnergyEstimator` predicts the total energy of a structure with a pretrained model and compares
it to the convex hull of LeMat-Bulk.

```python
from pycrystalgen.data import load_dataset
from pycrystalgen.estimator import EnergyEstimator

estimator = EnergyEstimator.from_lematerial(device="cuda")
print(estimator.hull_source)

# a dataset: total energy in eV and energy above the hull in eV/atom
energy, above = estimator.predict_energy_above_hull(load_dataset("mp-20", "test"))

# a batch of Structures on the device of the model
above = estimator.energy_above_hull(structures)
```

- The result is **negative** for a structure estimated more stable than the hull, and **NaN**
  for a structure with an element absent from the hull, or with two atoms at the same position.
- The energy is the one of the structure **as given**: it is not relaxed. A generated crystal
  must be relaxed before its energy is compared with a hull of relaxed structures.
- The hull loaded by `from_lematerial` is the one of the structures of LeMat-Bulk from the
  Materials Project and Alexandria, without the ones from OQMD. See
  [Convex hull](../design/hull.md) for the reason.

`estimator(structures)` and `estimator.predict(dataset)` give the total energy alone.

### How accurate is it

Measured on structures which were not used to fit the estimator
([details](../design/estimator.md)):

| Structures | Mean error of the energy | Median | Same side of 0.1 eV/atom as DFT |
|---|---|---|---|
| Materials Project and Alexandria, held out | 0.0097 eV/atom | 0.0042 | 98.9% |
| OQMD, structures absent from both | 0.0105 eV/atom | 0.0056 | 98.9% |
| OQMD, compositions absent from both | 0.0123 eV/atom | 0.0067 | 99.1% |

It tells reliably if a structure is **within 0.1 eV/atom** of the hull. It does not tell
reliably if a structure is **exactly on** the hull: an error of 0.01 eV/atom is as large as the
difference between a stable phase and its close competitors.

## 3. Compute it from DFT energies

`pycrystalgen.data.hull` computes the hull itself. Energies must be comparable: same
functional and same calculation settings.

```python
from pycrystalgen.data.hull import energy_above_hull, energy_above_reference, load_hull

# hull of a set of structures, from its own energies
above = energy_above_hull(z, num_atoms, total_energy, num_workers=8)

# structures against a saved hull
hull = load_hull("hull.hdf5")
above = energy_above_reference(hull, z, num_atoms, total_energy)
```

`z` is the atomic number of all the atoms, `num_atoms` and `total_energy` have one value per
structure: numpy arrays, as stored in the datasets.

```python
dataset = load_dataset("oqmd", "unique")
z, num_atoms = dataset.dataset("z")[:], dataset.dataset("num_atoms")[:]
total_energy = dataset.dataset("total_energy")[:]
```

`save_hull` writes the structures on the hull to a small file, and `fill_missing` completes the
missing values of a dataset. The same functions are available from the command line:

```bash
uv run python scripts/convex_hull.py dataset.hdf5 above.npy
uv run python scripts/convex_hull.py dataset.hdf5 above.npy --save-hull hull.hdf5
uv run python scripts/convex_hull.py generated.hdf5 above.npy --reference hull.hdf5
```

!!! warning "Energies from different sources"
    A hull compares energies across compositions. Energies computed with different settings
    differ by an amount which depends on the elements, and a hull mixing them is wrong.
    Carbon is at -9.2 eV/atom in LeMat-Bulk and at -154 eV/atom in Carbon-24: the same
    element, on two scales.
