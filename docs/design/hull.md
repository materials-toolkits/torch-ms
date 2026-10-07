# Convex hull

## Computation

The energy above the hull of a structure is its energy per atom minus the energy of the most
stable mix of phases with the same composition. `pycrystalgen.data.hull` computes it from
total energies:

1. **Compositions**: elements of each structure and fraction of its atoms.
2. **References**: for each element, the lowest energy per atom of the structures made of
   this element only.
3. **Formation energy**: energy per atom minus the references weighted by the composition.
   Elements are then at zero; this shift does not change distances to the hull.
4. **One chemical system at a time**, from binaries to the systems with the most elements. The
   points defining the hull of a system are its elements, the stable structures of all its
   sub-systems (already known), and its own structures with a negative formation energy. The
   lower convex hull of these points is computed with qhull (SciPy), and evaluated at the
   composition of each structure.
5. A structure closer than 1e-8 eV/atom to the hull is on the hull, with exactly 0.

A system where qhull fails (degenerate points) is solved with linear programs. Systems of a
level are independent and computed in parallel. LeMat-Bulk (5.3 million structures, 673,173
chemical systems) takes about 30 seconds.

## Saved hull

A hull only depends on its stable structures. `save_hull` stores their composition and energy:

| Dataset | Content |
|---|---|
| `elements` (n_pairs,) | atomic number of the elements of each structure |
| `counts` (n_pairs,) | number of atoms of this element |
| `ptr` (n_struct + 1,) | index of the first element of each structure |
| `energy_per_atom` (n_struct,) | eV, in 64 bits |

`energy_above_reference` compares other structures with a loaded hull, without adding them to
it: the result is negative for a structure more stable than the hull. It also works for a
chemical system without any structure in the hull, from its sub-systems.

## Energies must be comparable

A DFT total energy depends on the settings of the calculation: functional, pseudopotentials,
Hubbard corrections, code. Two databases with different settings differ by an amount which
depends on the elements, and a hull mixing them is wrong.

## The hull of LeMat-Bulk, without OQMD

LeMat-Bulk gathers structures of the Materials Project, Alexandria and OQMD. For nine
elements, the energies of OQMD are not on the scale of the two others:

| Element | Shift of OQMD (eV per atom of the element) | OQMD structures with the element |
|---|---|---|
| Gd | +9.45 | 24,063 |
| Cu | +0.38 | 58,262 |
| Ni | +0.20 | 47,108 |
| Pu | +0.19 | 11,712 |
| Bi | −0.19 | 19,725 |
| Pa | +0.16 | 11,905 |
| Fe | +0.14 | 37,826 |
| Np | +0.07 | 8,392 |
| Re | +0.05 | 19,076 |

**How it was measured.** The energy model predicts energies on the scale of the Materials
Project and Alexandria. The difference between its predictions and the energies of the
dataset, per atom of each element, is the same for these two sources and differs by the value
above for OQMD. For Gd, Cu and Ni, the same value is the difference between the lowest energy
of the element alone in OQMD and in the two other sources.

**Consequence.** With a positive shift, structures of OQMD look less stable than they are. With
a negative shift (Bi), they look more stable: in the hull of the three sources, 1,316 of the
3,044 stable structures with bismuth came from OQMD.

**Decision.** `hull.hdf5` is built from the 4,767,353 structures of the Materials Project and
Alexandria. It has 102,910 stable structures.

| Compared with the hull of the three sources | |
|---|---|
| Structures of the two sources newly on the hull | 7,487 |
| Energy above the hull unchanged (within 1e-4 eV/atom) | 89.9% |
| Lower by more than 0.01 eV/atom | 69,148 |
| Lower by more than 0.05 eV/atom | 18,070 |

**What is lost.** OQMD is left out for all elements. A phase which is stable and only known
from OQMD is not on the hull: 14,693 structures of OQMD without shifted element are below it,
most of them by very little (median 0.002 eV/atom), 2,589 by more than 0.01. In their chemical
systems the hull is too high by this amount.

**What is not changed.** `all.hdf5` and `unique.hdf5` keep the published energy above the hull,
computed with the three sources.

## Checks

- Comparing the structures of the two sources with the saved hull gives back their energy
  above the hull within 1e-14 eV/atom.
- The same computation on the three sources reproduces the published values of
  LeMat-Bulk-DFT-Hull-All within 1e-6 eV/atom for 5,335,263 of 5,335,265 structures. The two
  others are in the As-H-Hg-O-Zn system, where a stable structure has no published value.
- Unit tests compare with hand computed binary and ternary cases, and with an independent
  solution by linear programming on random datasets.
