# Models

`pycrystalgen.models` has two trainable models of the energy with terms on more than two
atoms. They follow the two ways recent models do it:

| Model | Graph | Terms on three atoms and more | As in |
|---|---|---|---|
| `LineGraphNetwork` | pairs, triplets, quadruplets | angles and dihedral angles | ALIGNN, M3GNet, CHGNet, DPA-3 |
| `EquivariantNetwork` | pairs only | products of sums of spherical harmonics | MACE |

Both take a batch of `Structures` and return one energy per structure. There is no training
loop in the library: they are `torch.nn.Module`.

```python
from pycrystalgen.data import load_dataset
from pycrystalgen.models import EquivariantNetwork, LineGraphNetwork

structures = load_dataset("mp-20", "train")[:32]

model = LineGraphNetwork(order=2, radius=(5.0, 3.5))
energy = model(structures)          # (32,)

model = EquivariantNetwork(correlation=3, radius=5.0)
energy = model(structures)          # (32,)
```

## What both guarantee

The energy does not depend on the description of the crystal, and it is checked in the tests
with the transforms of [data augmentation](augmentation.md):

- same energy after a permutation of the atoms, a rotation or a reflection of the cell, a
  translation of the atoms and a change of basis of the lattice;
- energy multiplied by the number of cells in a supercell;
- same energy for a structure alone and in a batch;
- every term vanishes smoothly at its radius.

## LineGraphNetwork

Features live on the atoms and on the bonds, the pairs of atoms closer than the first radius.
`order` adds the higher terms:

| `order` | Adds | Initialized from | Radius |
|---|---|---|---|
| 1 | — | — | `radius[0]` |
| 2 | triplets: two bonds around an atom | their angle | `radius[1]` |
| 3 | quadruplets: chains of three bonds | their dihedral angle | `radius[2]` |

A layer updates from the highest order down: a dihedral angle updates its middle bond, an
angle its first bond, a bond its atom. Each order has its own radius because the number of
triplets grows with the square of the number of neighbours and the number of quadruplets with
its cube: M3GNet uses 5 Å for the bonds and 4 Å for the angles, CHGNet 6 Å and 3 Å.

`chiral=True` gives the sines of the dihedral angles to the model, which then distinguishes a
crystal from its mirror image. It is no longer invariant to reflections: use
`LatticeRotation(proper=True)` to augment its data.

## EquivariantNetwork

No triplet is listed. An atom sums over its neighbours a learned function of the distance
times the spherical harmonics of the direction (l ≤ 3). `correlation` is the number of
factors of the products of this sum with itself:

| `correlation` | Invariants of each channel | Atoms in a term |
|---|---|---|
| 1 | the constant part of the sum | 2 |
| 2 | its squared norm in each degree l | 3 |
| 3 | the scalar product of the sum with its own square, in each degree l | 4 |

The square is taken with the Gaunt coefficients (`gaunt_coefficients`), computed in the
library by an exact quadrature on the sphere. They are the couplings which do not change in a
mirror: the model cannot distinguish a crystal from its mirror image.

This is not MACE: MACE learns one weight per coupling of its symmetric contraction and keeps
vector and tensor features between layers in its larger models. `MaceEnergyModel`, in
`pycrystalgen.estimator`, is the exact rewrite of its small model.

## Which one

- Terms near the radius are small in both models, because of the smooth cutoff. The terms of
  `LineGraphNetwork` on three atoms carry the cutoff of two bonds and are smaller still: use a
  radius of the angles well above the bond lengths of the crystal.
- The cost of `LineGraphNetwork` grows with the number of triplets; the one of
  `EquivariantNetwork` with the number of pairs.

Neither model was trained here: their accuracy is not measured.
