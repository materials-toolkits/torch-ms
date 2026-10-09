# Data augmentation

A crystal has many descriptions. The order of the atoms, the orientation and the origin of the
cell, the basis of the lattice and the size of the cell are arbitrary choices.
`torchms.data.augmentation` moves from one description to another one of the same crystal.

| Transform | Group | Changes | Function |
|---|---|---|---|
| `AtomPermutation` | permutations of the atoms | order of `x` and `z` | `permute_atoms` |
| `LatticeRotation` | O(3) or SO(3) | `lattice` | `rotate_lattice` |
| `AtomTranslation` | translations of the torus | `x` | `translate_atoms` |
| `LatticeBasisChange` | SL(3, Z) | `lattice` and `x` | `change_lattice_basis` |
| `Supercell` | — | `lattice`, `x`, `z`, `num_atoms`, `total_energy` | `make_supercell` |

## Random transforms

A transform draws one element per structure of the batch and applies it. Transforms are chained
with `Compose` and given to a dataset:

```{.python notest}
from torchms.data import load_dataset
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
    Supercell,
)

augment = Compose(
    [
        Supercell(max_repeat=2, max_atoms=64),
        LatticeBasisChange(num_shears=2),
        LatticeRotation(),
        AtomTranslation(),
        AtomPermutation(),
    ]
)

dataset = load_dataset("mp-20", "train", transform=augment)
```

Each transform accepts a `generator` for reproducible draws, on the device of the structures.
Without it the global random state of torch is used.

Augmentations come **before** the [graph transforms](graphs.md#graphs-as-transforms) in a
chain. They change the atoms or the cell, which a graph does not follow: structures with a
graph, a line graph, triplets or quadruplets come out of an augmentation without them.

```{.python notest}
from torchms.graph.transforms import AddRadiusGraph, AddTriplets

transform = Compose([augment, AddRadiusGraph(5.0), AddTriplets(3.0)])
```

## Applying a chosen element

The functions take the elements to apply, one per structure:

```python
import torch
from torchms.data import Structures
from torchms.data.augmentation import change_lattice_basis, make_supercell

structures = Structures.diamond()

# replace the lattice vector a by a + b
basis = torch.tensor([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
sheared = change_lattice_basis(structures, basis)

supercell = make_supercell(structures, torch.tensor([[2, 1, 1]]))
supercell.num_atoms   # tensor([16])
```

The elements are drawn with `random_permutation`, `random_orthogonal`, `random_translation`,
`random_unimodular` and `random_repeat`.

## Conventions

- **O(3)**: every lattice vector `v` becomes `R v`, so `lattice @ R.T` with lattice vectors as
  rows. Fractional coordinates are unchanged. Half of the draws are rotoreflections, which
  give the mirror image of a chiral crystal. `LatticeRotation(proper=True)` draws rotations
  only, in SO(3).
- **SL(3, Z)**: the new lattice is `U @ lattice` and the new coordinates `x @ U⁻¹`, wrapped to
  [0, 1). Draws are products of `num_shears` elementary matrices (a lattice vector is added to
  or subtracted from another one): the cell gets more skewed as `num_shears` grows.
- **Supercell**: the cell is repeated along its lattice vectors, atoms are ordered cell by
  cell. `total_energy` is multiplied by the number of cells, the other labels are per atom or
  intensive and unchanged. `max_atoms` leaves a structure unchanged when its supercell would be
  larger. A supercell which is not aligned with the lattice vectors is a change of basis,
  a supercell, then another change of basis.
- **Molecules**: a structure which is not periodic is permuted, rotated and translated as a
  crystal, without wrapping. A change of basis and a supercell leave it unchanged. See
  [Periodicity](../design/periodicity.md).
