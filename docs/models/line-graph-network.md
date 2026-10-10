# LineGraphNetwork

A network of the energy, written for this library, which lists the triplets and the quadruplets of atoms and uses
their angles, as ALIGNN, M3GNet, CHGNet or DPA-3. It has no pretrained weights, and no model of the library is built on it.

| | |
|---|---|
| Class | `torchms.models.layers.LineGraphNetwork` |
| Predicts | one energy per structure |
| Graph | pairs, triplets, quadruplets |
| Terms on three atoms and more | angles and dihedral angles |

```{.python notest}
from torchms.data import load_dataset
from torchms.models.layers import LineGraphNetwork

model = LineGraphNetwork(order=2, radius=(5.0, 3.5))

# the graph and the triplets the network reads, attached when a structure is loaded
dataset = load_dataset("mp-20", "train", transform=model.transform())
energy = model(dataset[:32])        # (32,)
```

The network does not build its graph. `model.transform()` gives the transform which attaches
what it reads, for its order and its radii: `AddRadiusGraph(radius[0], self_images=True)`,
then `AddTriplets(radius[1], strict=True)` from order 2 and
`AddQuadruplets(radius[2], strict=True)` for order 3. Structures without them are refused.

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

## What it guarantees

As [EquivariantNetwork](equivariant-network.md), the energy does not depend on the description of the crystal, and it is checked in the tests
with the transforms of [data augmentation](../guide/augmentation.md):

- same energy after a permutation of the atoms, a rotation or a reflection of the cell, a
  translation of the atoms and a change of basis of the lattice;
- energy multiplied by the number of cells in a supercell;
- same energy for a structure alone and in a batch;
- every term vanishes smoothly at its radius.

## Compared with EquivariantNetwork

- Terms near the radius are small in both models, because of the smooth cutoff. The terms of
  `LineGraphNetwork` on three atoms carry the cutoff of two bonds and are smaller still: use a
  radius of the angles well above the bond lengths of the crystal.
- The cost of `LineGraphNetwork` grows with the number of triplets; the one of
  `EquivariantNetwork` with the number of pairs.

Neither model was trained here: their accuracy is not measured.
