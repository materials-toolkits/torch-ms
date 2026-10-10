# EquivariantNetwork

A network of the energy, written for this library, which only uses pairs of atoms and takes products of sums of
spherical harmonics, as MACE. It has no pretrained weights, and no model of the library is built on it.

| | |
|---|---|
| Class | `torchms.models.layers.EquivariantNetwork` |
| Predicts | one energy per structure |
| Graph | pairs only |
| Terms on three atoms and more | products of sums of spherical harmonics |

```{.python notest}
from torchms.data import load_dataset
from torchms.models.layers import EquivariantNetwork

model = EquivariantNetwork(correlation=3, radius=5.0)

# the graph the network reads, attached when a structure is loaded
dataset = load_dataset("mp-20", "train", transform=model.transform())
energy = model(dataset[:32])        # (32,)
```

The network does not build its graph. `model.transform()` gives the transform which attaches
it, `AddRadiusGraph(radius, self_images=True)` for the radius of the network. Structures
without a graph are refused.

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
`torchms.models.estimator`, is the exact rewrite of its small model.

## What it guarantees

The same as [LineGraphNetwork](line-graph-network.md#what-it-guarantees): the energy does not
depend on the description of the crystal, grows with the size of the cell, and every term
vanishes smoothly at its radius. See there for which of the two to use.
