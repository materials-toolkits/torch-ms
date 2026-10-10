# orb-v3

Total energy of a structure predicted by orb-v3 conservative, the model of the [energy estimator](../guide/energy.md).

| | |
|---|---|
| Class | `torchms.models.OrbEnergyModel` |
| Predicts | total energy in eV, one per structure, in double precision |
| Source | [orb-models](https://github.com/orbital-materials/orb-models) 0.7.0, Orbital Materials |
| License | Apache-2.0 |
| Paper | [arXiv:2504.06231](https://arxiv.org/abs/2504.06231) |
| Weights | `orb-v3-conservative-inf-mpa` (default), `orb-v3-conservative-inf-omat`; official, 102 MB each |
| Transform | `AddRadiusGraph(6.0, self_images=True)`, `KeepNearestNeighbors(120)` |
| Largest difference with the original | 2.7e-5 eV/atom below 120 neighbours, 1.7e-4 above |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import OrbEnergyModel

dataset = load_dataset("mp-20", "test", transform=OrbEnergyModel.transform())
model = OrbEnergyModel.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))        # (32,), eV
```

## The rewrite

The model is orb-v3 conservative of Orbital Materials, checkpoint
`orb-v3-conservative-inf-mpa`
([orb-models](https://github.com/orbital-materials/orb-models), Apache-2.0,
[arXiv:2504.06231](https://arxiv.org/abs/2504.06231)).

`orb-models` needs Python 3.12 and several dependencies the project does not have. Its
prediction of the energy is therefore **rewritten** in `torchms/models/orb/`, for inference only,
with PyTorch alone: atom embedding, edge features (radial basis × spherical harmonics), five
attention layers, an energy head with a reference energy per element, and a repulsion between
close atoms. Forces, stress and training are not included. The official weights (102 MB) are
downloaded on the first use and checked against a pinned SHA-256.

The graph is built by `torchms.graph`: atoms closer than 6 Å, at most the 120 nearest,
periodic images of the atom itself included.

## Comparison with orb-models

Energies of `orb-models` 0.7.0 were computed once, outside of the project, for 253 structures:

| | Difference with orb-models |
|---|---|
| Mean | 1.5e-6 eV/atom |
| Largest, cells below 120 neighbours | 2.7e-5 eV/atom |
| Largest, cells above 120 neighbours | 1.7e-4 eV/atom |

Above 120 neighbours, atoms at the same distance are not kept in the same order as
`orb-models`. A test compares 40 of these structures at each run, when the weights are in the
cache.

## Behaviours kept from orb-models

The weights were trained with them, so they are reproduced:

- the repulsion between close atoms uses the atomic number **plus one**;
- this repulsion is **averaged** over the atoms of a structure instead of summed: this small
  term does not grow with the cell;
- the network is **not exactly invariant to rotations**: the energy changes by about
  2e-4 eV/atom on average, up to 8e-3, with the orientation of the cell.

## In the energy estimator

`EnergyEstimator` adds to this energy a correction per element and compares it to a convex
hull: see [the design note](../design/estimator.md) for the choice of the checkpoint, the
correction and its validation.
