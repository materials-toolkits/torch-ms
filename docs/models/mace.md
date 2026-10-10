# MACE

`MaceEnergyModel` is MACE-MP-0 small
([mace](https://github.com/ACEsuit/mace), MIT,
[arXiv:2206.07697](https://arxiv.org/abs/2206.07697)), in `torchms/models/mace/`. Its graph has
pairs only, the atoms closer than 6 Å: terms on three and four atoms come from products. An
atom sums over its neighbours a radial function times spherical harmonics (l ≤ 3) times the
features of the neighbour; products of two and three of these sums are contracted to
invariant features with coupling tensors. The layers of e3nn used by mace (linear layers,
tensor products, normalized activations) are written with torch alone.

| | |
|---|---|
| Class | `torchms.models.MaceEnergyModel` |
| Predicts | total energy in eV, one per structure, in double precision |
| Source | [mace](https://github.com/ACEsuit/mace) 0.3.16 |
| License | MIT |
| Paper | [arXiv:2206.07697](https://arxiv.org/abs/2206.07697) |
| Weights | `mace-mp-0-small`, official |
| Transform | `AddRadiusGraph(6.0, self_images=True)` |
| Largest difference with the original | 5e-15 eV/atom, in double precision |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import MaceEnergyModel

dataset = load_dataset("mp-20", "test", transform=MaceEnergyModel.transform())
model = MaceEnergyModel.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))        # (32,), eV
```

## Comparison with the original

| Against mace 0.3.16, 40 structures, double precision | |
|---|---|
| Edges of each structure | identical |
| Largest difference of energy | 5e-15 eV/atom |

The energy is exactly invariant to the rotations and reflections of `LatticeRotation`, unlike
orb: it is checked in the tests, with the invariance of the coupling tensors.

**The weights are read without mace.** mace publishes a pickled model, the whole Python
object and not its tensors: unpickling it normally needs mace and e3nn. `torchms/models/mace/reader.py`
unpickles the official file with every class of mace and e3nn replaced by a placeholder, and
collects the tensors from the placeholders: the 69 tensors are the ones of the state dict of
mace, value for value. The unpickler only resolves a fixed list of names (tensors of torch,
containers of the standard library), so the file cannot run code; this is tested with a file
which runs a function when unpickled as usual. The coupling tensors are among these tensors:
mace stores them with the weights, and the weights are only valid for this choice of tensors.

Only the small model, whose features are invariant, is rewritten. The medium and large models
have vector and tensor features, which need Clebsch-Gordan coefficients in the tensor products
between layers.
