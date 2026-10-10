# M3GNet

`M3gnetEnergyModel` is the M3GNet potential `M3GNet-PES-MatPES-PBE-2025.2`
([matgl](https://github.com/materialsvirtuallab/matgl), BSD 3-Clause,
[arXiv:2202.02450](https://arxiv.org/abs/2202.02450)), in `torchms/models/m3gnet/`. Atoms closer
than 5 Å are bonded; a bond is updated from the angles it forms with the other bonds of its
atom of at most 4 Å (`triplets` with a mask), in a basis of spherical Bessel functions and
spherical harmonics. matgl builds these functions with sympy; they are computed here with
torch, their roots included. Forces and stress are not included.

| | |
|---|---|
| Class | `torchms.models.M3gnetEnergyModel` |
| Predicts | total energy in eV, one per structure, in double precision |
| Source | [matgl](https://github.com/materialsvirtuallab/matgl) 4.1.0 |
| License | BSD 3-Clause |
| Paper | [arXiv:2202.02450](https://arxiv.org/abs/2202.02450) |
| Weights | `m3gnet-matpes-pbe-2025.2`, official |
| Transform | `AddRadiusGraph(5.0, self_images=True)`, `AddTriplets(radius=4.0)` |
| Largest difference with the original | 1.6e-6 eV/atom |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import M3gnetEnergyModel

dataset = load_dataset("mp-20", "test", transform=M3gnetEnergyModel.transform())
model = M3gnetEnergyModel.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))        # (32,), eV
```

## Comparison with the original

| Against matgl 4.1.0, 40 structures | |
|---|---|
| Edges and angles of each structure (128,348 angles) | identical |
| Roots of the spherical Bessel functions | identical |
| Largest difference of energy | 1.6e-6 eV/atom |
| Mean difference of energy | 4.6e-7 eV/atom |

The model knows 89 elements, H to Bi and Ac to Pu: a structure with another element raises an
error.
