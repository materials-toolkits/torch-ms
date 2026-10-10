# CHGNet

`CHGNet` is CHGNet 0.3.0
([chgnet](https://github.com/CederGroupHub/chgnet), modified BSD,
[arXiv:2302.14231](https://arxiv.org/abs/2302.14231)), in `torchms/models/chgnet/`. Atoms closer
than 6 Å are bonded, an edge and its reverse sharing the features of one bond
(`reverse_edges`); the angles are the pairs of bonds shorter than 3 Å around an atom
(`triplets` with a mask), in a Fourier basis. Magnetic moments, forces and stress are not
included.

| | |
|---|---|
| Class | `torchms.models.CHGNet` |
| Predicts | total energy in eV, one per structure, in double precision |
| Source | [chgnet](https://github.com/CederGroupHub/chgnet) 0.4.2 |
| License | modified BSD |
| Paper | [arXiv:2302.14231](https://arxiv.org/abs/2302.14231) |
| Weights | `chgnet-0.3.0`, official |
| Transform | `AddRadiusGraph(6.0, self_images=True)`, `AddTriplets(radius=3.0, strict=True)`, `AddBonds()` |
| Largest difference with the original | 4.1e-6 eV/atom |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import CHGNet

dataset = load_dataset("mp-20", "test", transform=CHGNet.transform())
model = CHGNet.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))        # (32,), eV
```

## Comparison with the original

| Against chgnet 0.4.2, 40 structures | |
|---|---|
| Edges, bonds and angles of each structure (29,222 angles) | identical |
| Largest difference of energy | 4.1e-6 eV/atom |
| Mean difference of energy | 4.3e-7 eV/atom |

Behaviours kept from chgnet: the cosine of an angle is multiplied by 1 - 1e-6 before the
arc cosine, and an atom without neighbour is not an error.
