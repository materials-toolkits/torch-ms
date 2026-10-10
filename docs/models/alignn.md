# ALIGNN

`ALIGNN` is the ALIGNN 2.0 model `formation_energy_peratom_radius`
([alignn](https://github.com/usnistgov/alignn), NIST,
[arXiv:2106.01829](https://arxiv.org/abs/2106.01829)), in `torchms/models/alignn/`. It predicts
the formation energy in eV/atom, not a total energy. An atom is connected to its 12 nearest
neighbours closer than 5 Å (`nearest_neighbors`); the line graph connects a bond u→v to the
bonds v→w of at most 3.5 Å, its own reverse included (`line_graph`). The features of the
elements are not stored with the weights: they are in `cgcnn_features.npz`, written by the
reference script from jarvis-tools.

| | |
|---|---|
| Class | `torchms.models.ALIGNN` |
| Predicts | formation energy in eV/atom, one per structure |
| Source | [alignn](https://github.com/usnistgov/alignn) 2026.8.11 |
| License | NIST |
| Paper | [arXiv:2106.01829](https://arxiv.org/abs/2106.01829) |
| Weights | `alignn-formation-energy-radius`, official |
| Transform | `AddRadiusGraph(5.0, self_images=True)`, `KeepNearestNeighbors(12)`, `AddLineGraph(radius=3.5, backtrack=True)` |
| Largest difference with the original | 4.8e-7 eV/atom where both keep the same 12 neighbours, 0.012 elsewhere |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import ALIGNN

dataset = load_dataset("mp-20", "test", transform=ALIGNN.transform())
model = ALIGNN.from_pretrained(device="cuda")
formation_energy = model(dataset[:32].to("cuda"))        # (32,), eV/atom
```

## Comparison with the original

| Against alignn 2026.8.11, 40 structures | |
|---|---|
| Line graph built on the graph of atoms of alignn (19,944 angles) | identical |
| Prediction on the graph of atoms of alignn, largest difference | 1.8e-6 eV/atom |
| Structures where both libraries keep the same 12 neighbours | 16 of 40 |
| Prediction on these 16 structures, largest difference | 4.8e-7 eV/atom |
| Prediction on the 24 other structures, largest difference | 0.012 eV/atom |
| Prediction on the 24 other structures, mean difference | 0.0025 eV/atom |

**The limit of 12 neighbours is not well defined.** In a crystal many neighbours are at the
same distance. When the 12th and the 13th are, the ones which are kept depend on the order of
a sort: alignn and this rewrite do not keep the same ones (124 edges of 2,658 differ), and the
prediction changes by up to 0.012 eV/atom. For the same reason the prediction of such a
structure can change with a rotation, a translation or a change of basis of the cell; this
was not measured. `ALIGNN` reads the graph attached to the structures: with the graph of
alignn attached, it predicts on this exact graph.

The graphs follow the configuration of the training of the model. The function of alignn
which predicts a property builds another line graph, with all the bonds up to 5 Å; it was not
reproduced.
