# PyCrystalGen

PyCrystalGen is a Python library for machine learning on crystal structures. It provides:

- **Datasets** in one common format, published on Hugging Face and loaded by name.
- **`Structures`**, a batched container of crystals for PyTorch.
- **Data augmentation**: other descriptions of the same crystals, to train on.
- **Graphs** of crystals with periodic boundary conditions, with their angles and dihedral
  angles.
- **Models** of the energy with terms on three atoms and more, and rewrites of pretrained
  ones (orb, CHGNet, M3GNet, ALIGNN, MACE).
- **Energy above the convex hull**, from DFT energies or estimated with a pretrained model.

```python
from pycrystalgen.data import load_dataset
from pycrystalgen.estimator import EnergyEstimator

dataset = load_dataset("mp-20", "test")
structures = dataset[:32]  # a batch of 32 crystals

estimator = EnergyEstimator.from_lematerial(device="cuda")
energy, above_hull = estimator.predict_energy_above_hull(dataset)
```

## Where to go

| If you want to | Read |
|---|---|
| install the library | [Installation](guide/installation.md) |
| load a dataset | [Loading datasets](guide/datasets.md) |
| understand the batch format | [Structures](guide/structures.md) |
| augment the structures of a dataset | [Data augmentation](guide/augmentation.md) |
| build neighbour graphs | [Graphs](guide/graphs.md) |
| build a model with angles or products | [Models](guide/models.md) |
| know if a crystal is stable | [Energy above the hull](guide/energy.md) |
| convert your own CIF files | [Preprocessing your own data](guide/preprocessing.md) |
| know what is in a dataset and what to watch for | [Datasets](datasets/index.md) |
| understand a design choice | [Design notes](design/batching.md) |

## Status

The library is under development. The datasets and the energy estimator described here are
built and checked against their sources; the pages of each dataset list what was verified and
the known issues of the source data.
