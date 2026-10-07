# PyCrystalGen

PyCrystalGen is a Python library for machine learning on crystal structures. It provides:

- **Datasets** in one common format, published on Hugging Face and loaded by name.
- **`Structures`**, a batched container of crystals for PyTorch.
- **Graphs** of crystals with periodic boundary conditions.
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
| build neighbour graphs | [Graphs](guide/graphs.md) |
| know if a crystal is stable | [Energy above the hull](guide/energy.md) |
| convert your own CIF files | [Preprocessing your own data](guide/preprocessing.md) |
| know what is in a dataset and what to watch for | [Datasets](datasets/index.md) |
| understand a design choice | [Design notes](design/batching.md) |

## Status

The library is under development. The datasets and the energy estimator described here are
built and checked against their sources; the pages of each dataset list what was verified and
the known issues of the source data.
