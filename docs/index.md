# Torch Material Science (TorchMS)

TorchMS (Torch Material Science) is a Python library for machine learning on atomic structures:
crystals, metal-organic frameworks and molecules. It provides:

- **Datasets** in one common format, published on Hugging Face and loaded by name: inorganic
  crystals, metal-organic frameworks and organic molecules. They are filtered, split and mixed
  when they are loaded.
- **`Structures`**, a batched container for PyTorch of crystals, of molecules, or of both in
  the same batch.
- **Data augmentation**: other descriptions of the same structures, to train on.
- **Graphs** of crystals with periodic boundary conditions and of molecules, with their angles
  and dihedral angles.
- **Models**: rewrites of pretrained ones (orb, MACE, M3GNet, CHGNet, ALIGNN, CDVAE), models
  of the energy to train, and the layers they share.
- **Energy above the convex hull**, from DFT energies or estimated with a pretrained model.

```{.python notest}
from pathlib import Path

from torchms.data import parse_cif
from torchms.models.estimator import EnergyEstimator

# every data block of the file is a structure of the batch
structures = parse_cif(Path("structures.cif").read_text())

estimator = EnergyEstimator.from_lematerial(device="cuda")
energy, above_hull = estimator.predict_energy_above_hull(structures)

for i, (e, above) in enumerate(zip(energy, above_hull)):
    n = int(structures.num_atoms[i])
    print(f"{n:3d} atoms  {e / n:8.4f} eV/atom  {above:8.4f} eV/atom above hull")
```

## Where to go

| If you want to | Read |
|---|---|
| install the library | [Installation](guide/installation.md) |
| load a dataset | [Loading datasets](guide/datasets.md) |
| understand the batch format | [Structures](guide/structures.md) |
| keep some structures of a dataset | [Filtering a dataset](guide/datasets.md#filtering-a-dataset) |
| make a training, a validation and a test set | [Splitting a dataset](guide/datasets.md#splitting-a-dataset) |
| train on crystals and molecules together | [Several datasets](guide/datasets.md#several-datasets-crystals-and-molecules-together) |
| balance datasets of different sizes | [A fixed share of each dataset](guide/datasets.md#a-fixed-share-of-each-dataset-in-every-batch) |
| augment the structures of a dataset | [Data augmentation](guide/augmentation.md) |
| build neighbour graphs | [Graphs](guide/graphs.md) |
| run a pretrained model, or build one with angles or products | [Using models](guide/models.md) |
| know what a model is and how it was checked | [Models](models/index.md) |
| know if a crystal is stable | [Energy above the hull](guide/energy.md) |
| convert your own CIF files | [Preprocessing your own data](guide/preprocessing.md) |
| know what is in a dataset and what to watch for | [Datasets](datasets/index.md) |
| understand a design choice | [Design notes](design/batching.md) |

## Status

The library is under development. The datasets and the energy estimator described here are
built and checked against their sources; the pages of each dataset list what was verified and
the known issues of the source data.
