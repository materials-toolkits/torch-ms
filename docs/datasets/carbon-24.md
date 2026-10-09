# Carbon-24

Carbon structures with 6 to 24 atoms per cell, as released with CDVAE. The original data comes
from the AIRSS carbon data of C. J. Pickard (2020).

| Split | Structures |
|---|---|
| `train` | 6,091 |
| `val` | 2,032 |
| `test` | 2,030 |
| total | 10,153 |

```python
dataset = load_dataset("carbon-24", "train")
```

## Labels

| Field | Source column | Note |
|---|---|---|
| `total_energy` | `energy_per_atom` × number of atoms | eV |
| others | – | `NaN` |

The total energy is stored in 32 bits: the energy per atom recovered from it is accurate to
about 1e-5 eV/atom.

!!! warning "Energy scale"
    Energies are around -154 eV/atom. Carbon is at -9.2 eV/atom in LeMat-Bulk: the two
    are computed differently and cannot be compared, and the energy estimator of
    [Energy above the hull](../guide/energy.md) is not on the scale of this dataset.

## Filters

**No filter is needed**: every structure has its total energy, 6 to 24 atoms, and only carbon.

```python
from torchms.data import load_dataset

dataset = load_dataset("carbon-24", "train")               # recommended: everything
```

A filter on the number of atoms, `Filter(max_atoms=12)` for instance, is the only one which
selects something here: the three other labels are `NaN` for every structure.

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- `https://github.com/txie-93/cdvae/tree/main/data/carbon_24`
- Built by `scripts/build_cdvae.py`, with the same checks as [Perov-5](perov-5.md).
- Repository: [materials-toolkits/carbon-24](https://huggingface.co/datasets/materials-toolkits/carbon-24)
