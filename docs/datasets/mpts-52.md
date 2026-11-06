# MPTS-52

Experimental structures of the Materials Project with at most 52 atoms per cell, split by
the year of their first publication: the time split of Baird et al.
([mp-time-split](https://github.com/sparks-baird/mp-time-split)), as released with DiffCSP,
which made it a benchmark of the prediction of a structure. The models are trained on older
crystals and tested on newer ones.

| Split | Structures | Years |
|---|---|---|
| `train` | 27,380 | 1913–1997 |
| `val` | 5,000 | 1997–2004 |
| `test` | 8,096 | 2004–2013 |
| total | 40,476 | |

```{.python notest}
dataset = load_dataset("mpts-52", "train")
```

The weights `crystalflow-csp-mpts52` of [CrystalFlow](../models/crystalflow.md) and
`diffcsp-pp-mpts52` of [DiffCSP++](../models/diffcsp_pp.md) were trained on it.

## Labels

| Field | Source column |
|---|---|
| `formation_energy_per_atom` | `formation_energy_per_atom` |
| `energy_above_hull` | `energy_above_hull` |
| `total_energy` | – (`NaN`) |
| `band_gap` | – (`NaN`) |

The energy above the hull goes up to 5.2 eV/atom: unlike MP-20, the dataset is not selected
on stability. The year, the identifier of the Materials Project and the flag `theoretical`
(false for every structure) of the source are not stored.

## Filters

**No filter is needed**: every structure has its two labels, and filtering changes the splits
of the benchmark.

| Filter | Keeps, of `train` / `val` / `test` |
|---|---|
| none | 27,380 / 5,000 / 8,096 |
| `Filter(max_energy_above_hull=0.0)`, on the hull | 14,250 / 2,602 / 4,120 |
| `Filter(max_atoms=20)`, the size of MP-20 | 17,594 / 2,282 / 3,518 |
| `Filter(exclude_elements=(2, 10, 18, 36, 54, 86))`, without noble gases | 27,342 / 4,984 / 8,074 |

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- `https://github.com/jiaor17/DiffCSP/tree/7121d15/data/mpts_52`, the files DiffCSP++
  ships too, byte for byte
- Built by `scripts/build_cdvae.py --only mpts_52`, with the same checks as
  [Perov-5](perov-5.md): every label against its column, and 300 structures per split against
  pymatgen.
- Repository: [materials-toolkits/mpts-52](https://huggingface.co/datasets/materials-toolkits/mpts-52)
