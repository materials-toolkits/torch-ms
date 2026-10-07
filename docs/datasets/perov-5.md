# Perov-5

Perovskites with 5 atoms per cell, as released with CDVAE. The original data comes from
Castelli et al. (2012).

| Split | Structures |
|---|---|
| `train` | 11,356 |
| `val` | 3,787 |
| `test` | 3,785 |
| total | 18,928 |

```python
dataset = load_dataset("perov-5", "train")
```

## Labels

| Field | Source column | Note |
|---|---|---|
| `formation_energy_per_atom` | `heat_ref` | heat of formation, copied unchanged |
| `band_gap` | `ind_gap` | indirect band gap, copied unchanged |
| `total_energy`, `energy_above_hull` | – | `NaN` |

The source has two heats of formation (`heat_all`, `heat_ref`) and two gaps (`dir_gap`,
`ind_gap`). `heat_ref` is the one used by CDVAE; the indirect gap is the smallest one.
`heat_all` and `dir_gap` are not stored.

!!! note "Units"
    Values are copied from the source without conversion. They are consistent with eV/atom
    and eV, but this was not checked against the original publication.

## Source and checks

- `https://github.com/txie-93/cdvae/tree/main/data/perov_5`
- Built by `scripts/build_cdvae.py`. Same number of rows as the source, labels equal to their
  column row by row, 300 structures per split compared with pymatgen.
- Repository: [materials-toolkits/perov-5](https://huggingface.co/datasets/materials-toolkits/perov-5)
