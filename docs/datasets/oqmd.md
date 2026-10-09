# OQMD

One structure per entry of the Open Quantum Materials Database, version 1.8 (February 2026).

| | `all` | `unique` |
|---|---|---|
| Structures | 1,407,269 | 1,259,553 |
| Atoms | 14,266,404 | 13,089,068 |
| Stability ≤ 0 | 97,228 | 96,246 |
| Stability ≤ 0.1 eV/atom | 458,906 | 402,102 |

Structures have 1 to 312 atoms (median 6).

```python
dataset = load_dataset("oqmd", "unique")
```

## Selection

The database has several calculations per entry. For each entry, one is selected:

1. its output structure must be consistent: the number of atoms declared for the structure
   equals its number of rows in the table of atoms (47,859 of 8,374,305 structures are not);
2. the calculation is converged, its label starts with `static` or `standard`, and does not end
   with `lda`;
3. among those, the lowest energy per atom is kept (lowest calculation id when equal).

1,407,269 of the 1,407,285 entries have such a calculation.

`unique` groups each entry with the entry it duplicates according to OQMD and keeps the lowest
energy per atom of each group (the original entry when energies are exactly equal). 147,716
entries are removed. Duplicates which OQMD does not mark are not removed.

## Labels

| Field | Source | Note |
|---|---|---|
| `total_energy` | `calculations.energy_pa` × number of atoms | eV |
| `formation_energy_per_atom` | `formation_energies.delta_e` | eV/atom |
| `energy_above_hull` | `formation_energies.stability` | eV/atom, **negative below the hull** |
| `band_gap` | `calculations.band_gap` | eV, `NaN` for 3,155 structures |

## Things to know

- **`energy_above_hull` can be negative.** OQMD's stability is the distance to the hull built
  from the other phases: 97,115 structures of `all` are below zero, down to -2.45 eV/atom.
- **Failed calculations are kept.** In `all`: 559 structures with a positive total energy (up
  to +704 eV/atom), 668 with a stability above 5 eV/atom, one negative band gap.
- **Energies are on the scale of OQMD.** They are consistent with each other, but not with
  the Materials Project or Alexandria for some elements (see
  [Convex hull](../design/hull.md)). Use the labels of this dataset together, not with the
  hull of another one.
- `all_index.parquet` and `unique_index.parquet` give the `entry_id`, `calculation_id` and
  `structure_id` of each row.

## Filters

**Recommended**: remove the failed calculations, a positive total energy or a stability above
5 eV/atom. Use it on `unique`, without the duplicates marked by OQMD.

```python
from torchms.data import Filter, load_dataset

clean = Filter(maximum={"total_energy": 0.0, "energy_above_hull": 5.0})
dataset = load_dataset("oqmd/unique", filter=clean)
```

| Filter | Keeps, of `all` / `unique` |
|---|---|
| none | 1,407,269 / 1,259,553 |
| `Filter(maximum={"total_energy": 0.0, "energy_above_hull": 5.0})`, **recommended** | 1,406,469 / 1,258,843 |
| the same with `minimum={"band_gap": 0.0}`, a band gap known and not negative | 1,403,318 / 1,255,869 |
| `Filter(maximum={"total_energy": 0.0}, max_energy_above_hull=0.1)`, stable and meta-stable | 458,832 / 402,039 |
| the same with `minimum={"energy_above_hull": -0.5}` | 458,675 / 401,887 |
| the same as the fourth with `max_atoms=64` | 454,421 / 397,692 |

- The stability of OQMD is negative for the phases below the hull of the others, which is
  expected: `max_energy_above_hull` keeps them. 157 structures of `all` (152 of `unique`) are
  more than 0.5 eV/atom below it, which is suspicious: the fifth filter drops them.
- `total_energy` is the energy of the cell: `maximum={"total_energy": 0.0}` drops the
  structures with a positive energy per atom whatever their size.

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- SQL dump `qmdb__v1_8__022026.sql.gz` (21 GB), read as a stream without MySQL server.
- Built by `scripts/build_oqmd.py`. Every structure was compared with the tables of the dump.
- 1,007,099 entries are also in a 2024 export made independently from an earlier version of
  the database; 1,005,090 of them have the same selected structure.
- Repository: [materials-toolkits/oqmd](https://huggingface.co/datasets/materials-toolkits/oqmd)
