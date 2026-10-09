# CoRE MOF 2024

The public part of the CoRE MOF database updated in 2024 (Zhao et al., 2025), "CoRE MOF SI":
experimental metal-organic frameworks taken from the supporting information of papers
published until the end of 2023. [CoRE MOF 2019](core-mof.md) is another dataset, built from
the earlier version of the database.

| Split | Structures | Atoms | Atoms per structure |
|---|---|---|---|
| `asr`, computation-ready, all solvent removed | 1,371 | 370,621 | 23 to 2,548, median 196 |
| `fsr`, computation-ready, free solvent removed | 1,191 | 305,511 | 23 to 2,548, median 192 |
| `ion`, computation-ready, with their ions | 100 | 28,886 | 64 to 1,376, median 244 |
| `ncr`, not computation-ready | 1,826 | 849,226 | 10 to 11,696, median 240 |

```python
dataset = load_dataset("core-mof-2024")        # asr
dataset = load_dataset("core-mof-2024/ncr")
```

## Labels

Structures come from experiment: the four labels (`total_energy`, `band_gap`, ...) are `NaN`.
The splits `asr`, `fsr` and `ion` have the properties of the tables of the source. `ncr` has
none.

| `properties[...]` | Source column | Unit |
|---|---|---|
| `lcd`, `pld`, `lfpd` | largest cavity, pore limiting and largest free path diameters | Å |
| `density` | `Density (g/cm3)` | g/cm³ |
| `asa_m2_cm3`, `asa_m2_g` | accessible surface area | m²/cm³, m²/g |
| `nasa_m2_cm3`, `nasa_m2_g` | non accessible surface area | m²/cm³, m²/g |
| `void_fraction`, `pv_cm3_g` | `VF`, `PV (cm3/g)` | 0 to 1, cm³/g |
| `heat_capacity_300k` | `Heat_capacity@300K (J/g/K)`, **predicted** by a model | J/(g K) |
| `thermal_stability` | decomposition temperature, **predicted** | °C |
| `solvent_stability`, `water_stability` | probability of stability, **predicted** | 0 to 1 |
| `has_oms` | 1 if the structure has an open metal site | 0 or 1 |
| `unmodified` | 1 if the source did not modify the structure | 0 or 1 |

The three stabilities are `NaN` when the source says `unknown`: 9 structures of `asr`, 26 of
`fsr` and 99 of the 100 of `ion`.

## Things to know

- **`ncr` holds structures the source judged not ready for simulations.** They are there to
  be studied or repaired, not to train on as they are. `ncr_index.parquet` gives for each one the directory of the source which says
  why (`both`, `mofchecker`, `Chen_Manz`, `occupancy`).
- **3,812 structures of the source are missing**: they have sites with a partial occupancy,
  which `Structures` cannot hold. 3,810 of the 5,636 of `ncr`, and one of `asr` and of `fsr`.
  They are listed in the `*_skipped.csv` files.
- **One table row disagrees with its file.** `2018[ZnMo][qtz]3[ION]2` has 345 atoms in its
  file and 369 in the table: its properties may be the ones of another structure.
- **Large cells**, as in CoRE MOF 2019: see [its page](core-mof.md) for the cost of the graphs.
- The partial charges of the files (PACMAN) are not in the dataset.
- The part of the database derived from the Cambridge Structural Database, 32,537
  structures, is distributed by the CCDC and is not here: see
  [Datasets to build yourself](local.md).
- **Identifiers** are not in the file: the `*_index.parquet` files give the identifier of each
  row, its name in the source, its metals, its topology and its DOI.
- **Licence**: CC BY 4.0.

## Filters

**Recommended**: the computation-ready splits (`asr`, `fsr`, `ion`), at most 500 atoms. The
source already removed what it judged unfit from these splits.

```python
from torchms.data import Filter, load_dataset

dataset = load_dataset("core-mof-2024", "asr", filter=Filter(max_atoms=500))
```

| Filter | Keeps, of `asr` / `fsr` / `ion` / `ncr` |
|---|---|
| none | 1,371 / 1,191 / 100 / 1,826 |
| `Filter(max_atoms=500)`, **recommended** | 1,214 / 1,063 / 92 / 1,458 |
| `Filter(max_atoms=1000)` | 1,347 / 1,179 / 98 / 1,674 |
| `Filter(equal={"unmodified": 1})`, as published, not modified by the source | 411 / 613 / 34 / – |
| `Filter(max_atoms=500, known=("thermal_stability",))`, with predicted stabilities | 1,205 / 1,040 / 1 / – |

`ncr` has no property: a filter on one raises an error. Do not train on `ncr` as it is.

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- [CoRE MOF 2024 Dataset](https://doi.org/10.5281/zenodo.15055758), Zenodo, version 1.1:
  `CoREMOF2024DB_SI_20250204.zip` and the three tables `*_data_SI_20250204.csv`, checked with
  their MD5.
- Built by `scripts/build_core_mof_2024.py`. The atoms are read as they are listed in each
  file. Every structure was then compared with the reading of its file by pymatgen: cell,
  number of atoms, and each atom at the position of one atom of the same element. The number
  of atoms and three properties were compared with the tables.
- Repository: [materials-toolkits/core-mof-2024](https://huggingface.co/datasets/materials-toolkits/core-mof-2024),
  files `asr.hdf5`, `fsr.hdf5`, `ion.hdf5` and `ncr.hdf5`.
