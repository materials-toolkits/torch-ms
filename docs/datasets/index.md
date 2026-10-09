# Datasets

All datasets share one format and are **complete and unfiltered**: nothing of the source is
removed beyond what each page states. Failed calculations and duplicates of the sources are
kept, so that you choose your own filters.

| Name | Source | Structures | Atoms per structure | Labels |
|---|---|---|---|---|
| [`perov-5`](perov-5.md) | CDVAE | 18,928 | 5 | formation energy, band gap |
| [`carbon-24`](carbon-24.md) | CDVAE | 10,153 | 6–24 | total energy |
| [`mp-20`](mp-20.md) | CDVAE | 45,229 | 1–20 | formation energy, energy above hull, band gap |
| [`materials-project`](materials-project.md) | Materials Project 2026.09.28 | 271,325 | 1–444 | all four |
| [`oqmd`](oqmd.md) | OQMD v1.8 | 1,407,269 | 1–312 | all four |
| [`lematerial`](lematerial.md) | LeMat-Bulk `compatible_pbe` | 5,335,299 | 1–444 | total energy, energy above hull |
| [`qmof`](qmof.md) | QMOF Database v18 | 20,372 | 17–500 | total energy, band gap, 9 properties |
| [`core-mof`](core-mof.md) | CoRE MOF 2019 | 12,020 / 7,061 | 10–10,560 | 13 properties, no energy |
| [`core-mof-2024`](core-mof-2024.md) | CoRE MOF 2024, public part | 1,371 / 1,191 / 100 / 1,826 | 10–11,696 | 16 properties, no energy |
| [`qm9`](qm9.md) | QM9 | 133,885 | 3–29 | total energy, band gap, 17 properties |
| [`zinc250k`](zinc250k.md) | ZINC250k, positions from RDKit | 249,427 | 9–83 | 4 properties, no energy |

The first six are inorganic crystals. `qmof`, `core-mof` and `core-mof-2024` hold metal-organic
frameworks, computed and experimental. `qm9` and `zinc250k` hold organic molecules, which are not periodic.

ZINC20 in 3D and the MOFs of the Cambridge Structural Database are not published as files:
see [Datasets to build yourself](local.md).

See [Loading datasets](../guide/datasets.md) to load them.

## Recommended filters

Nothing is removed from the files. A good default for each dataset, with what it keeps; the
page of each dataset has the others and what they keep.

```{.python notest}
from torchms.data import Filter, load_dataset

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
```

| Dataset | Recommended filter | Keeps | Why |
|---|---|---|---|
| `perov-5`, `carbon-24`, `mp-20` | none | everything | already selected, every label known |
| `materials-project` | `Filter(max_energy_above_hull=0.1)` | 216,917 of 271,325 | stable and meta-stable |
| `oqmd/unique` | `Filter(maximum={"total_energy": 0.0, "energy_above_hull": 5.0})` | 1,258,843 of 1,259,553 | failed calculations |
| `lematerial/unique` | `Filter(maximum={"total_energy": 0.0})` | 4,926,943 of 4,927,038 | failed calculations |
| `qmof` | `Filter(max_atoms=200)` | 18,352 of 20,372 | size of the batches |
| `core-mof` | `Filter(equal={"disorder": 0}, max_atoms=500)` | 8,722 of 12,020 (`asr`) | disordered structures, size |
| `core-mof-2024` | `Filter(max_atoms=500)` | 1,214 of 1,371 (`asr`) | size |
| `qm9` | `Filter(equal={"uncharacterized": 0})` | 130,831 of 133,885 | failed consistency check |
| `zinc250k` | `Filter(equal={"mmff_converged": 1})` | 249,425 of 249,427 | relaxation not converged |

These are defaults to start from, not rules: they are what the known problems of each source
ask for. See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for the settings.

## Format

Each file stores flat concatenated arrays:

| Dataset | Shape | Type | Content |
|---|---|---|---|
| `lattice` | (n_struct, 3, 3) | float32 | lattice vectors as rows, in Å |
| `num_atoms` | (n_struct,) | int64 | number of atoms of each structure |
| `ptr` | (n_struct,) | int64 | index of the first atom of each structure in `x` and `z` |
| `x` | (n_atoms, 3) | float32 | fractional coordinates |
| `z` | (n_atoms,) | int64 | atomic numbers |
| `total_energy` | (n_struct,) | float32 | total energy of the cell, eV |
| `formation_energy_per_atom` | (n_struct,) | float32 | eV/atom |
| `energy_above_hull` | (n_struct,) | float32 | eV/atom |
| `band_gap` | (n_struct,) | float32 | eV |
| `pbc` | (n_struct, 3) | bool | whether the structure is periodic along each lattice vector |
| `properties/<name>` | (n_struct,) | float32 | the other labels of the dataset, by name |

- A label without value in the source is `NaN`.
- A file without `pbc` holds crystals only: this is the case of the six datasets of crystals.
- A molecule has the identity as `lattice` and its positions in Å as `x`, see
  [Periodicity](../design/periodicity.md).
- The properties are read in `Structures.properties`. Their names and units are on the page
  of each dataset.
- The lattice is rebuilt from the cell lengths and angles: it is in the standard orientation
  (first vector along x), not the one of the source.
- Fractional coordinates are wrapped into the cell. Because of the rounding to 32 bits, a
  coordinate can be exactly 1.0.
- The `batch` array is an artefact of the writer and is not used when reading.

## Things to know before mixing datasets

- **Labels with the same name are not comparable across datasets.** Total energies depend on
  the settings of the calculation: see [Convex hull](../design/hull.md). The `band_gap` of a
  molecule is its HOMO-LUMO gap.
- **A batch has the properties of all its datasets**, `NaN` for the structures of a dataset
  without them.
- **Licences differ.** Each page gives the one of its dataset.
- **Identifiers are not in the files.** Row *i* of a file is row *i* of its source; OQMD,
  LeMat-Bulk and the datasets of MOFs and of molecules come with an `*_index.parquet` giving the identifier of each row.
- **Every structure was compared with its source** when the files were built: number of atoms,
  elements, coordinates, lattice and labels. The smaller datasets were also compared with the
  reading of pymatgen on a sample.
