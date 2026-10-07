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

See [Loading datasets](../guide/datasets.md) to load them.

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

- A label without value in the source is `NaN`.
- The lattice is rebuilt from the cell lengths and angles: it is in the standard orientation
  (first vector along x), not the one of the source.
- Fractional coordinates are wrapped into the cell. Because of the rounding to 32 bits, a
  coordinate can be exactly 1.0.
- The `batch` array is an artefact of the writer and is not used when reading.

## Things to know before mixing datasets

- **Labels with the same name are not comparable across datasets.** Total energies depend on
  the settings of the calculation: see [Convex hull](../design/hull.md).
- **Identifiers are not in the files.** Row *i* of a file is row *i* of its source; OQMD and
  LeMat-Bulk come with an `*_index.parquet` giving the identifier of each row.
- **Every structure was compared with its source** when the files were built: number of atoms,
  elements, coordinates, lattice and labels. The smaller datasets were also compared with the
  reading of pymatgen on a sample.
