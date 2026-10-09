# Materials Project

Every material of the `summary` collection of the Materials Project, database version
2026.09.28, except the ones flagged as deprecated.

| | |
|---|---|
| Structures | 271,325 (280,159 in the collection, 8,834 deprecated) |
| Atoms | 6,967,262, 1 to 444 per structure |
| Stable (on the hull) | 141,128 |
| Within 0.1 eV/atom of the hull | 216,917 |

```python
dataset = load_dataset("materials-project")
```

## Labels

| Field | Source column | Note |
|---|---|---|
| `total_energy` | `energy_per_atom` × `nsites` | eV, **with the corrections** of the Materials Project |
| `formation_energy_per_atom` | `formation_energy_per_atom` | eV/atom |
| `energy_above_hull` | `energy_above_hull` | eV/atom |
| `band_gap` | `band_gap` | eV |

Eight materials miss labels in the source and are kept with `NaN`: each of the four labels is
missing for four of them.

## Things to know

- **Corrected energies.** The Materials Project applies empirical corrections and mixes
  functionals to build its formation energies and its hull. `total_energy` is this corrected
  energy, not the raw DFT energy, and is not on the scale of LeMat-Bulk.
- **Licence.** Each material has its own licence in the source: 154,377 of the structures are
  labelled `BY-C` and 116,948 `BY-NC` (non commercial). The file does not record which is
  which: treat it as non commercial unless you check against the source.
- **Identifiers** are the new ones of the Materials Project (`mp-aaahgjou`), and are not in
  the file. Structures are in the order of the collection, deprecated materials removed.
- No structure with partial occupancy was found in the collection.

## Filters

**Recommended**: the stable and meta-stable structures, at most 0.1 eV/atom above the hull. It
also drops the four structures without energy above the hull.

```python
from torchms.data import Filter, load_dataset

dataset = load_dataset("materials-project", filter=Filter(max_energy_above_hull=0.1))
```

| Filter | Keeps, of 271,325 |
|---|---|
| `Filter(max_energy_above_hull=0.1)`, **recommended** | 216,917 |
| `Filter(max_energy_above_hull=0.0)`, on the hull | 141,128 |
| `Filter(max_energy_above_hull=0.1, max_atoms=64)`, and cells which batch well | 203,780 |
| `Filter(max_energy_above_hull=0.1, max_atoms=20)`, the selection of MP-20 on today's database | 123,240 |
| `Filter(known=("total_energy", "formation_energy_per_atom", "energy_above_hull", "band_gap"))`, every label known | 271,317 |
| `Filter(exclude_elements=(2, 10, 18, 36, 54, 86))`, without noble gases | 271,141 |

No structure has a positive total energy: there is no failed calculation to remove, unlike in
OQMD and LeMat-Bulk.

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- Open data bucket of the Materials Project,
  `materialsproject-build/collections/summary/version=2026-09-28/`, no API key needed.
- Built by `scripts/build_materials_project.py`. Every structure was compared with the
  collection: number of atoms, elements, coordinates, lattice and the four labels.
- Repository: [materials-toolkits/materials-project](https://huggingface.co/datasets/materials-toolkits/materials-project),
  file `all.hdf5`. The repository also holds an older export (2019) in another layout.
