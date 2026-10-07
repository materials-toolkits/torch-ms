# LeMat-Bulk

The `compatible_pbe` subset of [LeMat-Bulk](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk):
structures of Alexandria (4,628,422), OQMD (567,945) and the Materials Project (138,931)
computed with the PBE functional.

| | `all` | `unique` |
|---|---|---|
| Structures | 5,335,299 | 4,927,038 |
| Atoms | 49,384,294 | 46,019,886 |
| Energy above hull ≤ 0 | 109,113 | 105,636 |
| Energy above hull ≤ 0.1 eV/atom | 1,640,789 | 1,547,665 |

Structures have 1 to 444 atoms (median 6).

```python
dataset = load_dataset("lematerial", "unique")
```

`unique` keeps, among the structures with the same `entalpic_fingerprint`, the one with the
lowest energy per atom. The 46 structures without fingerprint are kept. 408,261 structures are
removed.

## Labels

| Field | Source | Note |
|---|---|---|
| `total_energy` | `energy` of LeMat-Bulk | eV |
| `energy_above_hull` | `dft_hull` of LeMat-Bulk-DFT-Hull-All | eV/atom, see below |
| `formation_energy_per_atom`, `band_gap` | – | `NaN` |

34 structures of `all` (31 of `unique`) have no published energy above the hull; theirs was
computed from the total energies of the dataset.

## Things to know

!!! warning "The stored energy above the hull mixes sources which are not comparable"
    The total energies of the structures from OQMD are shifted compared with the two other
    sources for nine elements (Gd, Cu, Ni, Pu, Bi, Pa, Fe, Np, Re). The published energy above
    the hull is computed with the three sources together. It is not reliable for the
    structures from OQMD with one of these elements, and it differs from the hull of the two
    other sources by more than 0.001 eV/atom for 4.9% of their structures.

    Use [`hull.hdf5`](../design/hull.md), built without OQMD, to select stable structures.

- **Failed calculations are kept.** 152 structures have a positive energy per atom (up to
  +648 eV/atom), all from OQMD.
- **One corrupt entry.** `oqmd-2969647` declares 8 sites and lists 13 atoms, with copper and
  magnesium at the same positions. It is kept; the energy estimator gives no value for it.
- `all_index.parquet` and `unique_index.parquet` give the `immutable_id` of each row. One
  structure has no identifier.

## Other files of the repository

| File | Content |
|---|---|
| `hull.hdf5`, `hull_index.parquet` | [convex hull](../design/hull.md) of the structures of the Materials Project and Alexandria |
| `energy_correction_orb.hdf5` | correction of the [energy estimator](../design/estimator.md) |
| `energy_validation.json`, `energy_splits.parquet` | validation of the estimator |
| `lematerial*.cif` | an older, curated selection in CIF, unrelated to the HDF5 files |

## Source and checks

- LeMat-Bulk revision `0dc17eea`, LeMat-Bulk-DFT-Hull-All revision `fc063a96`.
- Built by `scripts/build_lematerial.py`. Every structure was compared with its row of
  LeMat-Bulk.
- Repository: [materials-toolkits/lematerial](https://huggingface.co/datasets/materials-toolkits/lematerial)
