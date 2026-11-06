# Alex-MP-20

The training set of [MatterGen](../models/mattergen.md), as its authors release it:
structures of Alexandria and of MP-20 with at most 20 atoms per cell and at most 0.1 eV/atom
above the hull, relaxed again with PBE so that their energies are consistent. MatterGen
excluded the structures with Tc, Pm or an element after Bi (atomic number 84 or more).

| Split | Structures | Alexandria | Materials Project |
|---|---|---|---|
| `train` | 607,683 | 568,103 | 39,580 |
| `val` | 67,521 | 63,115 | 4,406 |
| total | 675,204 | 631,218 | 43,986 |

MatterGen releases no test split. Two structures of `train` have identifiers `mvc-`, of the
Materials Project too. Five structures of Alexandria in `train` are given as a larger cell
than their primitive one, of 27 atoms (four) and 60 atoms (one): the dataset has 1 to 20
atoms per structure, but for them.

```{.python notest}
dataset = load_dataset("alex-mp-20", "train")
```

The weights `mattergen-base` of [MatterGen](../models/mattergen.md) were trained on it, and
its fine-tuned weights on its properties below.

## Labels

| Field | Source column |
|---|---|
| `energy_above_hull` | `energy_above_hull`, from 0 to 0.1 eV/atom |
| `band_gap` | `dft_band_gap`, known for 38,950 / 4,345 structures, `NaN` for the others |
| `total_energy` | – (`NaN`) |
| `formation_energy_per_atom` | – (`NaN`) |

## Properties

In `structures.properties`, `NaN` when the source has none:

| Name | Source column | Known for, of `train` / `val` |
|---|---|---|
| `space_group` | `space_group`, its symbol stored as its number | every structure |
| `dft_bulk_modulus` | `dft_bulk_modulus`, in GPa | 4,976 / 550 |
| `ml_bulk_modulus` | `ml_bulk_modulus`, in GPa, predicted by a machine-learned potential | every structure |
| `dft_mag_density` | `dft_mag_density`, magnetic density in Å⁻³ | 597,339 / 66,362 |
| `hhi_score` | `hhi_score`, Herfindahl–Hirschman index of the supply of the elements | every structure |

These are the conditions of the fine-tuned models of MatterGen. The identifier, the formula
and the chemical system of the source are not stored. Its column `num_sites` is not the
number of atoms of the CIF for 334 structures of `train` and 43 of `val`: larger by 1 to 15
for 369 structures of the Materials Project, smaller for 8 structures of Alexandria, the five
large cells among them. The structures are the ones of the CIF, which MatterGen reads.

## Filters

**No filter is needed**: the dataset was already selected (at most 20 atoms, at most
0.1 eV/atom above the hull) and has no noble gas.

| Filter | Keeps, of `train` / `val` |
|---|---|
| none | 607,683 / 67,521 |
| `Filter(max_energy_above_hull=0.0)`, on the hull | 48,917 / 5,415 |

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- `alex_mp_20.zip` of `data-release/alex-mp` in the repository of MatterGen, commit
  `9242366` (a file of git LFS, 186 MB, checked against its SHA-256). Alexandria is published
  under CC BY 4.0 (Schmidt et al., Materials Cloud Archive 2022.126); MatterGen is MIT.
- Built by `scripts/build_alex_mp_20.py`, with the checks of [Perov-5](perov-5.md): every
  label against its column, and 300 structures per split against pymatgen; and every
  property against its column, and the number of atoms of each structure against its CIF.
- The file `ref.csv` of the archive, the energies of the 968,035 structures of the convex
  hull MatterGen evaluates against, has no structure: it is not included.
- Repository: [materials-toolkits/alex-mp-20](https://huggingface.co/datasets/materials-toolkits/alex-mp-20)
