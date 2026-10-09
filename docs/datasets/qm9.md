# QM9

The 133,885 organic molecules of QM9 (Ramakrishnan et al., 2014): up to nine heavy atoms among
C, N, O and F, with their hydrogens, relaxed with B3LYP/6-31G(2df,p). They are **molecules, not
crystals**: `pbc` is `False` and `x` holds positions in Å, see
[Periodicity](../design/periodicity.md).

| | |
|---|---|
| Molecules | 133,885 |
| Atoms | 2,407,753, 3 to 29 per molecule (18 on average) |
| Elements | H, C, N, O, F |
| Flagged `uncharacterized` | 3,054 |

```{.python notest}
dataset = load_dataset("qm9")
```

## Labels

Energies are converted from Hartree to eV (1 Ha = 27.211386245988 eV). Everything else keeps
the unit of the source.

| Field | Source column | Unit |
|---|---|---|
| `total_energy` | U0, internal energy at 0 K | eV |
| `band_gap` | gap, LUMO − HOMO | eV |

`formation_energy_per_atom` and `energy_above_hull` are `NaN`.

| `properties[...]` | Source column | Unit |
|---|---|---|
| `rotational_a`, `rotational_b`, `rotational_c` | A, B, C | GHz |
| `dipole_moment` | mu | Debye |
| `polarizability` | alpha | bohr³ |
| `homo`, `lumo` | HOMO, LUMO | eV |
| `r2` | electronic spatial extent | bohr² |
| `zpve` | zero point vibrational energy | eV |
| `u`, `h`, `g` | U, H, G at 298.15 K | eV |
| `cv` | heat capacity at 298.15 K | cal/(mol K) |
| `u0_atomization`, `u_atomization`, `h_atomization`, `g_atomization` | energy minus the ones of its atoms (`atomref.txt`) | eV |
| `uncharacterized` | 1 if listed in `uncharacterized.txt` | 0 or 1 |

## Things to know

- **Use the atomization energies to train.** Labels are stored in 32 bits. A total energy of
  −19,000 eV is then known to 1 meV only, which is not far from the error of the best models.
  The atomization energies, at most 115 eV, were computed in 64 bits and are stored to
  0.01 meV.
- **Nothing is filtered.** The 3,054 molecules whose geometry failed the consistency check of
  the source are kept, with `uncharacterized` set to 1. Most papers remove them:
  `load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))` keeps the 130,831
  others.
- **No split.** The source has none; papers draw their own, as
  `random_split(dataset, [0.8, 0.1, 0.1], seed=0)`.
- **`band_gap` of a molecule** is its HOMO-LUMO gap, which is not the band gap of a crystal:
  do not compare it with the one of the crystal datasets.
- **Identifiers and SMILES** are not in the file. `all_index.parquet` gives for each row the
  index of the molecule in QM9 and its SMILES and InChI, from GDB-9 and from the relaxed
  geometry.
- **Licence**: CC0.

## Filters

**Recommended**: without the 3,054 molecules which failed the consistency check of the source,
as most papers do.

```{.python notest}
from torchms.data import Filter, load_dataset

dataset = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
```

| Filter | Keeps, of 133,885 |
|---|---|
| `Filter(equal={"uncharacterized": 0})`, **recommended** | 130,831 |
| `Filter(elements=(1, 6, 7, 8))`, without fluorine | 131,722 |

Every molecule has every label: a filter on missing values removes nothing.

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- [Quantum chemistry structures and properties of 134 kilo molecules](https://doi.org/10.6084/m9.figshare.c.978904.v5),
  figshare: `dsgdb9nsd.xyz.tar.bz2`, `atomref.txt`, `uncharacterized.txt`, checked with their
  MD5.
- Built by `scripts/build_qm9.py`. Every molecule was compared with its file, read a second
  time by other code: number of atoms, elements, positions, U0, gap, HOMO and the atomization
  energy.
- Repository: [materials-toolkits/qm9](https://huggingface.co/datasets/materials-toolkits/qm9),
  file `all.hdf5`.
