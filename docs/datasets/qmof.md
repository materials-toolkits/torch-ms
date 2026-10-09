# QMOF

The QMOF Database (Rosen et al., 2021 and 2022), version 18: metal-organic frameworks and
coordination polymers relaxed with PBE-D3(BJ). It is the database of the
[MOF Explorer](https://materialsproject.org/mofs) of the Materials Project.

| | |
|---|---|
| Structures | 20,372 |
| Atoms | 2,312,743, 17 to 500 per structure (114 on average, median 96) |
| Elements | 79, from H to Pu |
| With a band gap at HLE17 / HSE06* / HSE06 | 10,756 / 10,811 / 10,810 |

```python
dataset = load_dataset("qmof")
```

## Labels

| Field | Source | Unit |
|---|---|---|
| `total_energy` | `outputs.pbe.energy_total` | eV, of the cell |
| `band_gap` | `outputs.pbe.bandgap` | eV |

`formation_energy_per_atom` and `energy_above_hull` are `NaN`.

| `properties[...]` | Source | Unit |
|---|---|---|
| `band_gap_hle17`, `band_gap_hse06_10hf`, `band_gap_hse06` | `outputs.<theory>.bandgap` | eV, `NaN` when not computed |
| `energy_vdw` | `outputs.pbe.energy_vdw`, the D3(BJ) part of the energy | eV |
| `net_magmom` | `outputs.pbe.net_magmom` | μB |
| `pld`, `lcd` | pore-limiting and largest cavity diameters (Zeo++) | Å |
| `density` | `info.density` | g/cm³ |
| `synthesized` | 1 if the structure comes from an experimental database | 0 or 1 |

## Things to know

- **Not on the scale of the crystal datasets.** `total_energy` is a raw PBE-D3(BJ) energy with
  the settings of QMOF. It cannot be compared with the energies of the Materials Project or of
  LeMat-Bulk, and there is no convex hull.
- **Large cells.** One structure in ten has more than 200 atoms. The radius graph lists the
  pairs of atoms of each structure: use smaller batches than for MP-20.
- **Experimental and hypothetical.** `synthesized` is 0 for the hypothetical structures (from
  ToBaCCo, Boyd & Woo, and others).
- **Identifiers** are not in the file. `all_index.parquet` gives for each row its `qmof_id`,
  its name in the source database, its formula, its DOI and its MOFid.
- The partial charges, bond orders and magnetic moments per atom of the source are not in
  the file.
- No structure with partial occupancy was found in the database.
- **Licence**: CC BY 4.0. Cite the two papers of the database.

## Source and checks

- [QMOF Database](https://doi.org/10.6084/m9.figshare.13147324.v18), figshare, version 18:
  `qmof.json` and `qmof_structure_data.json` of `qmof_database.zip`, checked with its MD5.
- Built by `scripts/build_qmof.py`. Every structure was compared with the database, read by
  pymatgen: number of atoms, elements, coordinates, lattice, total energy, the four band gaps
  and the largest cavity diameter.
- Repository: [materials-toolkits/qmof](https://huggingface.co/datasets/materials-toolkits/qmof),
  file `all.hdf5`.
