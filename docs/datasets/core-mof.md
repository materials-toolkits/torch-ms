# CoRE MOF 2019

The public structures of the Computation-Ready Experimental MOF database (Chung et al., 2019),
version 1.1.4: metal-organic frameworks **determined by experiment**, cleaned to be used in
simulations. Their solvent is removed in two ways, which make the two splits:

| Split | Structures | Atoms | Atoms per structure |
|---|---|---|---|
| `asr`, all solvent removed | 12,020 | 3,926,702 | 10 to 10,560, median 192 |
| `fsr`, free solvent removed | 7,061 | 2,388,025 | 10 to 10,560, median 204 |

In `fsr` the solvent bound to the metals is kept. Most structures of `fsr` are also in `asr`,
with the same name up to its suffix.

```python
dataset = load_dataset("core-mof")        # asr
dataset = load_dataset("core-mof/fsr")
```

## Labels

Structures come from experiment: the four labels (`total_energy`, `band_gap`, ...) are `NaN`.
The properties are the geometric ones of the table of the source, computed with Zeo++:

| `properties[...]` | Source column | Unit |
|---|---|---|
| `lcd`, `pld`, `lfpd` | largest cavity, pore limiting and largest free path diameters | Å |
| `density` | `cm3_g`, which holds the density | g/cm³ |
| `asa_m2_cm3`, `asa_m2_g` | accessible surface area | m²/cm³, m²/g |
| `nasa_m2_cm3`, `nasa_m2_g` | non accessible surface area | m²/cm³, m²/g |
| `void_fraction` | `AV_VF` | 0 to 1 |
| `av_cm3_g`, `nav_cm3_g` | accessible and non accessible volume | cm³/g |
| `has_oms` | 1 if the structure has an open metal site | 0 or 1 |
| `disorder` | 1 if the source put the structure in its `disorder` folder | 0 or 1 |

## Things to know

- **Very large cells.** One structure in ten has more than 690 atoms, 559 of `asr` have more
  than 1,000, and the largest has 10,560. The radius graph lists the pairs of atoms of a
  structure: at 5 Å a cell of 2,000 atoms takes 0.5 GB on a GPU, one of 5,000 atoms 3.3 GB.
  Load it with `filter=Filter(max_atoms=500)` or use batches of a few structures.
- **Nothing is filtered.** The 1,877 structures of `asr` and the 1,053 of `fsr` that the
  source set apart as disordered are kept, with `disorder` set to 1.
  `filter=Filter(equal={"disorder": 0})` keeps the others.
- **Experimental structures are not relaxed**, and hydrogens can be missing.
- **Atoms are in the order of the files.** 8 files of `asr` and 7 of `fsr` are not in P1:
  their symmetry operations were applied by pymatgen.
- **Names** are not in the file. `asr_index.parquet` and `fsr_index.parquet` give for each
  row the name of the structure in the source (a CSD refcode or the name of a file of
  supporting information, with a suffix), its metals and its DOI.
- No structure with partial occupancy was found.
- **Licence**: CC BY 4.0.

## Source and checks

- [CoRE MOF 2019 Dataset](https://doi.org/10.5281/zenodo.7691378), Zenodo, version 1.1.4:
  `2019-11-01-ASR-public_12020` and `2019-11-01-FSR-public_7061`, archives and tables, checked
  with their MD5.
- Built by `scripts/build_core_mof.py`. The atoms are read as they are listed in each file.
  Every structure was then compared with the reading of its file by pymatgen: cell, number of
  atoms, and each atom at the position of one atom of the same element. Three properties were
  compared with the table.
- Repository: [materials-toolkits/core-mof](https://huggingface.co/datasets/materials-toolkits/core-mof),
  files `asr.hdf5` and `fsr.hdf5`.
