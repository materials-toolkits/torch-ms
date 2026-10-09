# ZINC250k

The 249,455 drug-like molecules drawn from ZINC by Gómez-Bombarelli et al. (2018), a usual
benchmark of molecule generation. They are **molecules, not crystals**: `pbc` is `False` and `x`
holds positions in Å, see [Periodicity](../design/periodicity.md).

!!! warning "The positions are not from the source"
    ZINC250k is a table of SMILES. The 3D positions of this dataset were **generated when it
    was built**: hydrogens added, one conformer per molecule with RDKit 2026.03.6 (ETKDGv3,
    seed 42), then relaxed with the MMFF94 force field. They are plausible geometries, not
    DFT ones, and not the conformers of another paper.

| | |
|---|---|
| Molecules | 249,427 (249,455 in the source, 28 without conformer) |
| Atoms | 10,923,322, 9 to 83 per molecule (44 on average), hydrogens included |
| Elements | H, C, N, O, F, P, S, Cl, Br, I |

```python
dataset = load_dataset("zinc250k")
```

!!! note "ZINC and redistribution"
    ZINC is free to use, but its terms say that significant portions of it may not be
    redistributed without the written permission of its authors. ZINC250k is a small sample,
    0.1 % of ZINC, and its table of SMILES is published by the authors of the benchmark: it is
    published here on this ground. Larger parts of ZINC are not: see
    [Datasets to build yourself](local.md).

    The file can also be built again, in about half an hour on eight cores:

    ```bash
    uv run python scripts/build_zinc250k.py datasets
    ```

## Labels

The four labels (`total_energy`, `band_gap`, ...) are `NaN`.

| `properties[...]` | Source column | |
|---|---|---|
| `logp` | `logP` | octanol-water partition coefficient |
| `qed` | `qed` | drug-likeness, 0 to 1 |
| `sas` | `SAS` | synthetic accessibility score |
| `mmff_converged` | — | 1 if the relaxation with MMFF94 converged |

## Things to know

- **28 molecules are missing**: RDKit found no conformer for them, mostly strained polycyclic
  cages. `all_index.parquet` gives for each row its row in the source and its SMILES.
- Two molecules are kept with `mmff_converged` at 0.
- **Charged molecules** are in the source (protonated amines, for instance) and are kept.
- **No bond is stored**, as for every dataset: graphs are built from the positions.
- **No split.** Papers draw their own, as `random_split(dataset, [0.8, 0.1, 0.1], seed=0)`.
- **Licence.** The table comes from the repository of the paper (Apache 2.0); the molecules
  come from ZINC: free to use, significant portions not to be redistributed without written
  permission.
- **Same positions with the same RDKit.** The build is reproducible with RDKit 2026.03.6;
  another version can give other conformers.

## Filters

**Recommended**: the molecules whose relaxation converged, which is all of them but two.

```python
from torchms.data import Filter, load_dataset

dataset = load_dataset("zinc250k", filter=Filter(equal={"mmff_converged": 1}))
```

| Filter | Keeps, of 249,427 |
|---|---|
| `Filter(equal={"mmff_converged": 1})`, **recommended** | 249,425 |
| `Filter(equal={"mmff_converged": 1}, max_atoms=60)`, hydrogens included | 243,396 |
| `Filter(minimum={"qed": 0.5})`, the more drug-like | 228,905 |
| `Filter(elements=(1, 6, 7, 8))`, the elements of QM9 without fluorine | 102,452 |

See [Filtering a dataset](../guide/datasets.md#filtering-a-dataset) for every setting.

## Source and checks

- `250k_rndm_zinc_drugs_clean_3.csv` of
  [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae), checked with its MD5.
- Built by `scripts/build_zinc250k.py`. For every molecule: its atoms are the ones of its
  SMILES with its hydrogens, every bond of the SMILES is shorter than 2.5 Å, no two atoms are
  closer than 0.5 Å, and the three properties are the ones of the table. 500 molecules were
  generated a second time and got the same positions.
- Repository: [materials-toolkits/zinc250k](https://huggingface.co/datasets/materials-toolkits/zinc250k),
  file `all.hdf5`.
