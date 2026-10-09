# Preprocessing your own data

A dataset is built in two steps: structures are written to one CIF file, then the CIF file is
converted to HDF5.

```
source  →  one CIF file with all the structures  →  HDF5
```

## CIF to HDF5

```bash
uv run preprocessing structures.cif structures.hdf5
```

The CIF file holds one data block per structure, in P1 (all atoms listed, no symmetry
operation). Labels are read from these tags when present:

| Tag | Field | Unit |
|---|---|---|
| `_energy_total` | `total_energy` | eV |
| `_formation_energy_per_atom` | `formation_energy_per_atom` | eV/atom |
| `_energy_above_hull` | `energy_above_hull` | eV/atom |
| `_band_gap` | `band_gap` | eV |

Numbers must be in fixed-point notation (`0.00001`, not `1e-05`). A block which cannot be
read is skipped with a warning.

### Filters

```bash
uv run preprocessing structures.cif filtered.hdf5 \
    --max-atoms 64 \
    --exclude-elements He Ne Ar Kr Xe \
    --max-energy-above-hull 0.1
```

`--max-energy-above-hull 0` keeps stable structures, `0.1` stable and meta-stable ones;
structures without energy above the hull are dropped by this filter. From Python, any function
returning a mask of the structures to keep can be used, and `Filter` has more settings than the
command line: see [Filtering a dataset](datasets.md#filtering-a-dataset). The same filter can
be given when a dataset is loaded instead, which keeps the file complete:

```python
from torchms.data.filters import Filter
from torchms.data.preprocessing import Preprocessing, preprocessing

preprocessing(
    Preprocessing(
        source="structures.cif",
        destination="filtered.hdf5",
        filter=Filter(max_atoms=64, exclude_elements=(2, 10, 18, 36, 54)),
    )
)
```

### Speed and order

`--num-workers 4` parses in parallel. Structures are then **not** written in the order of the
CIF file; keep the default single process when the order matters, for example to match the
rows of the source.

## Writing the CIF file

`scripts/_cif_writer.py` writes blocks in the expected format:

```{.python notest}
from _cif_writer import write_block

with open("structures.cif", "w") as fp:
    write_block(fp, "my-id", lattice, fractional, ["Na", "Cl"], total_energy=-7.5)
```

Converters exist for the sources of the published datasets:

| Script | Source |
|---|---|
| `csv_to_cif.py` | CSV with a CIF column: MP-20, Carbon-24, Perov-5, Alex-MP-20 |
| `materials_project_to_cif.py` | summary collection of the Materials Project open data |
| `oqmd_extract.py`, `oqmd_to_cif.py` | SQL dump of OQMD, without MySQL server |
| `lematerial_to_cif.py` | LeMat-Bulk on Hugging Face |

## Rebuilding the published datasets

Each build script downloads the source, converts it, and **checks every structure of the
result against the source** before finishing.

```bash
uv run python scripts/build_cdvae.py datasets              # Perov-5, Carbon-24, MP-20
uv run python scripts/build_materials_project.py datasets
uv run python scripts/build_oqmd.py datasets               # 21 GB download
uv run python scripts/build_oqmd.py datasets --unique
uv run python scripts/build_lematerial.py datasets
uv run python scripts/build_lematerial.py datasets --unique
uv run python scripts/build_lematerial.py datasets --hull
uv run python scripts/build_energy_correction.py datasets  # energy estimator
uv run python scripts/validate_energy.py datasets
```

## Publishing

```bash
hf upload user/repository structures.hdf5 --repo-type dataset
```

then `HDF5Dataset.from_hub("user/repository", "structures.hdf5")`.
