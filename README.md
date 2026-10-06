# PyCrystalGen

## Setup with uv

### Installation on linux and macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative with `pip`:
```bash
pip install uv
```

### Add dependencies

```bash
uv add 'requests==2.31.0'
```

### Run

Run `python main.py` inside uv's virtual env.
```bash
uv run python main.py
```

## Datasets

Every dataset is converted to a single concatenated CIF file (P1, with
`_energy_total`, `_energy_above_hull` and `_band_gap` tags) by a script, then
preprocessed to HDF5 with the same command.

```bash
# LeMat-Bulk (downloaded from Hugging Face), --limit for a quick trial
uv run scripts/lematerial_to_cif.py lemat.cif --limit 10000

# CDVAE style CSV (MP-20, Carbon-24, Perov-5, Alex-MP-20)
uv run scripts/csv_to_cif.py mp_20/train.csv mp_20.cif \
    --hull-col e_above_hull --band-gap-col band_gap

# CIF to HDF5, with optional filters
uv run preprocessing lemat.cif lemat.hdf5 --num-workers 4 \
    --max-atoms 64 --exclude-elements He Ne Ar Kr Xe --max-energy-above-hull 0.1
```

`--max-energy-above-hull 0` keeps stable materials, `0.1` stable and meta-stable.
From Python, `Preprocessing(filter=...)` accepts any callable returning a mask of
the structures to keep (see `pycrystalgen.data.filters.Filter`).

```python
from pycrystalgen.data import HDF5Dataset

dataset = HDF5Dataset("lemat.hdf5")
dataset = HDF5Dataset.from_hub("user/repo", "lemat.hdf5")  # needs the "hub" extra
```

Upload a preprocessed file with `hf upload user/repo lemat.hdf5 --repo-type dataset`.

## Testing

Run testing with `pytest`:
```bash
uv run pytest
```

## Formatting

Format all files with `black`:
```bash
uv run black .
```

## Linter

Check with `pylint`:
```bash
uv run pylint .
```

