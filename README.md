# Torch Material Science (TorchMS)

## Installation

> [!WARNING]
> TorchMS is in alpha (0.1.0): its interface may change between commits without notice.
> Pin a commit (see below) for results that can be reproduced.

TorchMS is a library: add it to the dependencies of your project. It needs Python 3.10 or
later. It is not on PyPI yet and is installed from its GitHub repository.

```bash
uv add "torchms @ git+https://github.com/materials-toolkits/torch-ms.git"
# or
pip install "torchms @ git+https://github.com/materials-toolkits/torch-ms.git"
```

Use `torchms[hub]` instead of `torchms` to load the published datasets and the energy
estimator from Hugging Face. To pin a commit, end the address with it:
`git+https://github.com/materials-toolkits/torch-ms.git@3dd18de`.

## Datasets

Preprocessed datasets are published on Hugging Face, in the same HDF5 format, and load
directly as batched `Structures`:

```python
from torchms.data import load_dataset  # needs the "hub" extra

dataset = load_dataset("mp-20", "train")  # or load_dataset("mp-20/train")
structures = dataset[:32]
```

`load_dataset` takes the name of the first column and a split; the default split is the first
one listed. `available_datasets()` lists them, `dataset_url("oqmd", "unique")` gives the URL of
a file, and `HDF5Dataset.from_hub(repo_id, filename)` loads a file of any repository.

| Name | Dataset | Repository | Splits | Structures | Labels |
|---|---|---|---|---|---|
| `perov-5` | Perov-5 | [materials-toolkits/perov-5](https://huggingface.co/datasets/materials-toolkits/perov-5) | `train`, `val`, `test` | 18,928 | formation energy, band gap |
| `carbon-24` | Carbon-24 | [materials-toolkits/carbon-24](https://huggingface.co/datasets/materials-toolkits/carbon-24) | `train`, `val`, `test` | 10,153 | total energy |
| `mp-20` | MP-20 | [materials-toolkits/mp-20](https://huggingface.co/datasets/materials-toolkits/mp-20) | `train`, `val`, `test` | 45,229 | formation energy, energy above hull, band gap |
| `mpts-52` | MPTS-52, split by year | [materials-toolkits/mpts-52](https://huggingface.co/datasets/materials-toolkits/mpts-52) | `train`, `val`, `test` | 40,476 | formation energy, energy above hull |
| `alex-mp-20` | Alex-MP-20, the training set of MatterGen | [materials-toolkits/alex-mp-20](https://huggingface.co/datasets/materials-toolkits/alex-mp-20) | `train`, `val` | 675,204 | energy above hull, band gap, 5 properties |
| `materials-project` | Materials Project (2026.09.28) | [materials-toolkits/materials-project](https://huggingface.co/datasets/materials-toolkits/materials-project) | `all` | 271,325 | total energy, formation energy, energy above hull, band gap |
| `oqmd` | OQMD (v1.8) | [materials-toolkits/oqmd](https://huggingface.co/datasets/materials-toolkits/oqmd) | `all`, `unique` | 1,407,269 / 1,259,553 | total energy, formation energy, stability, band gap |
| `lematerial` | LeMat-Bulk (PBE) | [materials-toolkits/lematerial](https://huggingface.co/datasets/materials-toolkits/lematerial) | `all`, `unique` | 5,335,299 / 4,927,038 | total energy, energy above hull |
| `qmof` | QMOF Database (v18), MOFs | [materials-toolkits/qmof](https://huggingface.co/datasets/materials-toolkits/qmof) | `all` | 20,372 | total energy, band gap, 9 properties |
| `core-mof` | CoRE MOF 2019, experimental MOFs | [materials-toolkits/core-mof](https://huggingface.co/datasets/materials-toolkits/core-mof) | `asr`, `fsr` | 12,020 / 7,061 | 13 properties, no energy |
| `core-mof-2024` | CoRE MOF 2024, public part | [materials-toolkits/core-mof-2024](https://huggingface.co/datasets/materials-toolkits/core-mof-2024) | `asr`, `fsr`, `ion`, `ncr` | 1,371 / 1,191 / 100 / 1,826 | 16 properties, no energy |
| `qm9` | QM9, molecules | [materials-toolkits/qm9](https://huggingface.co/datasets/materials-toolkits/qm9) | `all` | 133,885 | total energy, HOMO-LUMO gap, 17 properties |
| `zinc250k` | ZINC250k, molecules, positions from RDKit | [materials-toolkits/zinc250k](https://huggingface.co/datasets/materials-toolkits/zinc250k) | `all` | 249,427 | 4 properties, no energy |

Molecules are structures which are not periodic (`structures.pbc`). Crystals and molecules have
the same format and are loaded together: `auto_loader(["mp-20/train", "qm9"], batch_size=128)`.
The labels of a dataset beyond the four common ones are in `structures.properties`. ZINC20 in
3D and the MOFs of the Cambridge Structural Database are built locally by a script, see the
documentation.

A dataset is filtered when it is loaded, split at random from a seed, and mixed with others in
fixed shares per batch:

```python
from torchms.data import Filter, auto_loader, load_dataset, random_split

qm9 = load_dataset("qm9", filter=Filter(equal={"uncharacterized": 0}))
train, val, test = random_split(qm9, [0.8, 0.1, 0.1], seed=0)

# 64 crystals and 64 molecules in every batch
loader = auto_loader(["mp-20/train", train], batch_size=128, counts=[64, 64], shuffle=True)
```

Datasets are complete and unfiltered: unphysical labels of the sources are kept, filter
them when loading or preprocessing. Missing labels are NaN. `unique` files drop the
duplicates marked by the source (OQMD) or sharing a fingerprint (LeMat-Bulk), keeping the
structure with the lowest energy. The dataset cards describe the selection, the labels and
the checks of each dataset. OQMD and LeMat-Bulk come with an `*_index.parquet` giving the
identifier of each structure in its source.

### Format

Flat concatenated arrays: `lattice` (n_struct, 3, 3) in angstrom, `num_atoms` (n_struct,),
`ptr` (n_struct,) index of the first atom of each structure, `x` (n_atoms, 3) fractional
coordinates, `z` (n_atoms,) atomic numbers, and the labels `total_energy` (eV),
`formation_energy_per_atom` (eV/atom), `energy_above_hull` (eV/atom) and `band_gap` (eV),
all of shape (n_struct,).

### Building the datasets

Every dataset is converted to a single concatenated CIF file (P1, with `_energy_total`,
`_formation_energy_per_atom`, `_energy_above_hull` and `_band_gap` tags) by a script, then
preprocessed to HDF5 and checked against its source. Files are written to `datasets/`.

```bash
uv run python scripts/build_cdvae.py datasets              # Perov-5, Carbon-24, MP-20
uv run python scripts/build_materials_project.py datasets  # open data bucket, no API key
uv run python scripts/build_oqmd.py datasets               # 21 GB SQL dump, no MySQL server
uv run python scripts/build_oqmd.py datasets --unique
uv run python scripts/build_lematerial.py datasets         # read from Hugging Face
uv run python scripts/build_lematerial.py datasets --unique
```

The converters can be used alone, for another source or a quick trial:

```bash
# LeMat-Bulk, --limit for a quick trial
uv run scripts/lematerial_to_cif.py lemat.cif --limit 10000

# CDVAE style CSV (MP-20, Carbon-24, Perov-5, Alex-MP-20)
uv run scripts/csv_to_cif.py mp_20/train.csv mp_20.cif \
    --formation-energy-col formation_energy_per_atom \
    --hull-col e_above_hull --band-gap-col band_gap

# CIF to HDF5, with optional filters
uv run preprocessing lemat.cif lemat.hdf5 --num-workers 4 \
    --max-atoms 64 --exclude-elements He Ne Ar Kr Xe --max-energy-above-hull 0.1
```

`--max-energy-above-hull 0` keeps stable materials, `0.1` stable and meta-stable. With
`--num-workers`, structures are not written in the order of the CIF file. From Python,
`Preprocessing(filter=...)` accepts any callable returning a mask of the structures to
keep (see `torchms.data.filters.Filter`).

Upload a preprocessed file with `hf upload user/repo lemat.hdf5 --repo-type dataset`.

### Energy above the convex hull

`torchms.data.hull` computes the energy above the convex hull from total energies,
one chemical system at a time. Energies must be comparable (same level of theory).

```python
from torchms.data.hull import energy_above_hull, energy_above_reference, load_hull

# hull of a dataset, from its own structures: 0 on the hull
above = energy_above_hull(z, num_atoms, total_energy, num_workers=8)

# other structures against a saved hull: negative below the hull
hull = load_hull("hull.hdf5")
above = energy_above_reference(hull, z, num_atoms, total_energy)
```

`z`, `num_atoms` and `total_energy` are numpy arrays, as stored in the datasets. A saved
hull holds the composition and the energy per atom of the structures on the hull
(`elements`, `counts`, `ptr`, `energy_per_atom`). The same functions from the command line:

```bash
uv run python scripts/convex_hull.py dataset.hdf5 above.npy
uv run python scripts/convex_hull.py dataset.hdf5 above.npy --fill  # only missing values
uv run python scripts/convex_hull.py dataset.hdf5 above.npy --save-hull hull.hdf5
uv run python scripts/convex_hull.py generated.hdf5 above.npy --reference hull.hdf5
uv run python scripts/build_lematerial.py datasets --hull  # hull of LeMat-Bulk (PBE)
```

The hull of LeMat-Bulk (`hull.hdf5` of the LeMaterial dataset) is built from its structures of
the Materials Project and Alexandria only. LeMat-Bulk also holds structures from OQMD, whose
total energies are shifted compared with the two other sources for nine elements (about
+9.4 eV per atom of Gd, +0.38 Cu, +0.20 Ni, +0.19 Pu, +0.16 Pa, +0.14 Fe, +0.07 Np, +0.05 Re
and -0.19 Bi): mixed in one hull, they make it wrong for these elements. The
`energy_above_hull` stored in `all.hdf5` and `unique.hdf5` is the published one, which mixes
the three sources; it differs from the one of `hull.hdf5` by more than 0.001 eV/atom for 5% of
the structures of the Materials Project and Alexandria.

### Estimated total energy

`torchms.models.estimator` estimates the PBE total energy of a structure with a pretrained
graph network, on the scale of LeMat-Bulk, so that an energy above its convex hull can be
computed without DFT:

```python
from torchms.data import HDF5Dataset
from torchms.models.estimator import EnergyEstimator

# model, correction and convex hull of LeMat-Bulk, needs the "hub" extra
estimator = EnergyEstimator.from_lematerial(device="cuda")
print(estimator.hull_source)

# dataset: total energy in eV and energy above the hull in eV/atom
energy, above = estimator.predict_energy_above_hull(HDF5Dataset("structures.hdf5"))

# batch of Structures on the device of the model
above = estimator.energy_above_hull(structures)
```

`from_lematerial` downloads `energy_correction_orb.hdf5` and `hull.hdf5` from the
[LeMaterial dataset](https://huggingface.co/datasets/materials-toolkits/lematerial). The hull
is the convex hull of the structures of LeMat-Bulk (`compatible_pbe`) from the Materials
Project and Alexandria, without its structures from OQMD (see above); `estimator.hull_source`
says so. `EnergyEstimator.from_pretrained(hull_file=...)` takes another hull.

The energy above the hull is negative for a structure estimated more stable than the hull,
and NaN for a structure with an element absent from the hull. `estimator(structures)` and
`estimator.predict(dataset)` give the total energy alone.

The model is orb-v3 conservative (`orb-v3-conservative-inf-mpa`) of Orbital Materials,
rewritten in `models/orb/` for the prediction of the energy only, without any new
dependency; the official weights (102 MB) are downloaded on the first call into
`~/.cache/torchms`. A small energy per atom of each element, fitted on LeMat-Bulk, is
added (`models/estimator/calibration.py`, `energy_correction_orb.hdf5` in the LeMaterial dataset).

- The energy is the one of the structure as given: it is not relaxed. Compare relaxed
  structures with the hull.
- On structures of LeMat-Bulk which were not used to fit, and on OQMD structures absent from
  the sources of the training data of the model, the mean absolute error is about
  0.01 eV/atom (median 0.005). This is enough to tell if a structure is within 0.1 eV/atom
  of the hull (99% agreement with DFT), not to tell if it is exactly on the hull.
- About 600 structures per second on a laptop GPU.

```bash
uv run python scripts/build_energy_correction.py datasets  # fit the correction, test sets
uv run python scripts/validate_energy.py datasets          # end to end validation
```

## Models

`torchms.models` has supervised models, which predict the energy of a structure, generative
models, which generate structures, and the layers they share. The pretrained ones are models
of other projects, rewritten with PyTorch and the graphs of the library.

| | Model | Class | Output | Weights |
|---|---|---|---|---|
| Supervised | orb-v3 | `OrbV3` | total energy | official |
| | MACE | `MACE` | total energy | official |
| | M3GNet | `M3GNet` | total energy | official |
| | CHGNet | `CHGNet` | total energy | official |
| | ALIGNN | `ALIGNN` | formation energy | official |
| Generative | CDVAE | `CDVAE` | crystals | third party |
| | DiffCSP | `DiffCSP` | a crystal from its composition | official |
| | DiffCSP | `DiffCSPGeneration` | crystals, with their elements | official |

```python
from torchms.data import load_dataset
from torchms.models import M3GNet

dataset = load_dataset("mp-20", "test", transform=M3GNet.transform())
model = M3GNet.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))
```

A model does not build its graph: its class gives the transform which attaches it to the
structures. `torchms.models.layers` has the bases of distances and of angles, DimeNet++,
GemNet-T and two networks of the energy to train; `torchms.models.estimator` the estimator above. The documentation has a page for
each model, with how it was compared with the code it comes from.

## Development

To work on the library itself, clone the repository and let
[uv](https://docs.astral.sh/uv/) build its environment:

```bash
git clone https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git
cd torch-ms
uv sync
```

```bash
uv run pytest          # tests, including the examples of the docstrings
uv run pyright src     # type checking
uv run black .         # formatting
uv run pylint src      # linter
uv run poe docs        # documentation, preview on http://127.0.0.1:8000
uv run poe docs-build  # documentation, static site in site/
```

The documentation (user guide, datasets, design notes and API reference) is in `docs/`.
