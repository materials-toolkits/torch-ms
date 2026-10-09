# Datasets to build yourself

Two sources are not published as files: one is too large to publish whole, the other must not
be redistributed. A script builds a local HDF5 file from each, in the format of the other
datasets, which is then loaded by its path:

```python
loader = auto_loader("datasets/zinc20_3d/small.hdf5", batch_size=128)
```

## ZINC20, molecules in 3D

ZINC20 (Irwin et al., 2020) gives 230 million purchasable molecules in 3D, ready to dock, with
their hydrogens. The positions are the ones of ZINC. Molecules have no label.

The molecules are in *tranches*, named by six letters: weight, logP, reactivity,
purchasability, pH and charge, each one a letter given by the documentation of ZINC: `BAAARN`
for instance. Tranches are selected by a regular expression on this code:

```bash
uv run python scripts/build_zinc20.py datasets --select "B[A-C]AARN" --list    # what it holds
uv run python scripts/build_zinc20.py datasets --select "B[A-C]AARN" --name small
```

- The file is `datasets/zinc20_3d/<name>.hdf5`, with `<name>_index.parquet` giving the ZINC
  identifier and the tranche of each row. Only the 3D positions and the elements are stored:
  no SMILES, no label.
- A molecule can be given several times, in several protonation states: each one is a row.
- Every molecule is compared with its file, read a second time by other code.
- The files are read from the copy of ZINC that its authors keep in the
  [open data of Amazon](https://registry.opendata.aws/zinc15/) (the bucket `zinc3d`, read
  without an account), several at a time (`--streams`).
- **Everything is 53,947 files of 39,375 tranches, 534 GB to download**, and more than twice
  as much once written. Without `--select` every tranche is taken: give a destination on a
  large disk. The in-stock tranches (fourth letter A or B) are 10 GB.
- The build can be stopped and started again, and `--download-only` stops before the HDF5
  file is written.
- **Do not publish the file.** ZINC is free to use, but significant portions of it may not be
  redistributed without the written permission of its authors.

## MOFs of the Cambridge Structural Database

The CSD MOF subset (about 137,000 entries) needs a licence of the CCDC. The CSD MOF Collection
(about 12,500 porous 3D frameworks, in P1, without solvent) is free for academic research, on
the site of the CCDC, after registration. **Neither can be published again**: torchms has no
file of them.

`scripts/build_csd_mof.py` converts CIF files you obtained from the CCDC:

```bash
uv run python scripts/build_csd_mof.py path/to/cifs datasets/csd_mof/all.hdf5
```

- The source is a directory of CIF files, or a CIF file with several structures, as ConQuest
  exports them.
- Structures with a site which is not fully occupied by one element are skipped. They are
  listed with the files which cannot be read in `all_skipped.csv`. For the MOF subset, export
  the non-disordered subset.
- `all_index.parquet` gives the name of each row: the name of its file or of its data block,
  a CSD refcode.
- The atoms of a P1 file are in the order of the file; the symmetry operations of the others
  are applied by pymatgen. Every structure is compared with the reading of pymatgen.
- Experimental structures: no label, hydrogens can be missing, cells can be very large (see
  [CoRE MOF](core-mof.md)).

[CoRE MOF 2019](core-mof.md) and [CoRE MOF 2024](core-mof-2024.md) are the published
alternatives: experimental MOFs under CC BY 4.0. The part of CoRE MOF 2024 derived from the
CSD is also distributed by the CCDC, and is converted by the same script.
