# Structures

`Structures` is the container of a batch of crystals. All the atoms of the batch are stored in
flat tensors, one after the other, with index tensors to find the atoms of each structure.

## Fields

| Field | Shape | Content |
|---|---|---|
| `lattice` | (n_struct, 3, 3) | lattice vectors as **rows**, in Å |
| `x` | (n_atoms, 3) | fractional coordinates |
| `z` | (n_atoms,) | atomic numbers |
| `num_atoms` | (n_struct,) | number of atoms of each structure |
| `batch` | (n_atoms,) | index of the structure of each atom |
| `ptr` | (n_struct + 1,) | index of the first atom of each structure, then n_atoms |
| `total_energy` | (n_struct,) | total energy of the cell in eV |
| `formation_energy_per_atom` | (n_struct,) | formation energy in eV/atom |
| `energy_above_hull` | (n_struct,) | energy above the convex hull in eV/atom |
| `band_gap` | (n_struct,) | band gap in eV |

A label which is unknown is `NaN`. Cartesian positions are `x @ lattice`:

```python
positions = torch.bmm(structures.x.unsqueeze(1), structures.lattice[structures.batch]).squeeze(1)
```

## Creating structures

Only `lattice`, `x` and `z` are required; the other fields are derived. Lists and numpy arrays
are converted.

```python
import torch
from torchms.data import Structures

structure = Structures(
    lattice=torch.eye(3).unsqueeze(0) * 4.0,
    x=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    z=[11, 17],
)
structure.num_atoms   # tensor([2])
structure.ptr         # tensor([0, 2])

Structures.diamond()      # 8 carbon atoms
Structures.zincblende()   # 4 Zn and 4 S
```

For several structures in one call, give `num_atoms`.

## Batching

```python
batch = Structures.collate([Structures.diamond(), Structures.zincblende()])
batch.num_atoms               # tensor([8, 8])
batch.batch                   # tensor([0, 0, ..., 1, 1])

first, second = Structures.separate(batch)
```

`collate` also takes batches of several structures, and `separate` gives structures which
are batches of one structure:

```python
larger = Structures.collate([batch, Structures.diamond(), batch])
larger.num_atoms              # tensor([8, 8, 8, 8, 8])

first.batch                   # tensor([0, 0, 0, 0, 0, 0, 0, 0])
```

The atoms of structure `i` are `batch.x[batch.ptr[i] : batch.ptr[i + 1]]`.

## Devices

All the tensors of a `Structures` must be on the same device; a mismatch raises
`InvalidStructure`. `structures.device()` gives the device, `structures.to(device)` a copy on
another one:

```python
structures = structures.to("cuda")
structures.device()   # device(type='cuda', index=0)
```

Datasets load on the CPU: move each batch after loading it.

## Reading CIF

```python
from torchms.data import parse_cif

structures = parse_cif(open("file.cif").read())   # one structure per data block
```

Only P1 CIF files are supported: atoms must be listed explicitly, without symmetry operations.
The lattice is rebuilt from the cell lengths and angles, in the standard orientation (first
vector along x).
