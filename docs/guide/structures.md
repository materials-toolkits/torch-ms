# Structures

`Structures` is the container of a batch of structures: crystals, molecules, or both. All the atoms of the batch are stored in
flat tensors, one after the other, with index tensors to find the atoms of each structure.

## Fields

| Field | Shape | Content |
|---|---|---|
| `lattice` | (n_struct, 3, 3) | lattice vectors as **rows**, in Å |
| `x` | (n_atoms, 3) | fractional coordinates; positions in Å for a molecule |
| `z` | (n_atoms,) | atomic numbers |
| `num_atoms` | (n_struct,) | number of atoms of each structure |
| `batch` | (n_atoms,) | index of the structure of each atom |
| `ptr` | (n_struct + 1,) | index of the first atom of each structure, then n_atoms |
| `total_energy` | (n_struct,) | total energy of the cell in eV |
| `formation_energy_per_atom` | (n_struct,) | formation energy in eV/atom |
| `energy_above_hull` | (n_struct,) | energy above the convex hull in eV/atom |
| `band_gap` | (n_struct,) | band gap in eV |
| `pbc` | (n_struct, 3) | whether the structure is periodic along each lattice vector |
| `properties` | dict of (n_struct,) | the other labels of a dataset, by name |

Four more fields are `None` unless a [graph transform](graphs.md#graphs-as-transforms) sets
them:

| Field | Type | Content |
|---|---|---|
| `graph` | `Graph` | edges between the atoms |
| `line_graph` | `EdgeTuples` | pairs of edges u→v, v→w |
| `triplets` | `EdgeTuples` | pairs of edges i→j, i→k from the same atom |
| `quadruplets` | `EdgeTuples` | chains of three edges u→v, v→w, w→x |

Their own fields are listed in
[Fields attached to the structures](graphs.md#fields-attached-to-the-structures). They are
batched, separated and moved to a device with the structures.

A label which is unknown is `NaN`. Cartesian positions are `x @ lattice`, for a crystal and for
a molecule:

```{.python notest}
positions = structures.positions()   # (n_atoms, 3), in Å
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

## Molecules

A molecule is a structure which is not periodic. It is created from its positions in Å:

```python
molecule = Structures.molecule([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], z=[1, 1])
molecule.pbc          # tensor([[False, False, False]])
molecule.lattice      # the identity: x is the positions

Structures.water()    # O, H, H
```

Crystals and molecules are batched together, and every transform takes such a batch:

```python
batch = Structures.collate([Structures.diamond(), Structures.water()])
batch.pbc.all(dim=1)  # tensor([ True, False])
```

The graph of a molecule has no periodic image, and the augmentations do not wrap nor repeat
it: see [Periodicity](../design/periodicity.md).

## Named properties

`properties` holds the labels of a dataset which are not one of the four fields, each one a
tensor with one value per structure:

```{.python notest}
structures = load_dataset("qm9")[:4]
structures.properties["homo"]          # tensor of 4 values, in eV

import dataclasses
structures = dataclasses.replace(structures, properties={"score": torch.zeros(4)})
```

They follow the structures through `collate`, `separate`, `select`, `split` and `to`. A batch
of structures with different properties has all of them, `NaN` where a structure has none.
They are not changed by the augmentations: a supercell keeps the values of its cell.

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

```{.python continuation}
larger = Structures.collate([batch, Structures.diamond(), batch])
larger.num_atoms              # tensor([8, 8, 8, 8, 8])

first.batch                   # tensor([0, 0, 0, 0, 0, 0, 0, 0])
```

The atoms of structure `i` are `batch.x[batch.ptr[i] : batch.ptr[i + 1]]`.

## Devices

All the tensors of a `Structures` must be on the same device; a mismatch raises
`InvalidStructure`. `structures.device()` gives the device, `structures.to(device)` a copy on
another one:

```{.python notest}
structures = structures.to("cuda")
structures.device()   # device(type='cuda', index=0)
```

Datasets load on the CPU: move each batch after loading it.

## Reading CIF

```python
from torchms.data import parse_cif

structures = parse_cif(open("file.cif").read())   # one structure per data block
```

Any space group is read. The atoms are the atom sites and their images by the symmetry
operations the block lists (`_space_group_symop_operation_xyz` or
`_symmetry_equiv_pos_as_xyz`), in its own setting; without them, by the operations of the
standard setting of its space-group number. Images closer than 1e-3 in fractional
coordinates are one atom, and sites of several elements at one place are one atom of the
most occupied element, as ASE reads them. Blocks without atoms are left out. The lattice is rebuilt from the cell lengths and angles, in the standard orientation (first
vector along x).
