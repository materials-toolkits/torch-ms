# MatterSim

`MatterSim` is the M3GNet potential of MatterSim, `mattersim-v1.0.0-1M` and `-5M`
([mattersim](https://github.com/microsoft/mattersim) 1.2.5, MIT,
[arXiv:2405.04967](https://arxiv.org/abs/2405.04967)), in `torchms/models/mattersim/`.
MatterSim is a universal potential of Microsoft Research AI for Science, trained on
structures from 0 to 5000 K and up to 1000 GPa. Its released models are a rewrite of M3GNet
of its own, close to [M3GNet](m3gnet.md) of matgl but not the same network: they do not load
in one another. Forces and stress are not included.

| | |
|---|---|
| Class | `torchms.models.MatterSim` |
| Predicts | total energy in eV, one per structure, in single precision, as MatterSim |
| Source | [mattersim](https://github.com/microsoft/mattersim) 1.2.5 (tag `v1.2.5`, commit `40a1eb8`) |
| License | MIT |
| Paper | [arXiv:2405.04967](https://arxiv.org/abs/2405.04967) |
| Weights | `mattersim-v1.0.0-1m` (default), official, 880K parameters, 18 MB; `mattersim-v1.0.0-5m`, official, 4.5M parameters, 91 MB |
| Transform | `AddRadiusGraph(5.0, self_images=True)`, `AddTriplets(radius=4.0)` |
| Largest difference with the original | 9.5e-7 eV/atom (1M), 1.9e-6 eV/atom (5M) |

```{.python notest}
from torchms.data import load_dataset
from torchms.models import MatterSim

dataset = load_dataset("mp-20", "test", transform=MatterSim.transform())
model = MatterSim.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))        # (32,), eV
per_atom = model.atom_energies(dataset[:32].to("cuda"))

# the larger model: the same graph, a larger network
large = MatterSim.from_pretrained("mattersim-v1.0.0-5m", device="cuda")
```

**Two sizes.** The authors publish two models of the same architecture: `1M`, 3 blocks of
128 features trained on 3M structures, the one MatterSim loads by default, and `5M`, 4 blocks
of 256 features trained on 6M structures. The section `variants` of `config.yaml` lists what
the larger one changes; `MatterSim.for_weights(name)`, which `from_pretrained` calls, builds
the network of the size of the weights, and the arguments stored with them are checked: the
weights of one size are refused by the network of the other one.

## The rewrite

Atoms closer than 5 Å are bonded, the periodic images of an atom included. A bond starts
from a smooth Bessel basis of its length (Kocer, Mason and Erturk,
`torchms.models.layers.basis.SmoothBesselBasis`). In each of the three blocks, a bond i→j is
first updated from the angles it forms with the other bonds i→k of at most 4 Å, in a basis
of spherical Bessel functions of the length of i→k and of spherical harmonics of the angle,
weighted by the smooth cutoffs of the two bonds and by the atom i. The bonds, then the atoms,
are updated from each other. The energy of an atom is scaled and shifted by its element, and
the energy of a structure is their sum.

The spherical Bessel functions are written in closed form, as MatterSim writes them, over the
length divided by the radius: MatterSim keeps a table of their roots and of their
normalizations, which the rewrite computes from `spherical_bessel_roots` and
`spherical_bessel` (to 2.4e-7, one or two float32 steps). A recurrence, the one of
`SphericalBesselBasis`, rounds otherwise at short lengths: in single precision, MatterSim
differs from it by up to 2e-4 relative at 0.26 Å. In double precision, the two agree to 1e-6
from 1 to 4 Å.

## Comparison with the original

| Against mattersim 1.2.5 with torch 2.8.0, 40 structures (225 atoms) | `1M` | `5M` |
|---|---|---|
| Bonds and angles of each structure (8,252 bonds, 128,348 angles) | identical | identical |
| Sum of the lengths of the bonds of each structure | 8e-5 Å, the rounding of float32 | the same |
| Energy of each atom (values up to 14.7 and 16.5 eV): largest, mean | 5.7e-6, 6.9e-7 eV | 5.7e-6, 6.4e-7 eV |
| Energy: largest, mean | 9.5e-7, 3.5e-7 eV/atom | 1.9e-6, 4.0e-7 eV/atom |
| Energy, against MatterSim in double precision | 3.1e-6 eV/atom, as MatterSim in single precision (3.1e-6) | 4.0e-6 eV/atom, MatterSim 4.2e-6 |

MatterSim builds its graph with `find_points_in_spheres` of pymatgen, as matgl does: the
transforms give the same bonds and the same angles. A change of 0.1 % of one tensor of
weights or of a buffer makes the comparison fail.

**MatterSim 1.2.5 does not batch its structures correctly.** When several structures are
batched by its `build_dataloader`, the indices of its angles (`three_body_indices`) are not
offset by the bonds of the structures before them: on the 40 structures of the tests, batched
together, the largest index is 1,442 for 8,252 bonds, and the energies are wrong by up to 29
eV. Its calculator runs one structure at a time and is not affected. The reference was
computed one structure at a time; the rewrite has no such limit.

## Behaviours kept from MatterSim

- an angle i→j, i→k is weighted by the atom i, which the two bonds share. matgl weights it by
  the atom k: this is the main difference between the two networks, with the bases;
- the embedding of the elements is a linear map of a one-hot vector of the atomic number,
  from 0 to 94: atomic numbers 1 to 94 are read, the other ones are refused;
- the cosine of an angle is kept 1e-7 from ±1, then the basis reads the cosine of its arc
  cosine;
- the energy is in single precision, the default of MatterSim.

## Limits

- MatterSim places a molecule in a large periodic box. The rewrite reads it without
  periodicity, which gives the same bonds as long as the box is larger than the radius, as
  MatterSim makes it.
