# MatterGen

MatterGen is the diffusion model of crystals of Zeni, Pinsler, Zügner, Fowler, Horton, Fu
et al. ([MatterGen](https://github.com/microsoft/mattergen), commit `9242366`, version
1.0.3, MIT, [Nature 2025](https://doi.org/10.1038/s41586-025-08628-5)), in
`torchms/models/mattergen/`. Seven of its models are rewritten: the base model, which
generates crystals of given numbers of atoms with their elements, without condition, and
the models fine-tuned on a chemical system, a space group, the band gap, the magnetic
density, a chemical system with the energy above the hull, and the bulk modulus, which
generate them with given elements or towards these properties: see
[Conditioned on a chemical system](#conditioned-on-a-chemical-system),
[Conditioned on a space group](#conditioned-on-a-space-group),
[Conditioned on a band gap](#conditioned-on-a-band-gap),
[Conditioned on a magnetic density](#conditioned-on-a-magnetic-density),
[Conditioned on a chemical system and the energy above the hull](#conditioned-on-a-chemical-system-and-the-energy-above-the-hull)
and [Conditioned on a bulk modulus](#conditioned-on-a-bulk-modulus).

| | |
|---|---|
| Class | `MatterGen`; `MatterGenChemicalSystem` for a chemical system, `MatterGenSpaceGroup` for a space group, `MatterGenBandGap` for a band gap, `MatterGenMagneticDensity` for a magnetic density, `MatterGenChemicalSystemEnergyAboveHull` for both, `MatterGenBulkModulus` for a bulk modulus |
| Does | samples the elements, the lattice and the coordinates of crystals of given numbers of atoms |
| Source | [MatterGen](https://github.com/microsoft/mattergen), commit `9242366` (1.0.3) |
| License | MIT; its diffusion of the elements and its sampler are adapted from Google Research code under Apache-2.0 |
| Paper | [A generative model for inorganic materials design](https://doi.org/10.1038/s41586-025-08628-5), Nature 639, 624 (2025), [arXiv:2312.03687](https://arxiv.org/abs/2312.03687) |
| Weights | `mattergen-base`, of the authors (`mattergen_base` on [Hugging Face](https://huggingface.co/microsoft/mattergen)), trained on Alex-MP-20, 461 MB; `mattergen-chemical-system` (`chemical_system`), `mattergen-space-group` (`space_group`), `mattergen-dft-band-gap` (`dft_band_gap`), `mattergen-dft-mag-density` (`dft_mag_density`), `mattergen-chemical-system-energy-above-hull` (`chemical_system_energy_above_hull`) and `mattergen-ml-bulk-modulus` (`ml_bulk_modulus`), fine-tuned from it, 512 MB each |
| Network | [GemNet-T](gemnet.md), 512 features, 4 blocks, 44.6 M parameters |
| Transform | `AddRadiusGraph(7.0, max_image_shift=5)`, `KeepNearestNeighbors(50)`, `MirrorEdges()`, `AddTriplets()`, `AddBonds()` |
| Difference with the original | a step of the sampling: identical, given the same scores and noise. The network: as near to MatterGen in double precision as MatterGen in single precision is, up to 1.6e-2 on the score of a lattice (values up to 5.6) |

```{.python notest}
import torch

from torchms.models import MatterGen

model = MatterGen.from_pretrained(device="cuda")
num_atoms = model.sample_num_atoms(64)      # as the ones of Alex-MP-20, or any numbers

structures = model.sample(num_atoms)         # 1000 steps, 2000 calls of the network
structures.z                                 # atomic numbers
structures.lattice                           # (64, 3, 3), in Å, symmetric matrices
structures.x                                 # fractional coordinates
```

With a generator of seed 0 on a GPU, 8 crystals of 4 to 18 atoms take 155 s (a GPU of 8 GB)
and have the compositions Dy4AsSe2Tl, KFeF3, ScMgGaNd, Li2Na2PbF12 (twice LiNaPbF6),
Pr4CuGa2O11, ZrAs2, Sr2IrRhGa4 and Cu4F10, with 12 to 26 Å³ per atom. These are single
draws: the accuracy of the model is the one of its paper, it was not measured here.

## The rewrite

A crystal is a lattice, the fractional coordinates of its atoms and their elements. From
the time 0 to the time 1, the three get noise:

- the lattice goes to a cubic cell of the mean density of the training set, 0.0577 atoms
  per Å³, by a diffusion which preserves the variance (rates from 0.1 to 20). Its noise is a
  symmetric matrix, whose variance grows as the number of atoms to the power 2/3;
- the coordinates get a noise wrapped around the cell, whose level grows geometrically from
  0.01 to 5, divided by the cube root of the number of atoms;
- the element of an atom becomes the mask, an element of its own, the 101st, with a
  probability `t` at the time `t` (an absorbing D3PM of 1000 steps).

The network, [GemNet-T](gemnet.md), reads a noisy crystal and the embedding of its time,
and gives the scores of the lattice and of the coordinates times their deviations, and the
scores of the element each atom had at the time 0 (`model(structures, times)`, a
`Denoised`). Its graph connects an atom to its 50 nearest neighbours closer than 7 Å, its
own images included; its edges also read their cosines with the lattice vectors, and a
read-out of its edges gives the score of the lattice.

`sample` starts from `sample_prior`, random lattices and coordinates with every atom
masked, and makes 1000 steps from the time 1 to the time 1/1000. `sample_step` is one of
them, with its random draws as a `StepNoise`:

1. `correct`: a Langevin move of the coordinates and of the lattice, with a step computed
   from the signal to noise ratios 0.4 and 0.2; the lattice is then made symmetric again
   (`symmetric_lattice`, a polar decomposition);
2. `predict`: the ancestral step of each diffusion to the time of the next step. An atom
   keeps its element or, when masked, stays masked or takes an element drawn from the
   probabilities the network gives (`element_posterior`).

Each of the two calls the network on a graph built for the crystals it is given.
`correct_given_scores` and `predict_given_scores` are the same steps from given scores.
After the last step no noise is added, and each atom takes its most probable element:
these are the crystals MatterGen returns.

Elements are drawn with Gumbel noise: `argmax(logits + gumbel)` is a draw of the
categorical distribution of these logits. The noise of the last step is `elements=None`.

The number of atoms is not generated. MatterGen draws it from the numbers of atoms of its
training set, a table of its code; `sample_num_atoms` draws from it: from 1 to 20 atoms.

Not included: the models conditioned on the other properties (bulk modulus, and magnetic
density with HHI score), the model of MP-20 and the
prediction of a structure from its composition, the training, the fine-tuning, the datasets
and the evaluation. The checkpoints hold only tensors and are read with
`weights_only=True`. The configuration stored with them is checked: each model refuses the
weights of the other one and of the models conditioned on other properties.

## Conditioned on a chemical system

`MatterGenChemicalSystem` is MatterGen fine-tuned on Alex-MP-20 with the chemical system of
each crystal, the set of its elements: all its weights were trained again, from the ones of
the base model. It generates crystals whose elements are the ones given.

```{.python notest}
from torchms.models import MatterGenChemicalSystem

model = MatterGenChemicalSystem.from_pretrained(device="cuda")
num_atoms = model.sample_num_atoms(8)
guidance = model.guidance(["Li-Fe-P-O"] * 8)    # one system per crystal, factor 2

structures = model.sample(num_atoms, guidance=guidance)   # 4000 calls of the network
```

- **The network** has an adapter before each of its 4 blocks: a network on the features of
  an atom and the embedding of the chemical system of its crystal (a linear layer on 101
  values, one per atomic number), whose output is added to the features through a linear
  layer. A crystal without condition skips it (`gemnet.Condition`). For a conditioned
  crystal, the scores of the elements outside its system are lowered by 1e10, as the ones
  not allowed: its atoms can only take them.
- **The guidance** (classifier-free) mixes the scores without and with the condition,
  `unconditional + scale (conditional - unconditional)`, for the coordinates, the lattice
  and the elements: two calls of the network per score. `guidance(systems, scale=…)` gives
  a `Guidance`, which holds one value per crystal for each property of the model; its
  factor is 2 by default, the one of the README of MatterGen. 0 is the
  model without condition, 1 the model with it. Without guidance, `sample` samples the
  fine-tuned model without condition: these are not the crystals of `MatterGen`. The design
  note [Classifier-free guidance](../design/guidance.md) says how the scores are mixed, what
  is imposed and what is followed, and how far a number can be from the training set.
- `model(structures, times, guidance.values, conditioned)` is the network alone, with a
  mask of the conditioned crystals; `scores(structures, times, guidance)` the mixed scores.

With a generator of seed 0, 8 crystals of 6 to 16 atoms take 271 s on a GPU of 8 GB, and all
their atoms have the elements asked for: Li4O2 (Li-O, 0.002 eV/atom above the hull,
unrelaxed), Li3Fe2P2O7 (Li-Fe-P-O, 0.98), Na4Cl4 (0.087), Ti8O4 (0.19), Ga4N4 (0.058), Si4C4
(−0.002), MgAl5O8 (0.23) and CuZn3Sn3S9 (0.13). These are single draws, without
relaxation.

### Comparison with the original

Compared with MatterGen `9242366` as the base model, with the weights `chemical_system`: the
40 structures of the tests, every other one conditioned on its own chemical system, and the
sampling of 4 crystals of 4, 8, 2 and 6 atoms towards Li-O, Fe-Ni-O, Na-Cl and Ti-O, with a
factor of 2, recording both scores the guidance mixes at 10 steps.

| | |
|---|---|
| Mixed scores of the guidance, from the two scores of MatterGen | identical, at every recorded step |
| A step of the guided sampling, given its scores and its noise: coordinates, lattice, probabilities of the elements | identical |
| Elements of the crystals MatterGen returns | identical: Li2O2, FeNi3O4, NaCl, Ti2O4 |
| Network, conditioned structures without noise (12, 56 atoms): lattice, largest / mean difference with MatterGen in double precision (values up to 5.1) | 1.2e-2 / 3.1e-4, MatterGen in single precision 1.2e-2 / 2.3e-4 |
| The same, scores of the elements (values up to 20) | 4.6e-2 / 3.9e-3, MatterGen 2.3e-2 / 2.0e-3 |
| Network, conditioned structures with noise (16, 93 atoms): coordinates, lattice, elements | 3.9e-4, 1.4e-4, 4.7e-3; MatterGen 5.8e-4, 1.5e-4, 5.7e-3 |
| Network, structures without condition (23): all within the rounding of MatterGen | up to 2.8e-2 on a lattice, as MatterGen in single precision |

The scores of the elements of the conditioned structures without noise are twice as far
from MatterGen in double precision as MatterGen is, on average. This is rounding again,
measured: moving the lattice of these structures by its last bit moves these scores by
7.6e-3 on average, twice the difference. The tests accept a difference as small as MatterGen
in single precision, or within 3 times what a change of the last bits of the inputs does to
the rewrite. The same holds on the crystals of the sampling: on a masked crystal of 2 atoms
at the time 0.998, its last bits move the score of its lattice by up to 0.2.

## Conditioned on a space group

`MatterGenSpaceGroup` is MatterGen fine-tuned the same way on the space group of each
crystal. It works as `MatterGenChemicalSystem`, with a learned vector per space group (an
embedding of 230 rows) in place of the embedding of a chemical system, and nothing else:
**the space group is not imposed**. No score is masked and no symmetry is enforced: the
network was trained to give crystals of the group, and the guidance pushes towards it.

```{.python notest}
from torchms.models import MatterGenSpaceGroup

model = MatterGenSpaceGroup.from_pretrained(device="cuda")
guidance = model.guidance([225] * 8)            # Fm-3m, numbers 1 to 230, factor 2
structures = model.sample(model.sample_num_atoms(8), guidance=guidance)
```

`guidance` takes the numbers of the international tables only. The README of MatterGen
gives no guidance factor for the space groups: the default, 2, is the one of its examples
for the other properties.

With a generator of seed 0, 32 crystals, 8 for each of Fm-3m (225), P6₃/mmc (194), Pm-3m
(221) and P2₁/c (14), with numbers of atoms drawn as the ones of Alex-MP-20, take 1096 s on
a GPU of 8 GB. Without relaxation, pymatgen finds the space group asked for in 13 of them
with a tolerance of 0.1 Å (4, 3, 4 and 2 of 8), and in 6 with 0.01 Å. Most of the other
ones are near it: 225 for 221, 164 and 156 for 194. MatterGen measures this after a
relaxation, which these crystals did not have: these numbers are not the ones of its paper.

### Comparison with the original

As for the chemical systems: the 40 structures of the tests, every other one conditioned on
its own space group (pymatgen, 0.1 Å), and the guided sampling of 4 crystals towards 225,
194, 221 and 166.

| | |
|---|---|
| Mixed scores of the guidance, and a step of the guided sampling given its scores and its noise | identical |
| Elements of the crystals MatterGen returns | identical |
| Network, conditioned structures without noise (12, 56 atoms): lattice, largest / mean difference with MatterGen in double precision (values up to 5.1) | 1.5e-2 / 5.3e-4, MatterGen in single precision 9.8e-3 / 1.9e-4 |
| The same, scores of the elements (values up to 22) | 1.3e-1 / 4.9e-3, MatterGen 8.1e-2 / 3.1e-3 |
| Network, conditioned structures with noise (16, 84 atoms): coordinates, lattice, elements | 4.7e-5, 5.3e-4, 3.3e-3; MatterGen 4.9e-5, 9.9e-4, 1.4e-2 |
| Network, structures with noise without condition (17): coordinates | 1.2e-2 / 5.8e-4, MatterGen 3.1e-3 / 1.4e-4 |

The last row is one structure, of 12 atoms at the time 0.078: changes of the last bits of its
inputs move its scores by up to 3.2e-2, so the rewrite and MatterGen are both within this
rounding of the exact value. On the other 16 structures, the rewrite is as near to MatterGen
in double precision as MatterGen is.

## Conditioned on a band gap

`MatterGenBandGap` is MatterGen fine-tuned on the band gap from DFT, in eV. It works as the
two models above: the band gap of a crystal is standardized by the mean and the deviation of
the training set, which are in the weights (0.798 and 1.408 eV), then embedded by sines and
cosines as the time, with frequencies also in the weights. Like the space group, **the band
gap is followed, not imposed**: nothing in the sampling checks it.

```{.python notest}
from torchms.models import MatterGenBandGap

model = MatterGenBandGap.from_pretrained(device="cuda")
guidance = model.guidance([3.0] * 8)            # eV, factor 2
structures = model.sample(model.sample_num_atoms(8), guidance=guidance)
```

The guidance factor is 2 by default, the value of the README of MatterGen for its other
number, the magnetic density.

With a generator of seed 0, 8 crystals take 292 s on a GPU of 8 GB. The 4 asked at 0 eV are
intermetallics, V2As4Sb2, Tb2Fe5C3, Nd3Mg, and a hydroxyfluoride, Mg Mn5 H O5 F6; the 4
asked at 4 eV are oxides and hydroxides of aluminium, zinc and strontium. Their band gaps
were not computed: torchms has no model of the band gap.

### Comparison with the original

As for the other fine-tuned models: the 40 structures of the tests, every other one
conditioned on a band gap from 0 to 6 eV (they have none of their own), and the guided
sampling of 4 crystals towards 0, 1.5, 3 and 6 eV. MatterGen generated Yb2PdPb, Cs2SeOF4,
MgTe and NH4Cl.

| | |
|---|---|
| Mixed scores of the guidance, and a step of the guided sampling given its scores and its noise | identical |
| Elements of the crystals MatterGen returns | identical |
| Network, conditioned structures without noise (12, 56 atoms): coordinates, lattice, elements, largest / mean difference with MatterGen in double precision | 2.6e-3 / 1.2e-4, 1.4e-2 / 4.2e-4, 9.9e-2 / 3.8e-3; MatterGen in single precision 2.6e-3 / 1.2e-4, 1.4e-2 / 3.2e-4, 1.0e-1 / 4.1e-3 |
| Network, conditioned structures with noise (14, 66 atoms): coordinates, lattice, elements, largest | 7.1e-4, 8.0e-4, 1.1e-2; MatterGen 7.0e-4, 7.5e-4, 1.7e-2 |

## Conditioned on a magnetic density

`MatterGenMagneticDensity` is MatterGen fine-tuned on the magnetic density from DFT. It is
the model of the band gap with another number, embedded the same way, with the statistics
of its own training set in its weights (here a mean of 0.0042 and a deviation
of 0.0158). The code of MatterGen names no unit; its README asks for 0.15 with a guidance of
2, the default here. The density is followed, not imposed.

```{.python notest}
from torchms.models import MatterGenMagneticDensity

model = MatterGenMagneticDensity.from_pretrained(device="cuda")
structures = model.sample(model.sample_num_atoms(8), guidance=model.guidance([0.15] * 8))
```

With a generator of seed 0, 16 crystals take 560 s on a GPU of 8 GB. All the 8 asked at 0.15
have a magnetic element (Mn, Fe, rare earths): Gd Mn2 Si2, Eu9 Ir4 Sb4, Gd2 Ga Rh, Gd2 Li Cd,
Fe6 Si H, Fe10 Mn2 H8, Fe Gd2 N2, Fe3 Pt H. Of the 8 asked at 0, 3 have one, a rare earth in a
compound of other elements. Their magnetization was not computed.

### Comparison with the original

As for the band gap, with magnetic densities from 0 to 0.2 for the 40 structures of the tests
and 0, 0.05, 0.15 and 0.3 for the guided sampling, where MatterGen generated Lu2Sm, CoFeMnF6,
GdAs and Fe4Co2.

| | |
|---|---|
| Mixed scores of the guidance, and a step of the guided sampling given its scores and its noise | identical |
| Elements of the crystals MatterGen returns | identical |
| Network, conditioned structures without noise (12, 56 atoms): coordinates, lattice, elements, largest / mean difference with MatterGen in double precision | 1.1e-3 / 6.7e-5, 1.1e-2 / 3.6e-4, 8.4e-2 / 5.9e-3; MatterGen in single precision 1.1e-3 / 5.7e-5, 1.1e-2 / 2.4e-4, 6.6e-2 / 3.0e-3 |
| Network, conditioned structures with noise (14, 66 atoms): coordinates, lattice, elements, largest | 1.0e-4, 5.2e-4, 6.4e-3; MatterGen 1.5e-4, 6.0e-4, 1.1e-2 |

As for the chemical systems, the scores of the elements of the conditioned structures without
noise are twice as far from MatterGen in double precision as MatterGen is, on average; the
tests accept them as within what a change of the last bits of the inputs does to the rewrite.

## Conditioned on a chemical system and the energy above the hull

`MatterGenChemicalSystemEnergyAboveHull` is MatterGen fine-tuned on two properties together:
the chemical system and the energy above the hull, in eV/atom. MatterGen has no model of the
energy above the hull alone. The network has an adapter per property before each block and
adds their outputs. The chemical system restricts the elements as in
`MatterGenChemicalSystem`; the energy is standardized by the statistics of the training set
(0.046 and 0.030 eV/atom in the weights) and embedded as the band gap. Either property can be
left without condition with `None`.

```{.python notest}
from torchms.models import MatterGenChemicalSystemEnergyAboveHull

model = MatterGenChemicalSystemEnergyAboveHull.from_pretrained(device="cuda")
guidance = model.guidance(["Li-Fe-P-O"] * 8, [0.0] * 8)   # systems, eV/atom; factor 2
structures = model.sample(num_atoms, guidance=guidance)
model.guidance(None, [0.0] * 8)                             # the energy alone
```

With a generator of seed 0, 8 crystals of 8 chemical systems were sampled at 0 and at 0.2
eV/atom above the hull (281 s each on a GPU of 8 GB), with the same numbers of atoms and the
same noise. Every atom has an element of its system. Their energy above the hull, estimated
by `EnergyEstimator` without relaxation, has a median of 0.057 eV/atom at 0 and 0.116 at 0.2:
GaN goes from 0.008 to 0.147, SiC from −0.002 to 0.090, a Cu–Zn–Sn–S crystal from 0.012 to
0.121, while three systems go the other way. Eight pairs, unrelaxed: an indication, not a
measure.

Higher targets work only up to a point: the median reaches 0.210 eV/atom at 0.5, 15
deviations from the mean of the training set, and falls back to 0.080 at 1. The model
follows the energy up to about 0.2 to 0.5 eV/atom; the numbers of every system are in the
design note [Classifier-free guidance](../design/guidance.md#the-range-of-a-number).

### Comparison with the original

As for the other fine-tuned models: every other structure of the tests conditioned on its
own chemical system and an energy from 0 to 0.2 eV/atom, and the guided sampling of 4 crystals
towards Li-O, Fe-Ni-O, Na-Cl and Ti-O at 0, 0, 0.05 and 0.1 eV/atom, where MatterGen
generated Li2O2, Fe3NiO4, NaCl and Ti4O2.

| | |
|---|---|
| Mixed scores of the guidance, and a step of the guided sampling given its scores and its noise | identical |
| Elements of the crystals MatterGen returns | identical |
| Network, conditioned structures without noise (12, 56 atoms): coordinates, lattice, elements, largest / mean difference with MatterGen in double precision | 1.9e-3 / 6.6e-5, 1.2e-2 / 2.8e-4, 3.3e-2 / 2.7e-3; MatterGen in single precision 1.9e-3 / 6.2e-5, 1.2e-2 / 2.0e-4, 3.1e-2 / 2.4e-3 |
| Network, conditioned structures with noise (14, 72 atoms): coordinates, lattice, elements, largest / mean | 2.4e-2 / 4.8e-4, 6.6e-2 / 3.0e-3, 3.6e-1 / 7.3e-3; MatterGen 1.4e-2 / 3.8e-4, 6.6e-2 / 3.0e-3, 1.7e-1 / 4.7e-3 |

The scores of the elements of the noisy structures are up to twice as far from MatterGen in
double precision as MatterGen is; the tests accept them as within what a change of the last
bits of the inputs does to the rewrite.

## Conditioned on a bulk modulus

`MatterGenBulkModulus` is MatterGen fine-tuned on the bulk modulus, in GPa, predicted by a
machine-learned potential for the crystals of its training set, not computed by DFT. It is
the model of the band gap with another number, with one difference: the bulk modulus is
standardized after its decimal logarithm, as in MatterGen, by the statistics of the
logarithms in the weights (1.708, 51 GPa, and 0.364, a factor of 2.3). It must be positive.
The model card of MatterGen asks for 400 GPa; the README gives no guidance factor, the
default is 2. The bulk modulus is followed, not imposed.

```{.python notest}
from torchms.models import MatterGenBulkModulus

model = MatterGenBulkModulus.from_pretrained(device="cuda")
structures = model.sample(model.sample_num_atoms(8), guidance=model.guidance([400.0] * 8))
```

With a generator of seed 0, 16 crystals, 8 asked at 30 GPa and 8 at 400, take 537 s on a
GPU of 8 GB. Their bulk modulus was estimated from the curvature of the energy of
`EnergyEstimator` against the volume, the crystals scaled by ±6% without relaxation: a
median of 38 GPa (27 to 75) for the ones asked at 30, and of 347 GPa (334 to 387) for the
ones asked at 400. These are all alloys of the stiffest metals: OsIr4, WOs3, RuRe2Ir, OsIr7,
W4Re7Os9, WOs4, B Re Os5 Ir10, and pure rhenium, estimated at 354 GPa (about 370 measured).
The ones asked at 30 are salts and intermetallics of heavy, soft elements (Cs, Tl, Sr, rare
earths). Eight crystals per value, unrelaxed, and an estimate of the bulk modulus without
the relaxation of the atoms: an indication, not a measure.

### Comparison with the original

As for the band gap, with bulk moduli from 10 to 400 GPa for the 40 structures of the tests
and 20, 50, 150 and 400 GPa for the guided sampling, where MatterGen generated BaBr3, Tm2Cd6,
HgNb and Re2Os4.

| | |
|---|---|
| Mixed scores of the guidance, and a step of the guided sampling given its scores and its noise | identical |
| Elements of the crystals MatterGen returns | identical |
| Network, conditioned structures without noise (12, 56 atoms): coordinates, lattice, elements, largest / mean difference with MatterGen in double precision | 1.5e-3 / 9.6e-5, 1.2e-2 / 2.4e-4, 5.8e-2 / 3.3e-3; MatterGen in single precision 1.5e-3 / 9.9e-5, 1.2e-2 / 2.6e-4, 9.5e-2 / 3.4e-3 |
| Network, conditioned structures with noise (14, 66 atoms): coordinates, lattice, elements, largest | 4.4e-4, 4.6e-4, 5.0e-3; MatterGen 6.6e-4, 4.6e-4, 8.7e-3 |

## Comparison with the original

Compared with MatterGen `9242366` run with torch 2.2.1, on the 40 structures of the tests
(225 atoms), whole: its loader, its denoiser and its sampler of `sampling_conf/default.yaml`,
with the mask of the elements its script of generation adds.

### The graph

On the 19 structures where the 50 nearest neighbours of each atom are defined, the graph is
the one of MatterGen: the same numbers of edges and of triplets, the same edges, the same
sums of lengths. On the 21 other ones, the 50th and the 51st neighbour of an atom are at the
same distance and the graphs differ by a few edges, see below.

### The network

**The network amplifies the rounding of its inputs a thousand times.** A length which
differs by its last bit changes its outputs by 1e-3: the gaussians of the lengths are 0.055
Å wide, and the trained projections of the bases cancel large terms. MatterGen in single
precision is itself up to 1.6e-2 from MatterGen in double precision. The tests check that
the rewrite is as near to MatterGen in double precision as MatterGen in single precision is
(mean difference at most 1.5 times, largest at most 3 times), on the structures where the
graph is defined.

| Clean structures at times from 0.001 to 1 (19 structures, 82 atoms) | Rewrite, MatterGen | Rewrite, MatterGen in double | MatterGen, MatterGen in double |
|---|---|---|---|
| Scores of the coordinates (values up to 0.27), largest / mean | 3.3e-4 / 1.3e-5 | 1.8e-3 / 7.1e-5 | 1.8e-3 / 7.1e-5 |
| Score of the lattice (values up to 5.6) | 3.6e-3 / 9.2e-5 | 1.6e-2 / 3.1e-4 | 1.6e-2 / 2.2e-4 |
| Scores of the elements (values up to 23) | 8.5e-2 / 1.7e-3 | 1.6e-1 / 3.3e-3 | 1.0e-1 / 3.0e-3 |

| Structures with the noise of their time (31 structures, 173 atoms, 118 atoms masked in all) | Rewrite, MatterGen | Rewrite, MatterGen in double | MatterGen, MatterGen in double |
|---|---|---|---|
| Scores of the coordinates (values up to 3.4) | 4.6e-2 / 8.1e-4 | 4.3e-2 / 8.1e-4 | 2.5e-2 / 6.4e-4 |
| Score of the lattice (values up to 2.5) | 3.0e-2 / 3.1e-4 | 1.4e-2 / 1.9e-4 | 2.4e-2 / 4.0e-4 |
| Scores of the elements (values up to 23) | 3.5e-1 / 4.8e-3 | 3.2e-1 / 4.9e-3 | 2.7e-1 / 3.6e-3 |

MatterGen does not run in double precision as it is: two casts of its graph to single
precision were changed for this run, see `scripts/reference_mattergen.py`.

The rewrite scales the lengths by the radius as MatterGen does, times its inverse. It
divided by the radius before, which is one bit away: the score of a lattice then differed
from the one of MatterGen by 6.2e-2. This is now how [GemNet-T](gemnet.md) does it for
CDVAE too.

### The sampling

The sampling of 4 crystals of 4, 8, 2 and 6 atoms was run whole, with what MatterGen draws
and the scores its network gives recorded at 10 steps (0, 1, 2, 100, 300, 500, 700, 900, 998,
999). Each step starts from what MatterGen had and is given its scores and its noise; the
elements drawn by MatterGen are selected by the Gumbel noise:

| A step of the sampling, given the scores and the noise of MatterGen | |
|---|---|
| Corrector: coordinates and lattice | identical |
| Probabilities of the elements at the step before (`element_posterior`) | identical |
| Predictor: coordinates and lattice | identical |
| Elements after the last step, the most probable ones | identical |

The scores of the rewrite on the crystals of the sampling are compared to the ones of
MatterGen where no crystal of the batch has a tie of its neighbours (2 of the 20 calls):
within 1e-2 of the largest value, 1e-4 on average. A sampling is never compared through a
seed, and whole samplings are not compared: one ulp of difference in the network, amplified
as above, makes the first steps diverge, when the coordinates move by several cells.

## Behaviours kept from MatterGen

- **The crystals of a batch are not independent.** The step of the Langevin corrector is
  computed from the mean, over the batch, of the norms of the scores and of the noises: a
  crystal is sampled differently with other crystals. The scores of the network do not
  depend on the batch.
- **The noise of the predictor of the lattice is one number per crystal.** MatterGen draws
  it in the shape of a coefficient, `(n, 1, 1)`, and makes it symmetric: every element of
  the diagonal is this number, every other one √2 times it. `StepNoise.predictor_lattice`
  has one value per crystal.
- **The graph depends on the order of the atoms.** It is made symmetric from the edges
  towards atoms of lower index, after the 50 nearest neighbours are kept: when an atom has
  more than 50 neighbours closer than 7 Å, which is the rule in these cells, permuting the
  atoms changes the graph. On a graph without this cut, the network follows a permutation.
- the time is continuous, from 1 to 0.001, and the step of the elements is the integer part
  of `999 t`;
- the elements which are not allowed get a score lowered by 1e10: the noble gases,
  technetium, promethium and everything after bismuth, and the mask. An atom always ends
  with an element of the 76 others;
- the embedding of the time has the frequencies stored in the weights, which differ from the
  ones computed from the configuration by one bit for 13 of 256;
- the lower bound of `alpha_t / alpha_s`, 0.001, and the `1e-20` added to the probabilities
  of the elements before their logarithm;
- the lattice is a symmetric matrix, not reduced; the coordinates are wrapped in the cell
  after each step.

## Invariances

The scores of the coordinates and of the elements do not change with a rotation of the
crystal or a translation of its atoms, and the score of the lattice turns with it, `R^T S R`:
these are tested. They follow a permutation of the atoms on a graph without cut of the
neighbours, see above.

## Differences with MatterGen

- the noise of a step is drawn in another order, and the elements with Gumbel noise rather
  than with `torch.distributions.Categorical`: the samples are not the ones of MatterGen for
  the same seed, their distribution is. MatterGen also draws a sample of the elements from
  the scores of the network before the one it keeps; it is not drawn here;
- among neighbours at the same distance, the 50 nearest are not always the ones of
  MatterGen, which keeps one or the other from the rounding of their distances. In a
  crystal, the two images of an atom on each side of it are always at the same distance:
  during the sampling, about one batch of crystals in two has such a tie at a step;
- two atoms closer than 0.01 Å are an edge here, and give NaN; MatterGen leaves them out;
- the images of the radius graph are bounded to 5 cells along each lattice vector before
  they are listed (`max_image_shift`), as in MatterGen: the sampling meets flat cells, which
  have thousands of images within 7 Å;
- the graph is built by `torchms.graph`, without the padded matrices of MatterGen.
