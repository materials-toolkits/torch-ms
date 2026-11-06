# DiffCSP++

DiffCSP++ is the diffusion model of crystals of a space group of Jiao, Huang, Liu, Zhao and
Liu ([DiffCSP-PP](https://github.com/jiaor17/DiffCSP-PP), commit `e82e7ff`, MIT,
[Space Group Constrained Crystal Generation](https://openreview.net/forum?id=jkvZ7v4OmP),
ICLR 2024), in `torchms/models/diffcsp_pp/`. It is the diffusion of [DiffCSP](diffcsp.md),
whose atoms are placed on the Wyckoff positions of a space group. Two models, with six sets
of weights of the authors:

- `DiffCSPPlusPlus` gives a lattice and fractional coordinates to atoms of given elements
  on given Wyckoff positions;
- `DiffCSPPlusPlusGeneration` also generates the element of each site: it is only given the
  space group and the Wyckoff positions.

| | `DiffCSPPlusPlus` | `DiffCSPPlusPlusGeneration` |
|---|---|---|
| Does | samples the lattice and the coordinates of a crystal of a space group, for atoms of given elements on given Wyckoff positions | samples the elements, the lattice and the coordinates of a crystal of a space group on given Wyckoff positions |
| Source | [DiffCSP-PP](https://github.com/jiaor17/DiffCSP-PP), commit `e82e7ff`, `diffusion.py`, `lattice/crystal_family.py` | the same, `diffusion_w_type.py` |
| License | MIT | MIT |
| Paper | [ICLR 2024](https://openreview.net/forum?id=jkvZ7v4OmP) | the same |
| Weights | `diffcsp-pp-mp20` (default), `diffcsp-pp-mpts52`, `diffcsp-pp-perov5`, `diffcsp-pp-carbon24`, official, 49 MB each | `diffcsp-pp-gen-mp20` (default), `diffcsp-pp-gen-perov5`, official, 49 MB each |
| Transform | `AddCompleteGraph(self_loops=True)`, built by `sample` | the same |
| Largest difference with the original | 2.3e-6 on the noise of the lattice (values up to 12), 1.9e-6 on the scores; 6.4e-6 of a step | 9.6e-6 on the noise of the elements (values up to 9.2), 2.7e-6 on the scores; 1.9e-6 of a step |

```{.python notest}
import torch

from torchms.data.symmetry import WyckoffSites
from torchms.models import DiffCSPPlusPlus, DiffCSPPlusPlusGeneration

# the sites of each structure: its space group and the Wyckoff letter of each site.
# LiMnO2 in Pnnm (2a, 2d, 4g), NaCl in Fm-3m (4a, 4b), SrTiO3 in Pm-3m (1a, 1b, 3c)
sites = WyckoffSites.from_letters(
    [58, 225, 221], [["2a", "2d", "4g"], "ab", "abc"], device="cuda"
)
z = torch.tensor([25, 3, 8, 11, 17, 38, 22, 8], device="cuda")   # one element per site

model = DiffCSPPlusPlus.from_pretrained(device="cuda")
structures = model.sample(sites, z)        # 999 steps, 1998 calls of the network
structures.lattice                         # (3, 3, 3), symmetric matrices, in Å
structures.x                               # (21, 3), fractional coordinates

# without elements: the element of each site is generated
generation = DiffCSPPlusPlusGeneration.from_pretrained(device="cuda")
structures = generation.sample(sites)
structures.z                               # one element per site
```

With the default weights and a seed of 0, the rock salt is a cube of 5.75 Å, its atoms at
the corners and the centres of the faces, and SrTiO3 a cube of 3.90 Å, the perovskite (3.905
Å measured). With the weights of the generation and the same sites, the elements drawn are
PuSe in the rock salt and RbPdBr3 in the perovskite. These are single draws: the accuracy of
the models is the one of their paper, it was not measured here.

## Wyckoff positions

The Wyckoff positions of the 230 space groups are a table of torchms,
`torchms/data/constants/wyckoff.npz` (33 kB), read by
`torchms.data.constants.get_wyckoff_positions`: for each position its letter, its
multiplicity and the affine operation of each atom of an orbit, as 4 x 4 matrices. It is
written once by `scripts/update_wyckoff.py` from pyxtal 0.6.7 (MIT), which DiffCSP++ reads
the positions from, in its order and its settings: the standard ones, with the origin 2 for
the 24 groups with two origins. The tests check it against spglib: the general position of
each group is the group of spglib in this setting, and an atom on each position, with an
atom on the general position, is found by spglib in the same group and on a position of the
same multiplicity.

`WyckoffSites` (`torchms.data.symmetry`) holds the sites of a batch: the space group of each
structure, the operation and the site of each atom, with the pseudo-inverse of each linear
part. `from_letters` builds it from space groups and Wyckoff letters (`"adg"` or
`["2a", "2d", "4g"]`, as DiffCSP++). Its methods name the moves of the sampling: `place`
puts each atom at the image of the first atom of its site, `spread` sends a displacement of
the first atom to the others, `symmetrize` averages vectors over a site at its free point,
`mean_over_sites` and `unpool_sites` go from the atoms to the sites and back.

## The rewrite

A structure is a lattice and the fractional coordinates of its atoms. The lattice is six
numbers k: it is exp(sum k_i B_i), for six symmetric matrices B_i of norm 1, three off the
diagonal, two of the shape and one of the volume (`CrystalFamily`). The crystal family of
the space group fixes some numbers: the first three for an orthorhombic group, all but the
volume for a cubic one, and for a hexagonal one k_0 = -ln(3) sqrt(2) / 4, a cell of
a = b and gamma = 120°. The coordinates are the ones of the free point of each site: only
they move, and each atom is placed at its image after every move.

The schedules of the noise are the ones of DiffCSP (`torchms.models.layers.schedules`): the
numbers k get a normal noise with a cosine schedule, the coordinates a noise wrapped around
the cell from 0.005 to 0.5 cell. A [CSPNet](cspnet.md) reads k as it is
(`lattice_dim: 6`) and gives the noise of k and the score of the coordinates of each atom.

`sample` draws k and uniform coordinates, makes 999 steps from the time 999 to 1, and
returns the symmetric lattice of k. A step corrects the coordinates by a Langevin move, then
predicts k and the coordinates of the time before. The noise and the scores of the
coordinates are brought back to the free point of each site by the pseudo-inverse of their
operation, averaged over the site, and sent back to its atoms; k is projected on the family
after each step. `sample_step` is one step, `denoise` the network, `integrate` every step
from a given state; the noise of a step is a `Noise`, drawn for every atom, of which only
the one of the first atom of each site is read.

**With the elements.** In `DiffCSPPlusPlusGeneration` the element of an atom is a vector of
100 values, which gets the noise of k. The atoms of a site share the vector of the first
one, and the network gives the noise of each atom, averaged over the site. The element of a
site is the largest value at the end.

**Step of the corrector.** The step of the Langevin move at a time t is the step size over
the mean squared score of the noise at t times 0.005², not the step size times
(σ_t / 0.005)² of DiffCSP. The step size of each set of weights is the one DiffCSP++
recommends for its dataset and task (`recommand_step_lr` of `scripts/eval_utils.py`): 1e-5
for MP-20 and MPTS-52, 5e-7 for Perov-5, 5e-6 and 1e-6 for the generation on MP-20 and
Perov-5. `sample(step_size=...)` takes another one.

**Weights.** The six checkpoints of the authors are on Google Drive, each with the
configuration of its training (`hparams.yaml`), downloaded from the identifier of their
file and checked against their SHA-256. They only hold tensors. The configuration is
checked against `config.yaml`: the weights of one model are refused by the other one. The
folder of the authors names `carbon_gen` a model of the generation on Carbon-24: its network
and its configuration are the ones of a prediction from the elements, and the authors
generate carbon with it, every site being carbon. It is `diffcsp-pp-carbon24`, a set of
weights of `DiffCSPPlusPlus`, with the step size of `scripts/generation.py`, 1e-5.

Not included: the losses and the training, the sampling from a noisy structure
(`diff_ratio`), the search of templates of Wyckoff positions (CSPML), the distribution of
the space groups and of the Wyckoff positions of a dataset (the generation of DiffCSP++
takes the ones of its training set), the datasets and the metrics.

## Comparison with the original

DiffCSP++ `e82e7ff` was run with torch 2.1.2, PyG 2.5.2 and pyxtal 0.6.7, on the 40
crystals of the tests (225 atoms) at times from 1 to 1000, as they are and with the noise
of their time, and its sampling `CSPDiffusion.sample` was run for eight structures of
`sample_api.get_data_from_syminfo`, one or more of each crystal family (4 to 12 atoms), with
the states of ten steps and their noise recorded. For the models given the elements, the
whole sampling of a monoclinic and a triclinic structure was recorded with every draw it
uses, in single and in double precision. A sampling is never compared through a seed.

| Largest difference, against DiffCSP++ | Network on 40 crystals: k, scores (and elements) | A step, relative | Whole sampling of k, monoclinic and triclinic (DiffCSP++ single against double) |
|---|---|---|---|
| `diffcsp-pp-mp20` | 2.3e-6, 1.9e-6 | 6.4e-6 | 2.8e-4 (6.2e-5), 0.14 (0.25) |
| `diffcsp-pp-mpts52` | 1.4e-6, 8.9e-7 | 3.6e-6 | 7.1e-5 (1.0e-4), 0.24 (0.47) |
| `diffcsp-pp-perov5` | 9.5e-7, 1.1e-6 | 3.6e-8 | 6.1e-5 (1.7e-3), 4.7e-2 (4.9e-2) |
| `diffcsp-pp-carbon24` | 9.5e-7, 1.2e-6 | 6.4e-6 | 1.6e-3 (5.0e-6), 0.76 (0.38) |
| `diffcsp-pp-gen-mp20` | 1.2e-6, 1.1e-6, 9.6e-6 | 1.9e-6 | not recorded |
| `diffcsp-pp-gen-perov5` | 1.4e-6, 2.7e-6, 4.3e-6 | 3.8e-7 | not recorded |

The values of the network are up to 12 for k, 4.3 for the scores and 9.2 for the elements.
The numbers k of the 40 cells agree to 3.8e-6. A step is compared from the state DiffCSP++
had before it, with its noise, at ten times from 999 to 1 (five for the generation).
The model of the generation of Perov-5, given monoclinic and triclinic sites unlike the
perovskites it was trained on, ends at numbers k of 100 to 1700 in DiffCSP++ as in the
rewrite.

The operations and the pseudo-inverses of the sites built from the table are the ones
DiffCSP++ builds with pyxtal, bit for bit; so are the masks, the biases and the basis of the
crystal families built from `config.yaml`, against the ones of the checkpoints. A change of
0.1 % of a tensor of weights makes the comparisons of the network fail, but for some biases
of the layers (eight of `diffcsp-pp-mp20`, two of `diffcsp-pp-gen-mp20`), small values which
a change of 1 % shows, and 10 % for the last bias of the last layer.

### The sampling multiplies the rounding

**The first steps multiply a difference by about 50 each.** At the time 999 the betas of
the cosine schedule are clipped at 0.9999, and a step divides k by the square root of
alpha_t (0.25 at the time 999): a relative change of 1e-7 of k at the time 999 moves it by
1.4e-2 three steps later, with the weights of MP-20. The tests compare every recorded step
from the state DiffCSP++ had, and chain the last two steps, which give the result of
DiffCSP++.

**The whole sampling is chaotic, even in double precision.** In double precision a relative
change of 1e-12 of the first k moves the last one by 5e-5 for the monoclinic structure and
by 0.11 for the triclinic one; DiffCSP++ in single and in double precision end 6e-5 and 0.25
apart. A sampling can also end at one of two results 1.6e-3 apart (Carbon-24, monoclinic):
the rewrite ends at the other one, and every change of 1e-6 of its start brings it to the one
of DiffCSP++. The tests hold the whole sampling of the rewrite, structure by structure, to
three times the largest of the gap of DiffCSP++ between single and double precision and of
the moves of changes of ±1e-6 of its first k.

## Behaviours kept from DiffCSP++

- the sampling starts at the time 999, not at the last time of its schedules, 1000: 999
  steps;
- the corrector reads the step size over the mean squared score of the noise, which
  DiffCSP++ estimates from random draws when a model is created and which is read from the
  weights;
- the noise and the scores of the coordinates are those of the first atom of each site,
  brought to its free point, sent back by the linear part of the operation of each atom;
- the noise of the lattice is drawn at the last step too, with no weight;
- the network reads k without the constraints of the space group and its noise is not
  projected: only the result of a step is;
- a lattice is a symmetric matrix, of the lengths and the angles of a crystal in another
  orientation;
- the Wyckoff positions are the ones of pyxtal, in its settings and its order.

## Differences with DiffCSP++

- the noise is drawn by torch with a generator, in the order of DiffCSP++: the samples are
  not the ones of DiffCSP++ for the same seed, their distribution is;
- the complete graph is built without a loop over the structures, once per sampling;
- the sites are given by `WyckoffSites`, the elements by site, and the Wyckoff positions are
  read from the table of torchms, without pyxtal;
- the pseudo-inverses of the operations are computed in the precision of the operations,
  float32 by default, as DiffCSP++.
