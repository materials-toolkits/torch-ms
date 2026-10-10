# DiffCSP

DiffCSP is the diffusion model of crystals of Jiao, Huang, Lin, Han, Chen, Lu and Liu
([DiffCSP](https://github.com/jiaor17/DiffCSP), commit `7121d15`, MIT,
[arXiv:2309.04475](https://arxiv.org/abs/2309.04475)), in `torchms/models/diffcsp/`. It has
two models, with two sets of weights:

- `DiffCSP` gives a lattice and fractional coordinates to atoms of given elements: the
  prediction of a structure from its composition;
- `DiffCSPGeneration` also generates the elements: it is only given the number of atoms
  of each structure.

| | `DiffCSP` | `DiffCSPGeneration` |
|---|---|---|
| Does | samples the lattice and the coordinates of a crystal from its composition | samples the elements, the lattice and the coordinates of a crystal from its number of atoms |
| Source | [DiffCSP](https://github.com/jiaor17/DiffCSP), commit `7121d15`, `diffusion.py` | the same, `diffusion_w_type.py` |
| License | MIT | MIT |
| Paper | [arXiv:2309.04475](https://arxiv.org/abs/2309.04475), NeurIPS 2023 | the same |
| Weights | `diffcsp-mp20`, of the authors (`mp_csp`), trained on MP-20, 47 MB | `diffcsp-gen-mp20`, of the authors (`mp_gen`), trained on MP-20, 47 MB |
| Transform | `AddCompleteGraph(self_loops=True)` | the same |
| Largest difference with the original | 7.6e-6 on the noise of the lattice (values up to 18), 4.6e-6 on the scores | 5.7e-6 on the noise of the lattice (values up to 20), 4.8e-6 on the noise of the elements |

```{.python notest}
import torch

from torchms.models import DiffCSP, DiffCSPGeneration

# from a composition: NaCl and SrTiO3, the atoms of a structure together
model = DiffCSP.from_pretrained(device="cuda")
z = torch.tensor([11, 17, 38, 22, 8, 8, 8], device="cuda")
num_atoms = torch.tensor([2, 5], device="cuda")

structures = model.sample(z, num_atoms)       # 1000 steps, 2000 calls of the network
structures.lattice                            # (2, 3, 3), in Å
structures.x                                  # (7, 3), fractional coordinates

# without composition: 64 structures, with numbers of atoms as the ones of MP-20
generation = DiffCSPGeneration.from_pretrained(device="cuda")
num_atoms = generation.sample_num_atoms(64)   # or any numbers of atoms

structures = generation.sample(num_atoms)
structures.z                                  # atomic numbers
```

With the weights of the first model and a seed of 0, the cell of NaCl has three vectors of
3.99 Å and a volume of 45.0 Å³, the one of SrTiO3 three vectors of 3.94 Å with the oxygens
on the faces of the cube: the primitive cells of rock salt and of a perovskite. With the
weights of the second one, six structures drawn with a seed of 0 have between 9 and 20
atoms and the compositions Li3Ti2Ni5O10, HgSb2Se4, LaNdAl4Cu3, KMnF3, Cs3PdO2 and TiNi7O8.
These are single draws: the accuracy of the models is the one of their paper, it was not
measured here.

## The rewrite

A structure is a lattice, a 3 x 3 matrix, and the fractional coordinates of its atoms.
Both get noise: the lattice a normal noise with a cosine schedule, the coordinates a noise
wrapped around the cell, whose level grows geometrically from 0.005 to 0.5 cell. A network,
[CSPNet](cspnet.md), reads the noisy structure and the time and gives the noise of the
lattice and the score of the coordinates.

`sample` starts from a normal lattice and uniform coordinates and makes the 1000 steps of
the diffusion, from the last time. `sample_step` is one of them, with its noise as
arguments: a Langevin correction of the coordinates, then the prediction of the lattice and
of the coordinates of the time before. `model(structures, times)` is the network alone.

**With the elements.** In `DiffCSPGeneration` the element of an atom is a vector of
100 values, one per element from hydrogen: 1 for its element and 0 elsewhere
(`element_vectors`). This vector gets the noise of the lattice, with the same schedule, the
network reads it in place of the element and also gives its noise. The vectors start from
normal noise and are predicted at each step as the lattice is; the element of an atom is
the largest value of its vector (`atomic_numbers`). The atomic numbers of the structures
given to this model are not read: `model(structures, elements, times)` takes the vectors,
and `sample_step` takes and returns them next to the structures, whose atomic numbers are
read from the vectors returned. The noise of a step is a `GenerationNoise`.

**The number of atoms is not generated.** DiffCSP draws it before the sampling from the
numbers of atoms of the training set, a table written in its `scripts/generation.py`.
`sample_num_atoms` draws from the table of MP-20, which is in `config.yaml`: from 1 to 20
atoms. `sample` takes any numbers of atoms.

Not included: the losses and the training, the model of the energy and the optimization of
a property, the graph of the nearest neighbours of CSPNet (`edge_style: knn`, used by no
published weights), the metrics.

The authors publish six checkpoints in a folder of Google Drive: `mp_csp`, `mpts_csp`,
`perov_csp` and `carbon_csp_and_gen` for the prediction of a structure, `mp_gen` and
`perov_gen` for the generation. The two of MP-20 are ported. A file of Google Drive can be
replaced by its owner under the same link: the SHA-256 of `config.yaml` tells if it was.
The checkpoints only hold tensors and are read with `weights_only=True`. Each model checks
the configuration stored with its weights and refuses the weights of the other one.

The graph is complete: an edge between every pair of atoms of a structure, and from each
atom to itself. It has no radius and no periodic image, since the network reads the
difference of the fractional coordinates modulo 1. It only depends on the numbers of atoms.
A network call costs the square of the number of atoms of each structure.

## Comparison with the original

Both models are compared with DiffCSP `7121d15` run with torch 1.9.0, on the 40 structures
of the tests (225 atoms, 1,971 edges). The sampling of DiffCSP was run whole, with
everything it draws recorded. Each step below starts from what DiffCSP had at this time and
adds the noise it drew: a sampling is never compared through a seed.

### From a composition

| `DiffCSP`, weights `diffcsp-mp20` | |
|---|---|
| Schedules of the noise computed from `config.yaml`, against the ones of the weights | identical, up to 6e-8 |
| Noise of the lattice, structures as they are, times from 1 to 1000: largest difference (values up to 17.9) | 7.6e-6 |
| The same, mean difference | 2.3e-7 |
| Scores of the coordinates, largest difference (values up to 3.4) | 3.0e-6 |
| The same, mean difference | 1.3e-7 |
| Noise of the lattice on structures with the noise of their time, largest difference (values up to 17.8) | 4.8e-6 |
| Scores of the coordinates on the same, largest and mean difference (values up to 3.7) | 4.6e-6, 2.2e-7 |
| A step of the sampling at the times 999, 998, 750, 500, 250, 100, 10, 2, 1 (6 structures, 36 atoms): lattice, largest difference | 7.2e-7 Å |
| The same at the time 1000 (values up to 4.0) | 2.4e-5 Å |
| The same ten steps: coordinates, largest difference | 1.2e-7 of the cell |
| Three steps in a row from the time 1000: lattice and coordinates | 7.4e-6 Å, 6.0e-8 of the cell |

### With the elements

| `DiffCSPGeneration`, weights `diffcsp-gen-mp20` | |
|---|---|
| Schedules of the noise | the ones of the first model |
| Noise of the lattice on structures with the noise of their time, times from 1 to 1000: largest and mean difference (values up to 20.2) | 5.7e-6, 2.3e-7 |
| Scores of the coordinates on the same, largest and mean difference (values up to 3.0) | 1.6e-6, 1.4e-7 |
| Noise of the elements on the same, largest and mean difference (values up to 6.3) | 4.8e-6, 2.2e-7 |
| A step of the sampling at the times 999, 998, 500, 1 (4 structures, 14 atoms): lattice and elements, largest difference | 1.4e-6 Å, 1.4e-6 |
| The same at the time 1000 (values up to 4.9 and 4.2) | 4.8e-5 Å, 9.5e-5 |
| The same five steps: coordinates, largest difference | 6.0e-8 of the cell |
| Two steps in a row from the time 999: lattice, elements and coordinates | 9.5e-7 Å, 1.4e-6, 6.0e-8 of the cell |
| Elements of the 14 atoms after the last step | identical |

### The first step

**The first step multiplies the rounding by 100.** At the time 1000 what gets normal noise
is predicted as `(values - c noise) / sqrt(alpha)` with `1 / sqrt(alpha) = 100`: a
difference of 2.4e-7 between the two networks becomes 2.4e-5 Å on the lattice. The step of
the rewrite in double precision tells that it is rounding: both codes in single precision
are as far from it as from each other.

| At the time 1000 | Rewrite, DiffCSP | Rewrite in double, DiffCSP | Rewrite in double, rewrite |
|---|---|---|---|
| Lattice, from a composition | 2.4e-5 Å | 5.3e-5 Å | 5.3e-5 Å |
| Lattice, with the elements | 4.8e-5 Å | 7.4e-5 Å | 5.3e-5 Å |
| Elements | 9.5e-5 | 1.5e-4 | 1.3e-4 |

The tests of the generation compare this step with a tolerance 100 times larger, the factor
of the step, and the other ones with 1e-5 of the largest value. DiffCSP itself was not run
in double precision.

## Behaviours kept from DiffCSP

- the normalization of the scores, `sigma_scheduler.sigmas_norm`, is estimated by DiffCSP
  from 10,000 random draws each time a model is created, and saved with the weights: the
  two sets of weights have different values, within a few percent. The rewrite reads the
  values of the weights and does not estimate them: a model without weights has ones;
- the embedding of the elements has 100 rows read from hydrogen: an element after fermium
  is refused;
- the lattice is a free 3 x 3 matrix. The one sampled is not reduced, can be left-handed,
  and nothing keeps it away from a flat cell;
- the coordinates of the corrector are not wrapped in the cell before the predictor, only
  the ones a step returns;
- the last step, at the time 1, adds no noise;
- the step of the corrector, `step_size`, is the one the scripts of DiffCSP use for MP-20:
  1e-5 from a composition, 5e-6 with the elements;
- with the elements, nothing makes the vectors of two atoms agree or the composition
  neutral: the elements are the largest values, atom by atom;
- the checkpoint `mp_gen` has a second schedule of noise, `type_sigma_scheduler`, which the
  code of DiffCSP does not have: it is not loaded.

## Invariances

The network reads fractional coordinates and the inner products of the lattice vectors. Its
outputs do not change with a translation of the atoms, the scores and the noise of the
elements follow a permutation of the atoms, and the noise of the lattice is rotated with
the lattice: these are tested. A change of basis of the lattice, which changes the
fractional coordinates, changes the outputs: the models are not invariant to the choice of
the cell.

## Differences with DiffCSP

- the noise of a step is drawn in another order, and the noise DiffCSP draws for its
  corrector and does not use (of the lattice, and of the elements) is not drawn: the
  samples are not the ones of DiffCSP for the same seed, their distribution is;
- the numbers of atoms are drawn by torch, not by numpy;
- the complete graph is built without a loop over the structures and without a dense
  matrix of the batch;
- the basis of the differences of the coordinates is computed once per call of the
  network, not in each of its layers.
