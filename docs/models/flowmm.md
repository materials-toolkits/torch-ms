# FlowMM

FlowMM is the Riemannian flow matching model of crystals of Miller, Chen, Sriram and Wood
([flowmm](https://github.com/facebookresearch/flowmm), commit `6a96aec`, CC BY-NC 4.0,
[arXiv:2406.04713](https://arxiv.org/abs/2406.04713)), in `torchms/models/flowmm/`. Its de
novo generation is ported: the elements, the coordinates and the cell of crystals are
generated together, from their numbers of atoms.

| | `FlowMM` |
|---|---|
| Does | generates crystals of given numbers of atoms, with their elements |
| Source | [flowmm](https://github.com/facebookresearch/flowmm), commit `6a96aec`, `model/arch.py`, `model/model_pl.py`, `rfm/` |
| License | **CC BY-NC 4.0**: no commercial use |
| Paper | [arXiv:2406.04713](https://arxiv.org/abs/2406.04713), ICML 2024 |
| Weights | `flowmm-mp20-essflow`, **third party**: trained on MP-20 by the authors of ESS-Flow, 226 MB |
| Transform | `AddCompleteGraph(self_loops=False)` |
| Largest difference with the original | 1.9e-5 on the velocity of the lattice (values up to 75), 2.9e-6 on the one of the bits (values up to 2.1); the bits near the time 1 as near to FlowMM in double precision as FlowMM |

```{.python notest}
import torch

from torchms.models import FlowMM

model = FlowMM.from_pretrained(device="cuda")

# 64 crystals, with numbers of atoms as the ones of MP-20
num_atoms = model.sample_num_atoms(64)
structures = model.sample(num_atoms)          # 1000 Euler steps, 1000 calls of the network
structures.lattice                            # (64, 3, 3), in Å
structures.z                                  # from 1 to 128: read the warning below
```

**The weights are not the authors'.** The authors of FlowMM publish no weights: they write
in the issues 8, 13 and 15 of their repository that they are not allowed to. The weights of
`config.yaml` are the ones of the authors of
[ESS-Flow](https://github.com/Adhithyan8/essflow), who trained FlowMM themselves on MP-20
with its default configuration of de novo generation (`data=mp_20 model=abits_params`, 2000
epochs planned, the checkpoint of the epoch 1894), and use it as the prior of their paper.
They publish it on [Hugging Face](https://huggingface.co/Adhi94/essflow-flowmm-mp20) under
CC BY-NC 4.0. How good these weights are was not measured here, and it is not the quality
the paper of FlowMM reports.

With a seed of 0, six crystals drawn have between 9 and 20 atoms and the compositions
La7CeSi12, Na4V3O7, Cs2AlF6, Cs2VGe6S11, LiV2F9 and Li5Cu3O8. These are single draws.

## The rewrite

A crystal is a point of three manifolds: the element of each atom as 7 analog bits, the
bits of its atomic number minus one at -1 or +1; its fractional coordinates on a flat
torus; and its cell as three lengths in Å and three angles, which are kept between 59.9 and
120.1 degrees by a sigmoid from an unconstrained space where the flow moves them. A `State`
holds the three. `state_of` gives the point of structures, `to_structures` the structures
of a point: an atomic number is read from the bits which are positive.

A CSPNet gives the velocity of the three at a time t. It is the network of DiffCSP
(`torchms.models.layers.cspnet`, whose layer it extends) with what FlowMM adds: the
difference of the coordinates of an edge is the shortest one on the torus, the edge also
reads its cosines with the lattice vectors and an embedding of the number of atoms of the
structure, the cell is read as its lattice parameters, the time through a linear map, and
the cell is predicted from the mean and the sum of the features of the atoms. The inputs
and the outputs are standardized with statistics of the training set, which are buffers of
the weights. The velocity of the coordinates has no drift: its mean over the atoms of a
structure is removed.

`sample` draws the initial state (normal bits, uniform coordinates, lengths of a log-normal
distribution with the statistics of MP-20, uniform angles), and makes 1000 Euler steps from
the time 0 to 1, the coordinates wrapped in the cell after each one (`integrate`).
`sample_step` is one step, `velocity` the network on a state. The graph is complete,
without self loops, and only depends on the numbers of atoms: the sampling builds it once.
`sample_num_atoms` draws them as in MP-20, from 1 to 20.

**Annealing.** FlowMM can multiply the velocity of the coordinates by
`1 + slope * max(t - offset, 0)` during the sampling (`--inference_anneal_slope` of its
script of evaluation). The default is no annealing, the value of the configuration of
these weights; `sample`, `integrate`, `sample_step` and
`velocity` take both values.

**The moving average.** The checkpoint has the parameters of the network and their moving
average, which FlowMM puts in the network whenever it is evaluated (`manifm.ema.EMA` in
evaluation mode): the rewrite loads the average. The checkpoint only holds tensors and is
read with `weights_only=True`; the configuration of the training, the second file, is
checked against `config.yaml`.

Not included: the losses, the training, the likelihood, the other manifolds of FlowMM (the
simplex of DiffCSP, symmetric positive matrices, lattice matrices, a first atom fixed at the
origin), the prediction of a structure from its composition (`null_params`), the
self-conditioning, the graph of the nearest neighbours, FlowLLM and the metrics.

## Comparison with the original

FlowMM `6a96aec` was run with torch 2.1.0 on the CPU, loaded with its
`flowmm.model.eval_utils.load_model`, on the 40 crystals of the tests (225 atoms), each one
at its own time from 0 to 0.95, and its sampling `gen_sample` was run whole for 6 crystals
of 1 to 20 atoms (51 atoms), with its initial state recorded. Each step below starts from
the state FlowMM had at this step. A sampling is never compared through a seed.

| Weights `flowmm-mp20-essflow` | Largest | Mean |
|---|---|---|
| Analog bits and coordinates of the 40 crystals | identical | |
| Lattice matrices from the lattice parameters of FlowMM (values up to 13.5) | identical | |
| Velocity on the 40 crystals: bits (values up to 2.1) | 2.9e-6 | 1.9e-7 |
| The same: coordinates (values up to 0.42) | 3.9e-7 | 4.0e-8 |
| The same: lattice parameters (values up to 75) | 1.9e-5 | 6.3e-7 |
| Steps 1, 2, 3 and 501: velocity of the bits, coordinates, lattice (values up to 3.9, 0.5, 7.5) | 1.6e-6, 1.9e-7, 1.2e-6 | |
| Steps 999 and 1000: velocity of the coordinates and of the lattice (values up to 0.70 and 6.8) | 1.5e-6, 7.6e-6 | |
| Steps 999 and 1000: velocity of the bits (values up to 4.0) | 7.4e-5 | |
| The 1000 steps chained from the initial state of FlowMM: bits, coordinates, lattice | 2.4e-7, 3.6e-7, 4.9e-6 | 3.4e-8, 5.4e-8, 5.2e-7 |
| Elements of the 51 atoms after the 1000 steps | identical | |

A change of 0.1 % of one tensor of weights or of statistics makes the comparisons fail. The
tests check the crystals and the step 501 by default, every recorded step and the whole
sampling with `--full`.

### Near the time 1

**The network multiplies the rounding of float32 near the time 1.** At the steps 999 and
1000 the velocity of the bits differs from FlowMM by 7.4e-5, 2e-5 of its largest value.
FlowMM in double precision on the same states tells that this is rounding: FlowMM in single
precision is as far from it as the rewrite.

| Velocity of the bits | FlowMM, FlowMM in double | Rewrite, FlowMM in double | Rewrite, FlowMM |
|---|---|---|---|
| Step 501 | 8.7e-6 | 7.5e-6 | 1.6e-6 |
| Step 999 | 9.0e-5 | 9.2e-5 | 4.9e-5 |
| Step 1000 | 1.6e-4 | 1.8e-4 | 7.4e-5 |

The tests hold the bits of these two steps to be as near to FlowMM in double precision as
FlowMM is (`as_near_as_original`), and everything else to 1e-5 of its largest value. FlowMM
keeps the frequencies of its Fourier basis and the bounds of the angles in float32 when it
is run in double precision, so its double precision is not exact either: the rewrite in
double is 5e-6 from it on these steps.

### Differences of exactly half a cell

The difference of the coordinates of an edge is the shortest one on the torus, between -0.5
and 0.5: an arc tangent of a sine and a cosine. At exactly half a cell, the sign is the one
of the rounding of the sine of pi. The cosines of the edge with the lattice vectors change
sign with it, the Fourier basis does not. In crystals of high symmetry (a perovskite, rock
salt) many atoms are half a cell apart: FlowMM in single and in double precision differ by
up to 0.38 on the velocity of the bits of these crystals. The rewrite computes the
difference as FlowMM does and agrees with it in float32 to 2.9e-6 on the 40 crystals of the
tests, but a structure with such differences has no velocity better defined than this.

## Behaviours kept from FlowMM

- the analog bits have 7 bits: an atomic number from 1 to 128 can be generated, and the
  ones after 118 are not elements. FlowMM does not refuse them, nor does the rewrite;
- the cosines of an edge with the lattice vectors are normalized as the three of the edge,
  not each one by the lengths of its two vectors;
- an angle outside of [59.9, 120.1] degrees, in a crystal given to `state_of`, is clamped by
  the sigmoid to a very large unconstrained value (-87 for 160 degrees), as in FlowMM: the
  network was not trained on such cells;
- the embedding of the number of atoms has 100 entries: a structure of 100 atoms or more is
  refused by `one_hot`;
- the bits are not clamped during the sampling (FlowMM only clamps them with
  self-conditioning, which these weights do not use).

## Differences with FlowMM

- the base distribution is drawn by torch with a generator, in another order than the
  manifolds of geoopt draw it: the samples are not the ones of FlowMM for the same seed,
  their distribution is;
- the padded "flat" representation of FlowMM and one manifold per structure are replaced by
  a `State` of the atoms and of the cells, without loop over the structures;
- the lattice parameters of a crystal given to `state_of` are computed by
  `torchms.graph.geometry.lattice_parameters`: they agree with the ones of FlowMM to 7e-5
  on unconstrained angles up to 87.
