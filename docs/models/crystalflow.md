# CrystalFlow

CrystalFlow is the flow matching model of crystals of Luo, Wang, Wang, Shao, Lv, Wang, Wang
and Ma ([CrystalFlow](https://github.com/ixsluo/CrystalFlow), tag `v1.0.0-alpha.1`, MIT,
[Nature Communications 2025](https://doi.org/10.1038/s41467-025-64364-4),
[arXiv:2412.11693](https://arxiv.org/abs/2412.11693)), in `torchms/models/crystalflow/`. It
is built on DiffCSP. Two models, with five sets of weights of the authors:

- `CrystalFlow` gives a lattice and fractional coordinates to atoms of given elements: the
  prediction of a structure from its composition (CSP);
- `CrystalFlowGeneration` also generates the elements: it is only given the number of atoms
  of each structure (DNG).

| | `CrystalFlow` | `CrystalFlowGeneration` |
|---|---|---|
| Does | samples the lattice and the coordinates of a crystal from its composition | samples the elements, the lattice and the coordinates of a crystal from its number of atoms |
| Source | [CrystalFlow](https://github.com/ixsluo/CrystalFlow), tag `v1.0.0-alpha.1` (`d77d6d0`), `flow.py`, `cspnet.py` | the same, with `type_module.py` |
| License | MIT | MIT |
| Paper | [arXiv:2412.11693](https://arxiv.org/abs/2412.11693), Nature Communications 2025 | the same |
| Weights | `crystalflow-csp-mp20` (default), `crystalflow-csp-mpts52`, `crystalflow-csp-calypso60-pressure`, official | `crystalflow-dng-mp20` (default), `crystalflow-dng-mp20-formation-energy`, official |
| Transform | `AddCompleteGraph(self_loops=True)` | the same |
| Largest difference with the original | 1.6e-6 on the network (values up to 1.5), 3.2e-6 of a step | 2.5e-4 on the velocity of the elements of a perovskite at the time 1, as near to CrystalFlow in double precision as CrystalFlow; 1.4e-5 elsewhere |

```{.python notest}
import torch

from torchms.models import CrystalFlow, CrystalFlowGeneration

# from a composition: NaCl and SrTiO3, the atoms of a structure together
model = CrystalFlow.from_pretrained(device="cuda")
z = torch.tensor([11, 17, 38, 22, 8, 8, 8], device="cuda")
num_atoms = torch.tensor([2, 5], device="cuda")
structures = model.sample(z, num_atoms)            # 100 Euler steps, 100 calls of the network

# under pressure, in GPa
model = CrystalFlow.from_pretrained("crystalflow-csp-calypso60-pressure", device="cuda")
pressure = torch.tensor([[0.0], [100.0]], device="cuda")
structures = model.sample(z, num_atoms, conditions=pressure)

# without composition, with numbers of atoms as the ones of MP-20
generation = CrystalFlowGeneration.from_pretrained(device="cuda")
structures = generation.sample(generation.sample_num_atoms(64))

# towards a formation energy per atom, in eV
generation = CrystalFlowGeneration.from_pretrained(
    "crystalflow-dng-mp20-formation-energy", device="cuda"
)
energy = torch.full((64, 1), -2.0, device="cuda")
structures = generation.sample(generation.sample_num_atoms(64), conditions=energy, guide_factor=2.0)
```

## The rewrite

A structure is its lattice and the fractional coordinates of its atoms. The lattice is given
to the flow as six numbers k of its polar decomposition: the symmetric part P = (L L^T)^(1/2)
of the lattice is exp(S) for a symmetric matrix S, whose 6 numbers are k
(`polar_lattice_parameters` and `lattice_from_polar_parameters` of `torchms.graph.geometry`).
The lattices CrystalFlow gives are these symmetric matrices, with the lengths and the angles
of a crystal in another orientation.

A CSPNet gives the velocity of k and of the coordinates at a time t. It is the network of
DiffCSP (`torchms.models.layers.cspnet`, whose layers are reused), which reads on each edge
the plane waves cos(G r) and sin(G r) of the 512 reciprocal vectors of the Miller indices
from 0 to 7 (`reciprocal_fourier` of `torchms.models.layers.basis`), the Fourier basis of
the difference of the coordinates, and k. The graph is complete, each atom included.

`sample` draws k with normal noise of deviation 0.1 around (0, 0, 0, 0, 0, 1), a cube of
volume e^3, and uniform coordinates, and makes 100 Euler steps from the time 0 to 1. A step
reads the network at the time of its end, from 1/100 to 1, as CrystalFlow, and the velocity
of the coordinates is multiplied by `1 + 5 max(t, 0)`, the annealing of the README of
CrystalFlow for the prediction of a structure. `sample_step` is one step, `velocity` the
network, `integrate` the whole flow from a given state.

**With the elements.** In `CrystalFlowGeneration` the element of an atom is two one-hot
vectors, of its row and of its column in a table of 13 rows and 15 columns of the elements
(`type_module.py` of CrystalFlow, written in `config.yaml`). The vectors start from normal
noise, get their velocity from the network, and are read at the end: the largest row, then
the largest column of an element in this row. `sample_num_atoms` draws the numbers of atoms
of MP-20. `scripts/generation.py` of CrystalFlow does not anneal: neither does the rewrite
by default.

**With a condition.** Two sets of weights are conditioned on a property: the pressure in GPa
(`crystalflow-csp-calypso60-pressure`) and the formation energy per atom in eV
(`crystalflow-dng-mp20-formation-energy`). The property is standardized with the mean and the
deviation of the training set, read from `prop_scalers.pt` of the weights, expanded in 50
gaussians from -2 to 2, and added through a network to the features of the atoms before each
layer. A step with `guide_factor` w computes `w v(c) + (1 - w) v`, the velocity with and
without the condition; with w = 1, the default, only the first one is computed.

**Weights.** The authors publish their checkpoints in the release `v1.0.0-alpha.1` of their
repository: four in `ckpt-v1.0.0.zip` (883 MB), which is downloaded once for all of them,
and `CSP-mpts-52.zip` (206 MB), added to the release in May 2026. The archives and each file
read from them are checked against their SHA-256. A checkpoint of pytorch-lightning holds
objects of omegaconf, and the scalers objects of CrystalFlow: they are read without importing
them (`read_pickle`). The configuration of each training is checked against `config.yaml`;
the conditioned weights have more modules, built by `for_weights` from the section
`variants`.

Not included: the losses and the training, the optimal transport, the symmetrization by
space groups, the other solvers of the ODE, the graph of nearest neighbours, the other
encodings of the elements and of the properties, the metrics.

## Comparison with the original

CrystalFlow `v1.0.0-alpha.1` (`d77d6d0`) was run with torch 2.3.1, loaded with its
`scripts/eval_utils.py`, on the 40 crystals of the tests (225 atoms), each one at its own
time from 0.01 to 1, and its sampling `CSPFlow.sample` was run whole for 6 structures of 1 to
12 atoms, with everything it draws recorded. The conditioned models were given a value for
each structure, the pressure from 0 to 200 GPa and the formation energy from -3 to 0 eV, and
sampled with a guide factor of 1.5, which runs both networks. Each step below starts from the
state CrystalFlow had at this step. A sampling is never compared through a seed.

| Largest difference, against CrystalFlow | Network on 40 crystals: lattice, coordinates, elements | A step, relative | The 100 steps chained: lattice (CrystalFlow single against double) | Elements sampled |
|---|---|---|---|---|
| `crystalflow-csp-mp20` | 1.3e-7, 9.3e-7 | 2.4e-6 | 3.4e-2 Å (2.4e-2) | given |
| `crystalflow-csp-mpts52` | 5.4e-7, 5.2e-7 | 3.2e-6 | 3.2e-2 Å (2.4e-2) | given |
| `crystalflow-csp-calypso60-pressure` | 1.8e-7, 1.6e-6 | 2.9e-6 | 1.6e-2 Å (1.5e-2) | given |
| `crystalflow-dng-mp20` | 1.4e-5, 7.2e-6, 2.5e-4 | 1.2e-5 | 7.1e-4 Å (8.1e-4) | identical |
| `crystalflow-dng-mp20-formation-energy` | 6.0e-7, 4.4e-7, 1.1e-5 | 4.0e-5 | 9.1e-4 Å (1.6e-3) | identical |

The values of the network are up to 1.5 for the lattice, 0.46 for the coordinates and 3.0
for the elements. A step from a recorded state to the next one agrees to 2.4e-7, two chained
steps to 3e-8. A change of 0.1 % of a tensor of weights or of a scaler makes the comparisons
fail, except for the bias of the last network of the last layer of the CSP model, on which
the outputs do not depend: 10 % of it moves them by 1e-5.

### The flow multiplies the rounding

**A relative change of 1e-7 at the second step moves the lattice by 2e-3 and the coordinates
by 2.7e-2 after 100 steps.** The whole sampling of the rewrite and of CrystalFlow agree as
CrystalFlow in single precision agrees with itself in double precision, the last column of
the table above: the tests hold the rewrite to three times this gap, and every step to 1e-5.

**The network which generates the elements multiplies the rounding near the time 1.** On the
perovskite of the tests at the time 1, CrystalFlow in single and in double precision differ
by 2.9e-4 on the velocity of the elements; the rewrite differs from CrystalFlow by 2.5e-4, and
from CrystalFlow in double precision by 2.8e-4. The tests of the generation hold a value
further than 1e-5 from CrystalFlow to be as near to CrystalFlow in double precision as
CrystalFlow is (`as_near_as_original`). CrystalFlow in double precision keeps its Fourier
frequencies in float32: it is not exact either.

## Behaviours kept from CrystalFlow

- the network is read at the end of each step, t = s / N for the step s, not at its start;
- the lattices sampled are symmetric matrices, the polar part of a cell;
- the reciprocal vectors are computed through the lattice, 2 pi times cross products over
  the volume plus 1e-9, although their plane waves only depend on the fractional
  coordinates;
- the velocity of the elements with a condition is not annealed even when the one without is
  (`anneal_elements`), as in CrystalFlow;
- an element is decoded from a row of 13 and a column of 15 values: a cell of the table
  without element is never chosen;
- the checkpoints keep the noise schedules of DiffCSP, a projection on space groups and, for
  the models without condition, unused conditioning layers: they are not loaded.

## Differences with CrystalFlow

- the noise is drawn by torch with a generator, in the order of CrystalFlow: the samples are
  not the ones of CrystalFlow for the same seed, their distribution is;
- the complete graph is built without a loop over the structures, once per sampling;
- the condition is given in the units of the data, a tensor of one column per property, and
  standardized by the model.
