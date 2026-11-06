# ADiT

ADiT, the All-atom Diffusion Transformer, is the latent diffusion model of crystals and
molecules of Joshi, Fu, Liao, Gharakhanyan, Miller, Sriram and Ulissi
([all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer),
commit `b9ce505`, CC BY-NC 4.0, [arXiv:2503.03965](https://arxiv.org/abs/2503.03965)), in
`torchms/models/adit/`. One model generates both: a variational autoencoder maps the atoms
of a crystal or of a molecule to latent vectors, and a diffusion transformer generates such
vectors, which the decoder turns into structures.

| | `ADiT` |
|---|---|
| Does | generates crystals or molecules of given numbers of atoms, with their elements; encodes and decodes structures |
| Source | [all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer), commit `b9ce505`, `vae_module.py`, `ldm_module.py`, `denoisers/dit.py` |
| License | **CC BY-NC 4.0**: no commercial use, see below |
| Paper | [arXiv:2503.03965](https://arxiv.org/abs/2503.03965), ICML 2025 |
| Weights | `adit-qm9-mp20`, of the first author, trained on QM9 and MP-20 together, 2.5 GB |
| Transform | none: the model reads no graph |
| Largest difference with the original | 9.5e-6 on the scores of the elements (values up to 29), 4.3e-6 on the prediction of a step of the flow (values up to 6.7) |

```{.python notest}
import torch

from torchms.data import Structures
from torchms.models import ADiT

model = ADiT.from_pretrained(device="cuda")

# 64 crystals, with numbers of atoms as the ones of MP-20
num_atoms = model.sample_num_atoms(64, crystals=True)
crystals = model.sample(num_atoms, crystals=True)   # 100 times, 100 calls of the denoiser
crystals.lattice                                    # (64, 3, 3), in Å
crystals.x                                          # fractional coordinates

# 64 molecules, with numbers of atoms as the ones of QM9
molecules = model.sample(model.sample_num_atoms(64, crystals=False), crystals=False)
molecules.positions()                               # in Å, molecules.pbc is False

# structures through the autoencoder
structures = Structures.collate([Structures.diamond(), Structures.water()]).to("cuda")
posterior = model.encode(structures)                # mean and log-variance, 8 per atom
decoded = model.decode(posterior.mean, structures.num_atoms, model.is_crystal(structures))
```

With a seed of 0, six crystals drawn have between 9 and 20 atoms and the compositions
BaSmMn2O6, FeGe7Dy6, NbMoS4Se2, CaZnAlEuO6, ScO2 and KCuGd2S4; six molecules have between
12 and 25 atoms, C8H16O, C8H11N, C5H6N2, C4H4N4, C8H12O and C7H12O2. These are single draws:
the quality of the model is the one its authors report, it was not measured here.

## The rewrite

**The autoencoder.** A structure is a sequence of tokens, one per atom, in the order of its
atoms. The encoder sums the embedding of the element, a network of the position in Å, a
network of the fractional coordinates (zero for a molecule) and a sinusoidal embedding of
the index of the atom, and runs a transformer of 8 layers of 512 features on the atoms of
each structure. A linear layer gives the mean and the log-variance of a normal distribution
of 8 values per atom: `encode` returns this `Posterior`. The decoder maps a latent vector
back to 512 features, adds the embedding of the index, runs a second transformer, and
predicts for each atom the scores of its element, its fractional coordinates and its
position in nm, and from the mean over the atoms the lengths and the angles of a cell
(`decode_outputs`, a `Decoded`). `to_structures` reads a crystal from the cell and the
fractional coordinates, a molecule from the positions.

**The diffusion.** DiT-B, a transformer of 12 blocks of 768 features whose normalizations
are shifted, scaled and gated by the sum of embeddings of the time, of the dataset and of
the space group (adaLN-Zero), reads the noisy latent vectors with the prediction of the step
before (self-conditioning) and predicts the latent vectors without noise. The flow goes from
centered normal noise at the time 0.01 to the latent vectors at the time 1, by Euler steps
along `(prediction - x) / (1 - t)`. It is guided by the dataset: crystals (MP-20) or
molecules (QM9). The denoiser is run with the dataset of each structure and without
dataset, and the prediction is the second one plus `cfg_scale` times their difference.

`sample` draws the noise, makes the 99 steps and decodes the prediction at the time 1.
`sample_step` is one step, from the latent vectors and the prediction of the step before;
`denoise` is the guided denoiser alone; `sample_latents` the whole flow from a given noise.
A batch can mix crystals and molecules: `crystals` has a value per structure.

**The numbers of atoms are not generated.** ADiT draws them from the numbers of atoms of its
training sets (`num_nodes_bincount.pt` of its datasets on Hugging Face), which are in
`config.yaml`: `sample_num_atoms` draws from 1 to 20 atoms for a crystal, from 3 to 29 for a
molecule. `sample` takes any numbers of atoms.

**Sampling settings.** 100 times and a guidance of 2.0, the defaults of
`configs/diffusion_module/ldm.yaml`, which `src/eval_diffusion.py` of ADiT samples with.
The checkpoint stores the values of its training run, 250 and 4.0; the sweep of the
repository tries 25 to 1000 times and scales from 1 to 8. Both are arguments of `sample`.

Not included: the losses, the training and the augmentations, the exponential moving
average of the weights, the Equiformer encoder (the published weights use the transformer
one), the conditioning on the space group (the weights were trained without it: the
denoiser reads the null class), the datasets, the metrics and the visualization.

The weights are on Hugging Face, published by the first author as "an independent
reproduction of this research", which "may not correspond to the exact models/performance
metrics reported in the final manuscript". The repository has the autoencoder `vae.ckpt`
and the diffusion `ldm.ckpt`, trained on QM9 and MP-20, and an undertrained pair on
GEOM-DRUGS, which is not ported. `ldm.ckpt` holds the frozen autoencoder it was trained
with, identical to `vae.ckpt`: it is the only file downloaded. It is a checkpoint of
pytorch-lightning, which pickles the modules of ADiT: it is read without importing them
(`read_pickle`), and the attributes of the pickled denoiser, interpolant and conditioning
are checked against `config.yaml`. The number of heads and the activation of the
autoencoder are not stored in this file and cannot be checked.

**Exponential moving average.** The checkpoint has a moving average of the weights, which
ADiT swaps in during the validation of its training. `src/eval_diffusion.py` builds a
trainer without callbacks and tests the state dict: the rewrite loads the state dict, as
it does. The moving average of the frozen autoencoder is not the autoencoder: it differs
from it by up to 5e-4, stuck there by the rounding of its updates in float32.

**License.** ADiT is under CC BY-NC 4.0: the code of `torchms/models/adit/` is derived
from it and keeps this license, which forbids commercial use. Its `LICENSE` is in the
directory. The weights are under the MIT license on Hugging Face. To keep the restriction
in one place, the DiT is in `torchms/models/adit/dit.py`, not in `torchms.models.layers`.

## Comparison with the original

ADiT `b9ce505` was run with torch 2.3.1 on the CPU, built as `src/eval_diffusion.py`
builds it, on the 40 crystals of the tests (225 atoms) and 4 molecules (water, methane,
ammonia, ethanol, 21 atoms), in one batch. Its sampling was run whole for 6 crystals and 6
molecules, with its noise recorded: each step below starts from the latent vectors and the
prediction ADiT had at this step. A sampling is never compared through a seed.

| Weights `adit-qm9-mp20` | Largest | Mean |
|---|---|---|
| Mean of the posterior (values up to 11) | 2.4e-6 | 3.9e-7 |
| Log-variance of the posterior (values up to 11) | 4.8e-6 | 7.9e-7 |
| Decoder, from the mean of ADiT: scores of the elements (values up to 29) | 9.5e-6 | 8.9e-7 |
| Decoder: lengths and angles of the cell (values up to 13) | 9.5e-7 | 1.2e-7 |
| Decoder: fractional coordinates, positions in nm (values up to 1.3) | 3.0e-7 | 6.3e-8 |
| Steps 1, 2, 3, 51, 98, 99 and the prediction at the time 1, crystals (70 atoms): prediction (values up to 6.7) | 4.3e-6 | |
| The same, latent vectors after the step | 2.4e-6 | |
| The same, molecules (106 atoms): prediction and latent vectors (values up to 4.0) | 2.6e-6, 1.9e-6 | |
| Whole sampling chained from the noise of ADiT, crystals: latent vectors at the time 1 (values up to 6.7) | 1.1e-5 | 8.4e-7 |
| The same, molecules (values up to 4.0) | 1.9e-6 | 3.3e-7 |
| Elements of the 176 atoms decoded from the latent vectors sampled by the rewrite | identical | |

A change of 0.1 % of one tensor of weights, in the encoder, in the decoder or in the
denoiser, makes the comparisons fail. The tests check one step in the middle of the flow by
default, every recorded step and the whole sampling with `--full`.

## Behaviours kept from ADiT

- **The GELU of the autoencoder is the exact one.** ADiT configures its transformers with
  the approximation of GELU by tanh, as `nn.GELU(approximate="tanh")`. When the weights of a
  `nn.TransformerEncoderLayer` have no gradient, as in every sampling and evaluation of ADiT
  (its autoencoder is frozen), torch takes a fast path which only knows the exact GELU, and
  computes it. The autoencoder was trained with the approximation and is always run with
  the exact function: the rewrite computes the exact one, in both paths. With the
  approximation, the decoder of the sampled crystals differs from ADiT by 1.2e-2 on the
  scores of the elements and 4.7e-4 on the cell, against 7.6e-6 and 4.8e-7. The DiT has its
  own feed-forward networks, with the approximation, which are computed as such.
- the initial noise is centered over each row of the padded batch, padding included: the
  noise of a structure depends on the largest structure of its batch (`center_noise`);
- the index of an atom in its structure is embedded: the model is not invariant to a
  permutation of the atoms, nor to a rotation or a translation, which it learned from
  augmentations of its training;
- the masked atom, the row 0 of the embedding, is decoded as hydrogen; an element after
  einsteinium has no row and is refused;
- the decoded cell is not checked: a negative length or angles of no cell give the lattice
  `lattice_from_parameters` gives them. ADiT marks such crystals as invalid in its metrics;
- the space group is read by the denoiser as its null class.

## Differences with ADiT

- the fractional coordinates of a decoded crystal are wrapped in the cell; ADiT gives them
  to pymatgen as they are;
- the batch is padded by `torchms.graph.sparse.to_dense`, without a loop, and the mean of the
  decoder over the atoms is taken in this layout;
- the numbers of atoms and the noise are drawn by torch with a generator, in another order:
  the samples are not the ones of ADiT for the same seed, their distribution is.
