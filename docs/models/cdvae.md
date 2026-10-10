# CDVAE

`CdvaeModel` is the Crystal Diffusion Variational Autoencoder
([cdvae](https://github.com/txie-93/cdvae), commit `f857f59`, MIT,
[arXiv:2110.06197](https://arxiv.org/abs/2110.06197)), in `torchms/models/cdvae/`. It generates
structures.

| | |
|---|---|
| Class | `torchms.models.CdvaeModel` |
| Does | encodes a structure into a latent vector, generates structures from latent vectors |
| Source | [cdvae](https://github.com/txie-93/cdvae), commit `f857f59` |
| License | MIT |
| Paper | [arXiv:2110.06197](https://arxiv.org/abs/2110.06197) |
| Weights | `cdvae-mp20-cowboys`, **trained by a third party**, 56 MB |
| Transform, encoder | a graph of bonds given with the structures, then `AddLineGraph(backtrack=True, same_atom=False)` |
| Transform, decoder | `AddRadiusGraph(7.0, self_images=True)`, `KeepNearImages(1)`, `KeepNearestNeighbors(20)`, `MirrorEdges()`, `AddTriplets()`, `AddBonds()` |
| Largest difference with the original | 4.0e-5 on the latent mean; 1.2e-5 Å on the moves where the neighbours are defined |

```{.python notest}
from torchms.models import CdvaeModel

model = CdvaeModel.from_pretrained(device="cuda")
structures = model.sample(64)                 # 5000 steps of dynamics

mean, log_variance = model(model.transform()(structures_with_bonds))
structures = model.generate(mean)             # reconstruction
```

## The rewrite

A [DimeNet++](dimenet.md) encodes a structure into a latent vector; three small networks
decode from it a number of atoms, a cell and a composition; a [GemNet-T](gemnet.md) then
moves atoms placed at random, in 50 levels of noise of 100 steps each. The losses, the
training, the prediction and the optimization of a property are not included.

**The weights are not the ones of the authors**, who publish none for this model: the
three checkpoints of their repository are predictors of a property, used to score what
is generated. `cdvae-mp20-cowboys` was trained on MP-20 by a third party with the
configuration of cdvae, and published on Hugging Face
([anitay/cdvae-mp20-cowboys](https://huggingface.co/anitay/cdvae-mp20-cowboys)) without
any description. The rewrite reproduces what cdvae computes with them; whether they
generate good structures was not measured. The checkpoint and the scaler of the cells
are pickles of objects of pytorch-lightning, omegaconf and cdvae: they are read with the
reader of MACE, which runs nothing of a file.

The model has two graphs. Its decoder builds its own at each step of the dynamics. Its
encoder was trained on the bonds found by CrystalNN of pymatgen, which torchms does
not build: `CdvaeModel.transform()` only completes a graph of bonds attached to the
structures. The comparison below attaches the graphs of CrystalNN, computed by the
reference script.

## Comparison with the original

| Against cdvae `f857f59`, torch 1.8.1, 40 structures | |
|---|---|
| Pairs of bonds of the encoder, on the graphs of CrystalNN (10,336 pairs) | identical |
| Latent mean, largest difference (values up to 7.3) | 4.0e-5 |
| Latent mean, mean difference | 2.3e-7 |
| Latent mean, largest difference without bond under 2 Å (29 structures) | 5.2e-6 |
| Scores of the numbers of atoms, lengths and angles of the cells | identical |
| Scores of the composition, largest difference (values up to 235) | 3.8e-5 |
| Structures where the 20 nearest neighbours of the decoder are defined | 23 of 40 |
| Edges and triplets of the decoder on these 23 (3,462 edges, 64,988 triplets) | identical |
| Moves of the atoms on these 23, largest difference (values up to 2.8 Å) | 1.2e-5 Å |
| Moves of the atoms on these 23, mean difference | 8.7e-7 Å |
| Scores of the elements on these 23, largest difference (values up to 30) | 6.6e-5 |
| Moves of the atoms on the 17 other structures, largest and mean difference | 0.11 Å, 0.018 Å |
| 10 steps of dynamics, 3 structures of 6 with defined neighbours at every step | under 1e-5 of the cell |
| 10 steps of dynamics, the 3 other structures, largest difference | 1.3e-3 of the cell |

The largest differences of the encoder are on the three structures with bonds of 1.2 to
1.55 Å, where the spherical Bessel functions of high degree lose digits in single
precision in both codes. cdvae run in double precision differs from itself in single
precision by 3.1e-5 on the same structures; the rewrite, in single precision, differs
from cdvae in double precision by 2.2e-5.

**The 20 nearest neighbours of the decoder are not always defined.** An atom is at the
same distance of its two images at +v and -v. When they are its 20th and 21st
neighbours, which happens in cells of a few atoms, cdvae keeps one or the other from
the rounding of two distances which are equal: its choice changes with the version of
torch, and with a stable sort it is yet another one. The rewrite keeps the image whose
first non-zero shift is negative. In 10 of the 17 structures where this happens the
graph has up to 10 edges more or less than the one of cdvae, and the moves differ by
0.018 Å on average. The tests compare the structures without such a tie, and find them
by themselves.

## Behaviours kept from cdvae

- the decoder only connects an atom to the images of the 26 cells around its own, even
  when the radius of 7 Å reaches further (`KeepNearImages`);
- its graph is made symmetric from the edges towards an atom of lower index
  (`MirrorEdges`): it depends on the order of the atoms. The decoder follows a rotation
  of the cell, which is tested, but not a permutation of the atoms, a translation or a
  change of basis;
- the encoder has an embedding of 95 rows read with the atomic number, the decoder one
  of 100 rows read from hydrogen: a structure with an element after plutonium cannot be
  encoded;
- a composition is rounded to whole atoms, completed or cut at random, and the number
  of atoms is at most 20.

## Differences with cdvae

- the elements of a batch are drawn without a loop over its structures: the draws are
  not the ones of cdvae for the same seed, their distribution is;
- a bond of the encoder longer than 7 Å contributes nothing, where the envelope of
  cdvae is not zero. No bond of the 40 structures is longer than 4.7 Å;
- two atoms at the same position are neighbours here and not in cdvae, and give NaN;
- a decoded cell which is nearly flat has a very large number of images within 7 Å:
  the graph of the decoder then needs more memory than there is, where cdvae, which
  looks at 27 cells, goes on.
