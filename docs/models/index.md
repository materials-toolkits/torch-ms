# Models

`torchms.models` has supervised models, which predict the energy of a structure, generative
models, which generate structures, and layers: what they share and what a new model can be
built from. Every model takes a batch of `Structures` and reads the graph attached to it:
**no model builds its graph**, each one gives the transform which does, `Model.transform()`.

See [Using models](../guide/models.md) to load and run them.

## Supervised models

Models which predict a property of a structure: its energy.

Models published by other projects, **rewritten** with PyTorch and the graphs of
`torchms.graph`, for inference only. Each rewrite loads the published weights and was
compared once with the original code, on the same 40 structures (253 for orb-v3).

| Model | Class | Predicts | Source | Weights | Largest difference with the original |
|---|---|---|---|---|---|
| [orb-v3](orb.md) | `OrbEnergyModel` | total energy | orb-models 0.7.0, Apache-2.0 | official | 2.7e-5 eV/atom below 120 neighbours |
| [MACE](mace.md) | `MaceEnergyModel` | total energy | mace 0.3.16, MIT | official | 5e-15 eV/atom, double precision |
| [M3GNet](m3gnet.md) | `M3gnetEnergyModel` | total energy | matgl 4.1.0, BSD 3-Clause | official | 1.6e-6 eV/atom |
| [CHGNet](chgnet.md) | `ChgnetEnergyModel` | total energy | chgnet 0.4.2, modified BSD | official | 4.1e-6 eV/atom |
| [ALIGNN](alignn.md) | `AlignnModel` | formation energy | alignn 2026.8.11, NIST | official | 4.8e-7 eV/atom with the same 12 neighbours |

Forces, stress, magnetic moments and training are not included. orb-v3 is the model of the
[energy estimator](../guide/energy.md); the other ones are not used by it.

## Generative models

Models which generate structures.

| Model | Class | Generates | Source | Weights | Largest difference with the original |
|---|---|---|---|---|---|
| [CDVAE](cdvae.md) | `CdvaeModel` | crystals of at most 20 atoms, from a latent vector | cdvae `f857f59`, MIT | **third party** | 4.0e-5 on the latent mean |

CDVAE is a rewrite too, compared with the original code as the supervised models. Its
weights were not trained by its authors, who publish none: see its page. Its training is
not included.

## Pretrained models

What the pretrained models of both kinds have in common. None of the projects they come from
is a dependency. The differences of the tables above are the ones of the page of each model,
which says on which structures they hold: orb-v3, ALIGNN and CDVAE keep a fixed number of
nearest neighbours, which is not defined when several neighbours are at the same distance.

### Graphs

| Model | Transform |
|---|---|
| orb-v3 | `AddRadiusGraph(6.0)`, `KeepNearestNeighbors(120)` |
| MACE | `AddRadiusGraph(6.0)` |
| M3GNet | `AddRadiusGraph(5.0)`, `AddTriplets(radius=4.0)` |
| CHGNet | `AddRadiusGraph(6.0)`, `AddTriplets(radius=3.0, strict=True)`, `AddBonds()` |
| ALIGNN | `AddRadiusGraph(5.0)`, `KeepNearestNeighbors(12)`, `AddLineGraph(radius=3.5, backtrack=True)` |
| CDVAE, encoder | a graph of bonds given with the structures, then `AddLineGraph(backtrack=True, same_atom=False)` |
| CDVAE, decoder | `AddRadiusGraph(7.0)`, `KeepNearImages(1)`, `KeepNearestNeighbors(20)`, `MirrorEdges()`, `AddTriplets()`, `AddBonds()` |

Every radius graph has `self_images=True`. See [Graphs](../guide/graphs.md) for the
transforms.

### Files of a model

Each pretrained model is a directory of `torchms/models/`:

| File | Content |
|---|---|
| `model.py` | the network, a subclass of `PretrainedModel`. It has no value of its own |
| `config.yaml` | the files of its weights with their SHA-256, the radii of its graph, its hyperparameters, and the arguments checked in the weights of the project it comes from |
| `LICENSE` | the license of the project it is derived from |

Weights are downloaded on the first use to `~/.cache/torchms` (`$TORCHMS_CACHE`) and checked
against their SHA-256.

## Layers

`torchms.models.layers` has what several models share, and what a new model can be built
from.

| Layer | Content | Used by |
|---|---|---|
| [Bases](bases.md) | radial bases, bases of angles, spherical harmonics, smooth cutoffs | every model |
| [DimeNet++](dimenet.md) ([arXiv:2011.14115](https://arxiv.org/abs/2011.14115)) | network on the bonds of a graph and their angles, one vector per structure | CDVAE (encoder) |
| [GemNet-T](gemnet.md) ([arXiv:2106.08903](https://arxiv.org/abs/2106.08903)) | network on the edges of a symmetric graph and their triplets, one vector per atom | CDVAE (decoder) |
| [`LineGraphNetwork`](line-graph-network.md) | network of the energy on pairs, triplets and quadruplets, with their angles and dihedral angles | no model of the library |
| [`EquivariantNetwork`](equivariant-network.md) | network of the energy on pairs only, with products of sums of spherical harmonics | no model of the library |

The first three take their hyperparameters as arguments and read the graph they are given.
The last two were written for this library, as the two ways recent models have terms on more
than two atoms (the first as ALIGNN, M3GNet, CHGNet or DPA-3, the second as MACE). They are
not these models, have no pretrained weights and were not trained here: their accuracy is
not measured. Each one reads the graph of its `transform()`, which depends on its
arguments: the radius, and the order of `LineGraphNetwork`.

The other blocks of a pretrained model stay in its directory: they are named after the
project they come from, to load its weights.

## Energy estimator

`torchms.models.estimator` has `EnergyEstimator`, which corrects the energy of orb-v3 and
compares it to a convex hull, and what the pretrained models share to load their weights.
See [Energy above the hull](../guide/energy.md) and the
[design note](../design/estimator.md).
