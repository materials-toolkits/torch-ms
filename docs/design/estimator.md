# Energy estimator

`torchms.estimator` estimates the PBE total energy of a structure on the scale of the
[hull of LeMat-Bulk](hull.md).

## Model

The model is orb-v3 conservative of Orbital Materials, checkpoint
`orb-v3-conservative-inf-mpa`
([orb-models](https://github.com/orbital-materials/orb-models), Apache-2.0,
[arXiv:2504.06231](https://arxiv.org/abs/2504.06231)).

`orb-models` needs Python 3.12 and several dependencies the project does not have. Its
prediction of the energy is therefore **rewritten** in `estimator/orb/`, for inference only,
with PyTorch alone: atom embedding, edge features (radial basis × spherical harmonics), five
attention layers, an energy head with a reference energy per element, and a repulsion between
close atoms. Forces, stress and training are not included. The official weights (102 MB) are
downloaded on the first use and checked against a pinned SHA-256.

The graph is built by `torchms.graph`: atoms closer than 6 Å, at most the 120 nearest,
periodic images of the atom itself included.

### Is the rewrite faithful

Energies of `orb-models` 0.7.0 were computed once, outside of the project, for 253 structures:

| | Difference with orb-models |
|---|---|
| Mean | 1.5e-6 eV/atom |
| Largest, cells below 120 neighbours | 2.7e-5 eV/atom |
| Largest, cells above 120 neighbours | 1.7e-4 eV/atom |

Above 120 neighbours, atoms at the same distance are not kept in the same order as
`orb-models`. A test compares 40 of these structures at each run, when the weights are in the
cache.

### Behaviours kept from orb-models

The weights were trained with them, so they are reproduced:

- the repulsion between close atoms uses the atomic number **plus one**;
- this repulsion is **averaged** over the atoms of a structure instead of summed: this small
  term does not grow with the cell;
- the network is **not exactly invariant to rotations**: the energy changes by about
  2e-4 eV/atom on average, up to 8e-3, with the orientation of the cell.

## Choice of the checkpoint

Two checkpoints were compared on the same structures of LeMat-Bulk.

| Mean error (eV/atom) | `mpa` | `omat` corrected |
|---|---|---|
| Materials Project and Alexandria, held out | 0.0097 | 0.0138 |
| OQMD, new structures | 0.0105 | 0.0141 |
| OQMD, new compositions | 0.0123 | 0.0157 |

`mpa` is trained on energies computed like the ones of the Materials Project and is already on
the scale of LeMat-Bulk; `omat` is 0.064 eV/atom off before correction. The predictions of
`omat` published by LeMaterial are reproduced by the rewrite (median difference 1e-6 eV/atom).

## Correction

An energy per atom of each element is added to the prediction, fitted on 300,000 structures of
the Materials Project and Alexandria to minimize the absolute error
(`estimator/calibration.py`). For `mpa` it is small: 0.003 eV/atom on average, at most 0.024.
It lowers the error of the held-out set from 0.0100 to 0.0097 eV/atom and the share of stable
structures found from 27% to 38%; it changes nothing on the structures of OQMD. A least squares
fit made the error slightly larger and is not used.

## Validation

Structures of LeMat-Bulk which were not used to fit, and never share a fingerprint with a
fitted one. The network is trained on data from the Materials Project and Alexandria, not
OQMD: sets B and C are structures of OQMD with a fingerprint (B) or a composition (C) absent
from both.

| Set | Structures | Mean error | Median | 90% | Same side of 0.1 eV/atom as DFT | Stable found | Wrongly stable |
|---|---|---|---|---|---|---|---|
| A. Materials Project and Alexandria | 60,000 | 0.0097 | 0.0042 | 0.018 | 98.9% | 38% | 35% |
| B. OQMD, new structures | 37,856 | 0.0105 | 0.0056 | 0.024 | 98.9% | 61% | 47% |
| C. OQMD, new compositions | 36,153 | 0.0123 | 0.0067 | 0.028 | 99.1% | 78% | 34% |
| D. OQMD, new structures, all elements | 59,994 | 0.136 | 0.012 | 0.115 | 92.7% | 55% | 61% |

Errors are in eV/atom. B and C leave out the nine elements whose energies in OQMD are shifted;
D includes them, and its error is the shift of the reference energies, not of the estimator.
Energies above the hull use `hull.hdf5` for DFT and for the estimator; a structure is stable
when it is on or below the hull.

`scripts/build_energy_correction.py` fits and `scripts/validate_energy.py` validates end to
end.

## Limits

- The error, 0.01 eV/atom, is as large as the distance between a stable phase and its close
  competitors: the estimator tells if a structure is near the hull, not if it is on it.
- The energy is the one of the structure as given, not relaxed.
- The training data of the network and the sources of LeMat-Bulk overlap: the error on
  structures unlike all of them is not measured here.
- A structure with two atoms at the same position has no energy (NaN).
- That OQMD is outside of the training data comes from the documentation of `orb-models`,
  which gives the Materials Project trajectories and Alexandria for this checkpoint.

## Other pretrained models

Other published models are rewritten the same way, to check the graph functions of
`torchms.graph` against models which use terms on three atoms and more. They predict the
energy only, load the official weights, and are not used by `EnergyEstimator`. CDVAE, the
last one, is a generative model in `torchms.models`, with weights which are not official. No package of
these models is a dependency: their energies were computed once, outside of the project, by
the scripts `scripts/reference_*.py`, for the 40 structures of the test of orb.

Each model has a file `config.yaml` next to its module `model.py`, with the files of its
pretrained weights, the radii of its graph, its hyperparameters and the arguments checked in
the weights of the project it comes from. `model.py` has no value of its own: adding weights
or reading a radius is done in this file.

None of these models builds a graph. Each class gives the transform which attaches to the
structures the graph and the tuples of edges it reads, `Model.transform()`, a chain of the
transforms of `torchms.graph.transforms`. It is applied when the structures are loaded, or to
a batch; a model refuses structures without them.

```{.python notest}
from torchms.data import HDF5Dataset
from torchms.estimator import M3gnetEnergyModel

dataset = HDF5Dataset("structures.hdf5", transform=M3gnetEnergyModel.transform())
model = M3gnetEnergyModel.from_pretrained(device="cuda")
energy = model(dataset[:32].to("cuda"))
```

| Model | Transform |
|---|---|
| orb-v3 | `AddRadiusGraph(6.0)`, `KeepNearestNeighbors(120)` |
| MACE | `AddRadiusGraph(6.0)` |
| M3GNet | `AddRadiusGraph(5.0)`, `AddTriplets(radius=4.0)` |
| CHGNet | `AddRadiusGraph(6.0)`, `AddTriplets(radius=3.0, strict=True)`, `AddBonds()` |
| ALIGNN | `AddRadiusGraph(5.0)`, `KeepNearestNeighbors(12)`, `AddLineGraph(radius=3.5, backtrack=True)` |
| CDVAE, encoder | a graph of bonds given with the structures, then `AddLineGraph(backtrack=True, same_atom=False)` |
| CDVAE, decoder | `AddRadiusGraph(7.0)`, `KeepNearImages(1)`, `KeepNearestNeighbors(20)`, `MirrorEdges()`, `AddTriplets()`, `AddBonds()` |

Every radius graph has `self_images=True`. `EnergyEstimator` applies the transform of its
model to the structures it is given without a graph.

### CHGNet

`ChgnetEnergyModel` is CHGNet 0.3.0
([chgnet](https://github.com/CederGroupHub/chgnet), modified BSD,
[arXiv:2302.14231](https://arxiv.org/abs/2302.14231)), in `estimator/chgnet/`. Atoms closer
than 6 Å are bonded, an edge and its reverse sharing the features of one bond
(`reverse_edges`); the angles are the pairs of bonds shorter than 3 Å around an atom
(`triplets` with a mask), in a Fourier basis. Magnetic moments, forces and stress are not
included.

| Against chgnet 0.4.2, 40 structures | |
|---|---|
| Edges, bonds and angles of each structure (29,222 angles) | identical |
| Largest difference of energy | 4.1e-6 eV/atom |
| Mean difference of energy | 4.3e-7 eV/atom |

Behaviours kept from chgnet: the cosine of an angle is multiplied by 1 - 1e-6 before the
arc cosine, and an atom without neighbour is not an error.

### M3GNet

`M3gnetEnergyModel` is the M3GNet potential `M3GNet-PES-MatPES-PBE-2025.2`
([matgl](https://github.com/materialsvirtuallab/matgl), BSD 3-Clause,
[arXiv:2202.02450](https://arxiv.org/abs/2202.02450)), in `estimator/m3gnet/`. Atoms closer
than 5 Å are bonded; a bond is updated from the angles it forms with the other bonds of its
atom of at most 4 Å (`triplets` with a mask), in a basis of spherical Bessel functions and
spherical harmonics. matgl builds these functions with sympy; they are computed here with
torch, and their roots with scipy. Forces and stress are not included.

| Against matgl 4.1.0, 40 structures | |
|---|---|
| Edges and angles of each structure (128,348 angles) | identical |
| Roots of the spherical Bessel functions | identical |
| Largest difference of energy | 1.6e-6 eV/atom |
| Mean difference of energy | 4.6e-7 eV/atom |

The model knows 89 elements, H to Bi and Ac to Pu: a structure with another element raises an
error.

### ALIGNN

`AlignnModel` is the ALIGNN 2.0 model `formation_energy_peratom_radius`
([alignn](https://github.com/usnistgov/alignn), NIST,
[arXiv:2106.01829](https://arxiv.org/abs/2106.01829)), in `estimator/alignn/`. It predicts
the formation energy in eV/atom, not a total energy. An atom is connected to its 12 nearest
neighbours closer than 5 Å (`nearest_neighbors`); the line graph connects a bond u→v to the
bonds v→w of at most 3.5 Å, its own reverse included (`line_graph`). The features of the
elements are not stored with the weights: they are in `cgcnn_features.npz`, written by the
reference script from jarvis-tools.

| Against alignn 2026.8.11, 40 structures | |
|---|---|
| Line graph built on the graph of atoms of alignn (19,944 angles) | identical |
| Prediction on the graph of atoms of alignn, largest difference | 1.8e-6 eV/atom |
| Structures where both libraries keep the same 12 neighbours | 16 of 40 |
| Prediction on these 16 structures, largest difference | 4.8e-7 eV/atom |
| Prediction on the 24 other structures, largest difference | 0.012 eV/atom |
| Prediction on the 24 other structures, mean difference | 0.0025 eV/atom |

**The limit of 12 neighbours is not well defined.** In a crystal many neighbours are at the
same distance. When the 12th and the 13th are, the ones which are kept depend on the order of
a sort: alignn and this rewrite do not keep the same ones (124 edges of 2,658 differ), and the
prediction changes by up to 0.012 eV/atom. For the same reason the prediction of such a
structure can change with a rotation, a translation or a change of basis of the cell; this
was not measured. `AlignnModel` reads the graph attached to the structures: with the graph of
alignn attached, it predicts on this exact graph.

The graphs follow the configuration of the training of the model. The function of alignn
which predicts a property builds another line graph, with all the bonds up to 5 Å; it was not
reproduced.

### MACE

`MaceEnergyModel` is MACE-MP-0 small
([mace](https://github.com/ACEsuit/mace), MIT,
[arXiv:2206.07697](https://arxiv.org/abs/2206.07697)), in `estimator/mace/`. Its graph has
pairs only, the atoms closer than 6 Å: terms on three and four atoms come from products. An
atom sums over its neighbours a radial function times spherical harmonics (l ≤ 3) times the
features of the neighbour; products of two and three of these sums are contracted to
invariant features with coupling tensors. The layers of e3nn used by mace (linear layers,
tensor products, normalized activations) are written with torch alone.

| Against mace 0.3.16, 40 structures, double precision | |
|---|---|
| Edges of each structure | identical |
| Largest difference of energy | 5e-15 eV/atom |

The energy is exactly invariant to the rotations and reflections of `LatticeRotation`, unlike
orb: it is checked in the tests, with the invariance of the coupling tensors.

**The weights are read without mace.** mace publishes a pickled model, the whole Python
object and not its tensors: unpickling it normally needs mace and e3nn. `estimator/mace/reader.py`
unpickles the official file with every class of mace and e3nn replaced by a placeholder, and
collects the tensors from the placeholders: the 69 tensors are the ones of the state dict of
mace, value for value. The unpickler only resolves a fixed list of names (tensors of torch,
containers of the standard library), so the file cannot run code; this is tested with a file
which runs a function when unpickled as usual. The coupling tensors are among these tensors:
mace stores them with the weights, and the weights are only valid for this choice of tensors.

Only the small model, whose features are invariant, is rewritten. The medium and large models
have vector and tensor features, which need Clebsch-Gordan coefficients in the tensor products
between layers.


### CDVAE

`CdvaeModel` is the Crystal Diffusion Variational Autoencoder
([cdvae](https://github.com/txie-93/cdvae), commit `f857f59`, MIT,
[arXiv:2110.06197](https://arxiv.org/abs/2110.06197)), in `models/cdvae/`. It is not a
model of the energy: it generates structures. A DimeNet++ encodes a structure into a
latent vector (`dimenet.py`); three small networks decode from it a number of atoms, a
cell and a composition; a GemNet-T (`gemnet.py`) then moves atoms placed at random, in
50 levels of noise of 100 steps each (`model.py`). The losses, the training, the
prediction and the optimization of a property are not included.

```{.python notest}
from torchms.models import CdvaeModel

model = CdvaeModel.from_pretrained(device="cuda")
structures = model.sample(64)                 # 5000 steps of dynamics

mean, log_variance = model(model.transform()(structures_with_bonds))
structures = model.generate(mean)             # reconstruction
```

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

Behaviours kept from cdvae:

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

Differences with cdvae:

- the elements of a batch are drawn without a loop over its structures: the draws are
  not the ones of cdvae for the same seed, their distribution is;
- a bond of the encoder longer than 7 Å contributes nothing, where the envelope of
  cdvae is not zero. No bond of the 40 structures is longer than 4.7 Å;
- two atoms at the same position are neighbours here and not in cdvae, and give NaN;
- a decoded cell which is nearly flat has a very large number of images within 7 Å:
  the graph of the decoder then needs more memory than there is, where cdvae, which
  looks at 27 cells, goes on.
