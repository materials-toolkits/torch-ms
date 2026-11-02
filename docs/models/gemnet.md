# GemNet-T

`torchms.models.layers.GemNetT` is GemNet-T, the variant of GemNet of Gasteiger, Becker and
Günnemann which uses triplets of atoms: a network on the edges of a symmetric graph and on
their triplets, which gives features and a vector for each atom, and, as in MatterGen, a
symmetric matrix for each structure.

Here it is rewritten from the code of [CDVAE](cdvae.md), which uses it as its decoder, and
of [MatterGen](mattergen.md), which uses it as the network of its diffusion: these are the
codes it was compared with, not the one of its authors.

| | |
|---|---|
| Classes | `GemNetT`, `gemnet.Config`, `gemnet.Output`, `gemnet.Condition` |
| Output | features of each atom, shape `(n_atoms, hidden_dim)`, a vector of each atom in Å, shape `(n_atoms, 3)`, and with `predict_lattice` a symmetric matrix of each structure, shape `(n_struct, 3, 3)` |
| Reads | a symmetric graph, its triplets and its bonds, a vector of each structure, with `lattice_cosines` the lattice, and with `conditions` one `Condition` per property |
| Paper | [GemNet: Universal Directional Graph Neural Networks for Molecules](https://arxiv.org/abs/2106.08903), Gasteiger, Becker and Günnemann (arXiv:2106.08903) |
| Code of the authors | [TUM-DAML/gemnet_pytorch](https://github.com/TUM-DAML/gemnet_pytorch) |
| Rewritten from | [cdvae](https://github.com/txie-93/cdvae) `f857f59` and [MatterGen](https://github.com/microsoft/mattergen) `9242366`, whose GemNet-T comes from the Open Catalyst Project, MIT |
| Pretrained weights | the ones of [CDVAE](cdvae.md), as its decoder, and of [MatterGen](mattergen.md) |

Features live on the atoms and on the edges. A block updates an edge towards an atom from
the other edges of this atom, in a basis of gaussians of its length and of spherical
harmonics of their angle, and shares this update with the reverse edge; then the atoms are
updated from their edges, and the edges from their atoms. Each edge gives a scalar: their
sum along the directions of the edges of an atom is its vector, which follows the rotations
of the structure.

MatterGen adds two things, which `lattice_cosines` and `predict_lattice` turn on. The
features of an edge also read the cosines of its angles with the three lattice vectors,
through a small network, before the first block. And after each block, each edge gives
another scalar: divided by the number of edges of the structure and multiplied by the outer
product of the direction of the edge with itself, its sum over the edges of a structure is
a symmetric matrix, which follows the rotations of the structure as `R^T M R`.

The models of MatterGen conditioned on a property add an adapter per property
(`conditions`): before each block, a small network reads the features of an atom and the
embedding of the property of its structure, and its output, through a linear layer, is added
to the features of the atom. A `gemnet.Condition` gives this embedding and which structures
are conditioned: the adapter adds nothing to the atoms of the other ones.

```{.python notest}
from torchms.graph.geometry import triplet_cosines
from torchms.graph.transforms import AddBonds, edges_vectors
from torchms.models.layers import GemNetT, gemnet

config = gemnet.Config(
    latent_dim=256, num_elements=100, radius=7.0, hidden_dim=128, num_blocks=3,
    num_spherical=7, num_radial=128, triplet_dim=64, radial_dim=16, circular_dim=16,
    bilinear_dim=64, num_before_skip=1, num_after_skip=2, num_concat=1,
    num_atom_layers=3, envelope_exponent=5, activation_scale=0.6,
    lattice_cosines=False, predict_lattice=False,
)
model = GemNetT(config)

graphs = Compose([AddRadiusGraph(7.0, self_images=True), AddTriplets(), AddBonds()])
structures = graphs(structures)

vectors = edges_vectors(structures.graph)
output = model(
    latent,                                 # (n_struct, 256)
    structures.z - 1,                       # index of the element, from hydrogen
    structures.graph,
    structures.triplets,
    structures.bonds,
    vectors,
    triplet_cosines(vectors, structures.triplets),
    structures.lattice,                     # read with lattice_cosines
)
output.features, output.move, output.lattice   # lattice is None without predict_lattice
```

## Hyperparameters

Every field of `gemnet.Config` is required but `conditions`: a model gives the values of its
`config.yaml`.

| Field | Meaning | In CDVAE | In MatterGen |
|---|---|---|---|
| `latent_dim` | size of the vector of a structure given to its atoms | 256 | 512, the embedding of the time |
| `num_elements` | rows of the embedding of the elements, from hydrogen | 100 | 101, the last one the mask |
| `radius` | radius of the graph and of the bases, in Å | 7.0 | 7.0 |
| `hidden_dim` | features of the atoms and of the edges | 128 | 512 |
| `num_blocks` | interaction blocks | 3 | 4 |
| `num_spherical` | degrees of the basis of the angles | 7 | 7 |
| `num_radial` | gaussians of the lengths of the edges | 128 | 128 |
| `triplet_dim` | features of the edges in a triplet | 64 | 64 |
| `radial_dim`, `circular_dim` | projections of the bases | 16, 16 | 16, 16 |
| `bilinear_dim` | features of the edges after the triplets | 64 | 64 |
| `num_before_skip`, `num_after_skip`, `num_concat` | residual layers of a block | 1, 2, 1 | 1, 2, 1 |
| `num_atom_layers` | residual layers of the updates of the atoms | 3 | 3 |
| `envelope_exponent` | exponent of the smooth cutoff | 5 | 5 |
| `activation_scale` | the SiLU is divided by it | 0.6 | 0.6 |
| `lattice_cosines` | the edges read their cosines with the lattice vectors | false | true |
| `predict_lattice` | a symmetric matrix for each structure | false | true |
| `conditions` | names of the properties with an adapter, `()` by default | `()` | `("chemical_system",)`, `("space_group",)` or `("dft_band_gap",)` for the fine-tuned models |

## What to know

- **The graph must be symmetric**, with the pairs of an edge and its reverse attached
  (`AddBonds`): after `KeepNearestNeighbors`, use `MirrorEdges`, as
  `CDVAE.decoder_transform()` does.
- The number of triplets grows with the square of the number of neighbours: 20 neighbours
  give 380 triplets per atom.
- The vector of an atom points towards itself along an edge with a positive scalar: an edge
  pushes or pulls its atom.
- The network has scaling factors which cdvae fitted to keep the variance of its features.
  They are in the weights of CDVAE; a network built here has them at 1.
- GemNet also has a read-out of an energy per atom. Its parameters are kept, to load the
  weights of cdvae, and it is not computed.
- Two atoms at the same position give NaN.
- The length of an edge is scaled by the radius as the Open Catalyst Project does, times
  the inverse of the radius rather than divided by it. The gaussians are narrow (1/127 of
  the radius) and the trained networks amplify them: in MatterGen, the division changed the
  outputs by 1e-2, a thousand times the rounding of single precision.
