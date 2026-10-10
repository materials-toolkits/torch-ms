# DimeNet++

`torchms.models.layers.DimeNetPlusPlus` is DimeNet++, a network on the bonds of a graph and
on the angles between them, which gives one vector per structure. DimeNet++ is a faster
version of DimeNet, both of Gasteiger and Günnemann and their co-authors.

Here it is rewritten from the code of [CDVAE](cdvae.md), which uses it as its encoder: this
is the code it was compared with, not the one of its authors.

| | |
|---|---|
| Classes | `DimeNetPlusPlus`, `dimenet.Config` |
| Output | one vector per structure, shape `(n_struct, out_channels)` |
| Reads | a graph of bonds and its line graph: the pairs of bonds i → j, j → k |
| Paper, DimeNet++ | [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115), Gasteiger, Giri, Margraf and Günnemann (arXiv:2011.14115) |
| Paper, DimeNet | [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123), Gasteiger, Groß and Günnemann (arXiv:2003.03123) |
| Code of the authors | [gasteigerjo/dimenet](https://github.com/gasteigerjo/dimenet) |
| Rewritten from | [cdvae](https://github.com/txie-93/cdvae) `f857f59`, whose DimeNet++ comes from the Open Catalyst Project and PyTorch Geometric 2.0.1, MIT |
| Pretrained weights | the ones of [CDVAE](cdvae.md), as its encoder |

Features live on the bonds. A block updates a bond i → j from the bonds j → k leaving its
receiver towards another atom than i, in a basis of spherical Bessel functions of the length
of j → k and of spherical harmonics of the angle between the two bonds. After each block the
bonds are summed on their atom, and the atoms are reduced over the structure.

```{.python notest}
from torchms.graph.geometry import triplet_cosines
from torchms.graph.transforms import edges_vectors
from torchms.models.layers import DimeNetPlusPlus, dimenet

config = dimenet.Config(
    out_channels=16, hidden_dim=128, num_blocks=4, triplet_dim=64, basis_dim=8,
    output_dim=256, num_spherical=7, num_radial=6, radius=5.0, envelope_exponent=5,
    num_before_skip=1, num_after_skip=2, num_output_layers=3, readout="mean",
    num_elements=95,
)
model = DimeNetPlusPlus(config)

graphs = Compose([AddRadiusGraph(5.0), AddLineGraph(backtrack=True, same_atom=False)])
structures = graphs(structures)

vectors = edges_vectors(structures.graph)
output = model(
    structures.z,
    structures.graph,
    structures.line_graph,
    vectors.norm(dim=-1),
    triplet_cosines(vectors, structures.line_graph),
)                                           # (n_struct, 16)
```

## Hyperparameters

Every field of `dimenet.Config` is required: a model gives the values of its `config.yaml`.

| Field | Meaning | In CDVAE |
|---|---|---|
| `out_channels` | size of the vector of a structure | 256 |
| `hidden_dim` | features of the bonds | 128 |
| `num_blocks` | interaction blocks | 4 |
| `triplet_dim` | features of the pairs of bonds | 64 |
| `basis_dim` | projection of the bases | 8 |
| `output_dim` | hidden layers of the outputs | 256 |
| `num_spherical` | degrees of the basis of the angles | 7 |
| `num_radial` | radial functions | 6 |
| `radius` | radius of the bases, in Å | 7.0 |
| `envelope_exponent` | exponent of the smooth cutoff | 5 |
| `num_before_skip`, `num_after_skip` | residual layers of a block | 1, 2 |
| `num_output_layers` | hidden layers of an output | 3 |
| `readout` | reduction of the atoms of a structure | `mean` |
| `num_elements` | rows of the embedding, read with the atomic number | 95 |

## What to know

- The pairs of bonds must not come back to the atom they start from:
  `AddLineGraph(same_atom=False)`. The cosine is the one of the two bonds as vectors, 1 when
  k continues the bond i → j: `triplet_cosines`, not `pair_cosines`.
- The embedding is read with the atomic number itself: an element needs
  `num_elements > z`.
- A bond longer than `radius` contributes nothing. The radius is the one of the bases, not
  of the graph: the graph can be any graph of bonds.
- The frequencies of the radial basis are parameters of the network.
- Its bases lose digits in single precision for bonds much shorter than the radius: see the
  comparison of [CDVAE](cdvae.md#comparison-with-the-original).
