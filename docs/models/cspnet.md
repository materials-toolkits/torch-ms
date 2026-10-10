# CSPNet

`torchms.models.layers.CSPNet` is CSPNet, the network of DiffCSP: a network on every pair of
atoms of a structure, which reads fractional coordinates and gives a vector for each atom
and a 3 x 3 matrix for each structure. On demand it reads the element of an atom as a vector
of weights and gives such a vector for each atom.

| | |
|---|---|
| Classes | `CSPNet`, `CSPNetConfig` |
| Output | a matrix of each structure, shape `(n_struct, 3, 3)`, a vector of each atom in fractional coordinates, shape `(n_atoms, 3)`, and with `predict_elements` a value of each element for each atom, shape `(n_atoms, num_elements)`, `None` otherwise |
| Reads | a complete graph with self loops, fractional coordinates, lattices, and a vector of each structure |
| Paper | [Crystal Structure Prediction by Joint Equivariant Diffusion](https://arxiv.org/abs/2309.04475), Jiao, Huang, Lin, Han, Chen, Lu and Liu (arXiv:2309.04475) |
| Rewritten from | [DiffCSP](https://github.com/jiaor17/DiffCSP) `7121d15`, the code of the authors, MIT |
| Pretrained weights | the ones of the two models of [DiffCSP](diffcsp.md), as their network |

Features live on the atoms. A layer builds a message for each pair of atoms from their
features, from the 9 inner products of the lattice vectors, and from sines and cosines of
the difference of their fractional coordinates, of period 1; an atom gets the mean of its
messages. The network reads no distance: nothing changes with a rotation of the structure
or a translation of its atoms. The matrix of a structure is a matrix read from the mean of
its atoms, times its lattice: it is rotated with the lattice.

```{.python notest}
from torchms.graph.transforms import AddCompleteGraph
from torchms.models.layers import CSPNet, CSPNetConfig

config = CSPNetConfig(
    latent_dim=256, num_elements=100, hidden_dim=512, num_layers=6, num_freqs=128,
    soft_elements=False, predict_elements=False,
)
model = CSPNet(config)

structures = AddCompleteGraph(self_loops=True)(structures)

matrices, vectors, _ = model(
    latent,                                 # (n_struct, 256)
    structures.z - 1,                       # index of the element, from hydrogen
    structures.graph,
    structures.x,                           # or noisy coordinates
    structures.lattice,                     # or noisy lattices
)
```

The coordinates and the lattices are arguments, apart from the graph: a diffusion gives
noisy ones on a graph which does not change.

With `soft_elements=True` the second argument is a vector of `num_elements` weights for
each atom, shape `(n_atoms, num_elements)`, read by a linear layer in place of the table of
the elements: a one-hot vector for a known element, anything for a noisy one. With
`predict_elements=True` the third output has one value per element for each atom. The
generation of DiffCSP sets both, to diffuse the elements.

## Hyperparameters

Every field of `CSPNetConfig` is required: a model gives the values of its `config.yaml`.

| Field | Meaning | In DiffCSP |
|---|---|---|
| `latent_dim` | size of the vector of a structure given to its atoms | 256, the embedding of the time |
| `num_elements` | rows of the embedding of the elements, from hydrogen | 100 |
| `hidden_dim` | features of the atoms and of the messages | 512 |
| `num_layers` | message passing layers | 6 |
| `num_freqs` | frequencies of the basis of the differences of the coordinates | 128 |
| `soft_elements` | the element of an atom is a vector of weights, not an index | false from a composition, true in the generation |
| `predict_elements` | the network also gives a vector of each atom, one value per element | false from a composition, true in the generation |

## What is kept from DiffCSP

Only the variant of the published weights: sines and cosines of the differences
(`dis_emb: sin`), inner products of the lattice vectors (`ip: true`), a normalization
before every layer and after the last one (`ln: true`), a complete graph (`edge_style: fc`).
The soft elements and their prediction are the `smooth` and `pred_type` of DiffCSP. The graph
of the nearest neighbours and the head which predicts a scalar are not included.

An edge of the graph of DiffCSP goes from an atom i to an atom j, and the atom i gets the
mean of its messages: i is the sender, and the messages are aggregated to the sender. The
difference of the coordinates is the one of j minus the one of i.

## Cost

A structure of n atoms has n² edges, and a layer reads `2 hidden_dim + 9 + 6 num_freqs`
values on each one: 1,801 with the hyperparameters of DiffCSP. See
[DiffCSP](diffcsp.md#comparison-with-the-original) for the comparison with the original
code.
