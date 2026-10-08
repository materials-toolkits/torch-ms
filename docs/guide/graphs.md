# Graphs

`torchms.graph` builds graphs of crystals under periodic boundary conditions. An edge
goes from an atom of the cell to an atom of the cell or of one of its periodic images.

## Radius graph

```python
from torchms.data import Structures
from torchms.graph.cutoff import cutoff

structures = Structures.zincblende()
graph = cutoff(structures, 3.0)

graph.edge_index          # (2, n_edges): source and target atoms
graph.edge_shift          # (n_edges, 3): periodic image of the target
graph.num_edges           # tensor([32])
graph.edges_distances()   # (n_edges,) in Å
```

The vector of an edge, in fractional coordinates, is
`x[target] + edge_shift - x[source]`.

By default an atom is never connected to itself. In a cell smaller than the radius an atom has
its own periodic images as neighbours; ask for them with `self_images=True`:

```python
import torch

cubic = Structures(lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=torch.tensor([29]))
cutoff(cubic, 3.5).num_edges                     # tensor([0])
cutoff(cubic, 3.5, self_images=True).num_edges   # tensor([6])
```

Fractional coordinates can be inside or outside the cell.

## Nearest neighbours

```python
from torchms.graph.cutoff import nearest_neighbors

graph = nearest_neighbors(cutoff(Structures.diamond(), 4.0), 4)
graph.num_edges   # tensor([32]): the 4 nearest neighbours of 8 atoms
```

Edges of the result are sorted by source atom, then by length.

## Chemical bonds

`atomic_radius` connects two atoms closer than a multiple of the sum of their atomic radii, and
can keep only the first shell of distances:

```python
from torchms.graph.atomic_radius import atomic_radius

graph = atomic_radius(structures, scaling=1.1)
```

The graph is undirected: every edge has its reverse.

## Batches

Graph functions work on batches. `graph.edge_batch` is the structure of each edge and
`graph.edge_ptr` the index of the first edge of each structure, as `batch` and `ptr` for atoms.

```python
batch = Structures.collate([Structures.diamond(), Structures.zincblende()])
graph = cutoff(batch, 3.0)
graph.num_edges    # edges of each structure
```

## Aggregation

```python
energy_per_atom = graph.aggregate_edges_to_nodes(edge_values)       # sum over the edges of an atom
per_structure = graph.aggregate_nodes_to_structures(atom_values)
```

`torchms.graph.sparse.scatter` is the underlying function (`sum`, `mean`, `amax`, ...).

## Angles and dihedrals

Models with terms on three or four atoms need the pairs and the chains of edges of a graph.
`triplets` gives the pairs of edges starting from the same atom, `quadruplets` the chains of
three edges:

```python
from torchms.graph.geometry import dihedral_angles, triplet_cosines
from torchms.graph.line_graph import quadruplets, triplets

graph = cutoff(structures, 5.0)
vectors = graph.edges_vectors()                      # (n_edges, 3), in Å

triplet_index, triplet_batch, triplet_ptr = triplets(graph)
cosines = triplet_cosines(vectors, triplet_index)    # angle at the shared atom

short = graph.edges_distances() <= 3.0               # a shorter cutoff for the chains
quadruplet_index, quadruplet_batch, quadruplet_ptr = quadruplets(graph, short)
angles = dihedral_angles(vectors, quadruplet_index)  # around the middle edge
```

`triplet_index` (2, n_triplets) and `quadruplet_index` (3, n_quadruplets) hold indices of
edges. An atom with d edges has d (d - 1) triplets, and the number of chains grows with the
cube of the number of neighbours: both functions take a mask of the edges to keep, for a
shorter cutoff than the one of the graph.

`line_graph` is the function behind them, the pairs of edges head to tail (u→v, v→w): its
result is the edge index of the graph whose nodes are the edges, and it applies to its own
result. `reverse_edges` gives the reverse of each edge, `batched_join` the pairs of elements
with the same key.

## Graphs as transforms

The functions above return a graph. To build graphs while loading a dataset, each of them
also exists as a **transform**, a step from `Structures` to `Structures` which attaches what
it builds to the structures. Transforms are chained with `Compose`, after the
[augmentations](augmentation.md), and given to a dataset (see
[Loading datasets](datasets.md#building-graphs) for the dataset and its loader):

```python
from torch.utils.data import DataLoader
from torchms.data import load_dataset
from torchms.data.augmentation import Compose, LatticeRotation
from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets

transform = Compose(
    [
        LatticeRotation(),                           # augmentations first
        AddRadiusGraph(5.0, geometry=True),          # the graph
        AddTriplets(radius=3.0, geometry=True),      # then what is built on it
        AddQuadruplets(radius=2.5, geometry=True),
    ]
)

dataset = load_dataset("mp-20", "train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

for structures in loader:               # always a Structures batch
    structures = structures.to("cuda")  # moves the graph and the tuples too

    structures.graph.edge_index         # (2, n_edges)
    structures.graph.edge_distance      # (n_edges,) in Å
    structures.triplets.index           # (2, n_triplets): edges of the graph
    structures.triplets.angle           # (n_triplets,) in radians
    structures.quadruplets.angle        # (n_quadruplets,) dihedral angles
```

Given to the dataset, the transforms run on the CPU, in the workers of the loader, on the
whole batch at once. The same transforms can run on the GPU instead, see
[Where to build the graphs](#where-to-build-the-graphs). The collate of the loader only
batches: it never builds a graph.

| Transform | Arguments | Needs | Sets | Removes | Function |
|---|---|---|---|---|---|
| `AddRadiusGraph` | `radius`, `self_images=False`, `geometry=False` | — | `graph` | `line_graph`, `triplets`, `quadruplets` | `cutoff` |
| `AddAtomicRadiusGraph` | `scaling=2.0`, `ensure_gap=0.5`, `geometry=False` | — | `graph` | `line_graph`, `triplets`, `quadruplets` | `atomic_radius` |
| `KeepNearestNeighbors` | `k` | `graph` | `graph`, with fewer edges | `line_graph`, `triplets`, `quadruplets` | `nearest_neighbors` |
| `KeepRadius` | `radius` | `graph` | `graph`, with fewer edges | `line_graph`, `triplets`, `quadruplets` | — |
| `AddLineGraph` | `radius=None`, `backtrack=False`, `geometry=False` | `graph` | `line_graph` | — | `edge_pairs` |
| `AddTriplets` | `radius=None`, `geometry=False` | `graph` | `triplets` | — | `triplets` |
| `AddQuadruplets` | `radius=None`, `geometry=False` | `graph` | `quadruplets` | — | `quadruplets` |

- **Add** transforms attach a field, **Keep** transforms remove edges of the graph.
- A transform which needs a graph raises a `ValueError` when the structures have none.
- A transform which replaces the graph removes the line graph, the triplets and the
  quadruplets: they point to the edges of the old graph. Put the `Keep` steps before them.
- `radius` of the three last transforms selects the edges which can be part of a pair or a
  chain (the ones not longer than it). The graph keeps all its edges, and the indices still
  point to them: a model can use 5 Å for its bonds and 3 Å for its angles.
- `geometry=True` also attaches the geometry of what is built: vectors and lengths of the
  edges, angles of the pairs, dihedral angles of the chains. It is computed once, in the
  workers. Without it these fields are `None`.
- The transforms do not modify the structures they are given, they return new ones sharing
  the same tensors.

Typical chains:

```python
Compose([AddRadiusGraph(5.0)])                                 # a radius graph
Compose([AddRadiusGraph(8.0), KeepNearestNeighbors(12)])       # 12 nearest neighbours
Compose([AddAtomicRadiusGraph(1.1), AddLineGraph()])           # bonds and their line graph
Compose([AddRadiusGraph(5.0), AddTriplets(3.0), AddQuadruplets(2.5)])
```

Augmentations change the atoms or the cell, which a graph already built does not follow:
they return structures with **nothing attached**. Build the graphs last.

### Where to build the graphs

A transform does not depend on where it runs: it builds its graph on the device of the
structures it is given. There are two places to call it.

**In the workers, on the CPU.** The transform is given to the dataset. Each worker reads a
batch, builds its graphs on one thread, and sends the batch to the training process. The GPU
does nothing for the loading.

```python
dataset = load_dataset("mp-20", "train", transform=Compose([augment, graphs]))
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

for structures in loader:
    structures = structures.to("cuda")
```

**On the GPU.** The dataset keeps the augmentations only, and `OnDevice` wraps the loader: it
moves each batch to the device as it comes out of the loader, then applies its transform
there, in the training process.

```python
from torchms.data import OnDevice

dataset = load_dataset("mp-20", "train", transform=augment)   # CPU, in the workers
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
loader = OnDevice(loader, "cuda", transform=graphs)           # GPU, on each batch

for structures in loader:               # already on the GPU, with its graphs
    ...
```

`OnDevice` is iterated like the loader, once per epoch, and has its length. Without transform
it only moves the batches. The two ways give the same graphs.

| Your case | Build the graphs | Why |
|---|---|---|
| 8 CPU cores or more for the loading, and a model which keeps the GPU busy | in the workers | 65 to 95 % of the rate of the GPU way, and the GPU is left to the model |
| Few CPU cores (4 workers or less) | on the GPU | 1.4 to 5 times faster than the workers can be |
| The loading limits the training: the GPU waits for batches | on the GPU | it is the fastest way measured, with 2 to 4 workers reading |
| No GPU | in the workers, 8 or more | — |
| Large batches (512 and more) | on the GPU | its rate grows with the batch, the one of a CPU thread decreases after 128 |
| Small batches, down to one structure | in the workers | a GPU is slower than a CPU thread on one structure |
| Evaluation or inference without loader | on the device of the model | call the transform on the batch |
| Graphs needed by the transform of the dataset itself (filter, statistics) | in the workers | they must exist before the batch leaves the dataset |

The numbers behind this table, for two datasets, each transform, 0 to 16 workers and both
ways, are in [Loader and graph construction: measures](../design/loader.md), with the script
to measure your own machine and dataset.

## Fields attached to the structures

Four fields of `Structures` hold what the transforms build. Each one is `None` until a
transform sets it.

| Field | Type | Elements | Set by |
|---|---|---|---|
| `structures.graph` | `Graph` | edges: pairs of atoms | `AddRadiusGraph`, `AddAtomicRadiusGraph` |
| `structures.line_graph` | `EdgeTuples` | pairs of edges u→v, v→w | `AddLineGraph` |
| `structures.triplets` | `EdgeTuples` | pairs of edges i→j, i→k | `AddTriplets` |
| `structures.quadruplets` | `EdgeTuples` | chains of edges u→v, v→w, w→x | `AddQuadruplets` |

There are three levels of indices, each one pointing to the level below:

```text
atoms          rows of structures.x and structures.z                 0 … n_atoms - 1
  ↑ graph.edge_index holds indices of atoms
edges          columns of graph.edge_index                           0 … n_edges - 1
  ↑ line_graph.index, triplets.index, quadruplets.index hold indices of edges
tuples         columns of line_graph.index, triplets.index, …        0 … n_tuples - 1
```

All the indices are **global in the batch**: an atom is numbered among the atoms of the whole
batch, an edge among the edges of the whole batch. `ptr` and `batch` fields give the
structure of each element, as for the atoms.

On four atoms u, v, w, x bonded in a chain, with the edges numbered a to f:

```text
        a →         c →         e →
   u ─────────  v ─────────  w ─────────  x
        ← b         ← d         ← f

graph.edge_index      a: u→v   b: v→u   c: v→w   d: w→v   e: w→x   f: x→w

line_graph.index      (a, c)   u→v then v→w   angle at v between u and w
                      (c, e)   v→w then w→x   angle at w between v and x
                      (f, d)   x→w then w→v   the same angles, from the other end
                      (d, b)
                      (a, b)   only with backtrack=True: u→v then v→u

triplets.index        (b, c)   v→u and v→w    angle at v between u and w
                      (c, b)   the same angle, the two edges exchanged
                      (d, e)   w→v and w→x    angle at w between v and x
                      (e, d)

quadruplets.index     (a, c, e)   u→v, v→w, w→x   dihedral angle around v-w
                      (f, d, b)   x→w, w→v, v→u   the same chain from the other end
```

The line graph and the triplets describe the same angles. They differ by the direction of the
first edge: in the line graph it **arrives** at the shared atom, in the triplets it **leaves**
it. The line graph is the one to chain (its pairs are the edges of a graph whose nodes are
the edges), the triplets are the ones to sum over the neighbours of an atom.

### Graph: `structures.graph`

`n_edges` is the number of edges of the batch, `n_struct` its number of structures.

| Field | Shape, dtype | Unit | Meaning | In a batch |
|---|---|---|---|---|
| `graph.edge_index` | (2, n_edges), int64 | index of atom | Row 0: the **sender** of the edge, an atom of the cell. Row 1: its **receiver**. Both are rows of `structures.x`. | shifted by the atoms of the structures before |
| `graph.edge_shift` | (n_edges, 3), int64 | lattice vectors | Periodic image of the receiver: the edge goes from `x[sender]` to `x[receiver] + edge_shift` in fractional coordinates. Zero inside the cell. | concatenated |
| `graph.num_atoms` | (n_struct,), int64 | count | Atoms of each structure, same as `structures.num_atoms`. | concatenated |
| `graph.num_edges` | (n_struct,), int64 | count | Edges of each structure. Can be 0. | concatenated |
| `graph.edge_batch` | (n_edges,), int64 | index of structure | Structure of each edge, in ascending order. | rebuilt |
| `graph.edge_ptr` | (n_struct + 1,), int64 | index of edge | First edge of each structure, and `n_edges` at the end: the edges of the structure `i` are `edge_ptr[i]:edge_ptr[i + 1]`. | rebuilt |
| `graph.edge_vector` | (n_edges, 3), float32 | Å | Cartesian vector from the sender to the receiver. **Only with `geometry=True`**, else `None`. | concatenated |
| `graph.edge_distance` | (n_edges,), float32 | Å | Length of the edge, the norm of `edge_vector`. **Only with `geometry=True`**, else `None`. | concatenated |
| `graph.structures` | `Structures` | — | The structures of the graph, sharing the tensors of `structures`, with nothing attached. Used by `graph.edges_vectors()`. | the batch |

Order of the edges: by structure, then as the function building the graph gives them. After
`KeepNearestNeighbors` they are sorted by sender, then by length. `AddRadiusGraph` and
`AddAtomicRadiusGraph` give both directions of each pair of atoms: every edge has its
reverse (`reverse_edges`). `KeepNearestNeighbors` can keep an edge without its reverse.

Without `geometry=True`, the same quantities are computed on request by
`graph.edges_vectors()` and `graph.edges_distances()`.

### Line graph: `structures.line_graph`

`n_pairs` is the number of pairs of the batch.

| Field | Shape, dtype | Unit | Meaning | In a batch |
|---|---|---|---|---|
| `line_graph.index` | (2, n_pairs), int64 | index of edge | Row 0: the edge u→v. Row 1: the edge v→w, which leaves the atom the first one arrives at. Both are columns of `graph.edge_index`. | shifted by the edges of the structures before |
| `line_graph.num_edges` | (n_struct,), int64 | count | Edges of the graph per structure, same as `graph.num_edges`: the size of what `index` points to. | concatenated |
| `line_graph.num_tuples` | (n_struct,), int64 | count | Pairs of each structure. Can be 0. | concatenated |
| `line_graph.batch` | (n_pairs,), int64 | index of structure | Structure of each pair, in ascending order. | rebuilt |
| `line_graph.ptr` | (n_struct + 1,), int64 | index of pair | First pair of each structure, and `n_pairs` at the end. | rebuilt |
| `line_graph.angle` | (n_pairs,), float32 | radians, in [0, π] | Angle at the atom v between u and w. **Only with `geometry=True`**, else `None`. | concatenated |

Order of the pairs: by the atom v they share. Without `backtrack=True`, an edge is not followed
by its reverse; it is still followed by an edge to another periodic image of the atom it comes
from.

### Triplets: `structures.triplets`

`n_triplets` is the number of triplets of the batch.

| Field | Shape, dtype | Unit | Meaning | In a batch |
|---|---|---|---|---|
| `triplets.index` | (2, n_triplets), int64 | index of edge | Row 0: the edge i→j. Row 1: the edge i→k, another edge leaving the same atom. Both are columns of `graph.edge_index`. | shifted by the edges of the structures before |
| `triplets.num_edges` | (n_struct,), int64 | count | Edges of the graph per structure, same as `graph.num_edges`. | concatenated |
| `triplets.num_tuples` | (n_struct,), int64 | count | Triplets of each structure. Can be 0. | concatenated |
| `triplets.batch` | (n_triplets,), int64 | index of structure | Structure of each triplet, in ascending order. | rebuilt |
| `triplets.ptr` | (n_struct + 1,), int64 | index of triplet | First triplet of each structure, and `n_triplets` at the end. | rebuilt |
| `triplets.angle` | (n_triplets,), float32 | radians, in [0, π] | Angle at the atom i between j and k. **Only with `geometry=True`**, else `None`. | concatenated |

Order of the triplets: by the atom i. Both orders of the two edges are listed, (i→j, i→k) and
(i→k, i→j): an atom with d edges has d (d − 1) triplets. Two periodic images of the same atom
are two different neighbours.

### Quadruplets: `structures.quadruplets`

`n_quadruplets` is the number of quadruplets of the batch.

| Field | Shape, dtype | Unit | Meaning | In a batch |
|---|---|---|---|---|
| `quadruplets.index` | (3, n_quadruplets), int64 | index of edge | Rows 0, 1, 2: the edges u→v, v→w and w→x of the chain. They are columns of `graph.edge_index`. | shifted by the edges of the structures before |
| `quadruplets.num_edges` | (n_struct,), int64 | count | Edges of the graph per structure, same as `graph.num_edges`. | concatenated |
| `quadruplets.num_tuples` | (n_struct,), int64 | count | Quadruplets of each structure. Can be 0. | concatenated |
| `quadruplets.batch` | (n_quadruplets,), int64 | index of structure | Structure of each quadruplet, in ascending order. | rebuilt |
| `quadruplets.ptr` | (n_struct + 1,), int64 | index of quadruplet | First quadruplet of each structure, and `n_quadruplets` at the end. | rebuilt |
| `quadruplets.angle` | (n_quadruplets,), float32 | radians, in [−π, π] | Dihedral angle between the planes (u, v, w) and (v, w, x), around the bond v-w: 0 when u and x are on the same side (cis), ±π on opposite sides (trans). Its sign changes in the mirror image of the chain. Undefined (0) when three atoms are aligned. **Only with `geometry=True`**, else `None`. | concatenated |

A chain never goes back along an edge. A chain and the same chain from x to u are both listed.
The number of chains grows with the cube of the number of neighbours: give a short `radius`.

### In a batch

The loader, `Structures.collate` and `Structures.separate` keep everything consistent:

- the atoms of `graph.edge_index` are shifted by the number of **atoms** of the structures
  before, as `structures.ptr`;
- the edges of `line_graph.index`, `triplets.index` and `quadruplets.index` are shifted by the
  number of **edges** of the structures before, as `graph.edge_ptr`;
- per structure counts and geometry are concatenated, `batch` and `ptr` fields are rebuilt.

```python
batch = Structures.collate([structures, structures])   # diamond twice, 32 edges each

batch.graph.edge_ptr                     # tensor([ 0, 32, 64])
batch.triplets.ptr                       # tensor([  0,  96, 192])

# the triplets of the second structure, and their edges
second = batch.triplets.index[:, batch.triplets.ptr[1] : batch.triplets.ptr[2]]
second.min()                             # tensor(32): edges of the second structure

# values per triplet, summed per structure
from torchms.graph.sparse import scatter
scatter(values, batch.triplets.batch, len(batch.num_atoms))
```

A batch needs the same fields in every structure: collating structures where only some have a
graph, tuples of edges or their geometry raises a `ValueError`. `separate` gives each
structure back with its graph and its tuples, numbered from 0.

### Reading the fields

Everything attached can be recomputed from the indices:

```python
graph, triplets = structures.graph, structures.triplets

# vector of the edge e
sender, receiver = graph.edge_index[:, e]
fractional = structures.x[receiver] + graph.edge_shift[e] - structures.x[sender]
vector = fractional @ structures.lattice[graph.edge_batch[e]]   # graph.edge_vector[e]
vector.norm()                                                   # graph.edge_distance[e]

# angle of the triplet t
first, second = triplets.index[:, t]
a, b = graph.edge_vector[first], graph.edge_vector[second]
torch.acos((a * b).sum() / (a.norm() * b.norm()))               # triplets.angle[t]

# atoms of the triplet t: the center, then its two neighbours
graph.edge_index[0, first], graph.edge_index[1, first], graph.edge_index[1, second]
```

Features of the atoms are gathered on the edges with `graph.edge_index`, features of the edges
on the tuples with `triplets.index`, and summed back with `scatter`:

```python
messages = edge_features[triplets.index[1]]                     # (n_triplets, F)
edge_update = scatter(messages, triplets.index[0], graph.edge_index.shape[1])
```

## Hash of a structure

`color_refinement` gives a hash of each structure which does not depend on the order of its
atoms, from a graph of its bonds:

```python
from torchms.graph.coloring import color_refinement

hashes = color_refinement(structures, atomic_radius(structures))   # (n_struct, 4) uint32
```

The hash depends on the number of atoms: a supercell has another hash than its cell.
`refined_coloring` gives the colors of the atoms the hash is computed from, which are the same
in a cell and in its supercells:

```python
from torchms.graph.coloring import refined_coloring

colors = refined_coloring(structures, atomic_radius(structures))   # (n_atoms, 4) uint32
```

Two structures have the same hash when they have the same elements, the same number of
atoms and graphs of bonds the refinement cannot tell apart. The hash does not see distances:
it groups candidates, it does not replace a comparison of structures. On LeMat-Bulk
(measured on 2026-10-07 with the default `atomic_radius`):

| Split | Structures | Different hashes | Structures minus hashes | Rate |
|---|---|---|---|---|
| `all` | 5,335,299 | 4,694,471 | 640,828 | 12.01 % |
| `unique` | 4,927,038 | 4,564,507 | 362,531 | 7.36 % |

`unique` keeps one structure per fingerprint of LeMat-Bulk: 7.36 % of the structures it holds
for different share a hash with another one. Of the 408,261 duplicates it drops, about 130,000
have a hash which no kept structure has. Which shared hashes are the same crystal, the same
bonds with other distances, or crystals the refinement confuses, was not measured.

!!! warning
    Graphs built before the fix of distances in non orthogonal cells (see
    [Graph conventions](../design/graphs.md)) were wrong for these cells. Hashes computed
    from them must be computed again.
