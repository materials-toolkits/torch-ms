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
