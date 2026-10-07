# Graphs

`pycrystalgen.graph` builds graphs of crystals under periodic boundary conditions. An edge
goes from an atom of the cell to an atom of the cell or of one of its periodic images.

## Radius graph

```python
from pycrystalgen.data import Structures
from pycrystalgen.graph.cutoff import cutoff

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
from pycrystalgen.graph.cutoff import nearest_neighbors

graph = nearest_neighbors(cutoff(Structures.diamond(), 4.0), 4)
graph.num_edges   # tensor([32]): the 4 nearest neighbours of 8 atoms
```

Edges of the result are sorted by source atom, then by length.

## Chemical bonds

`atomic_radius` connects two atoms closer than a multiple of the sum of their atomic radii, and
can keep only the first shell of distances:

```python
from pycrystalgen.graph.atomic_radius import atomic_radius

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

`pycrystalgen.graph.sparse.scatter` is the underlying function (`sum`, `mean`, `amax`, ...).

## Hash of a structure

`color_refinement` gives a hash of each structure which does not depend on the order of its
atoms, from a graph of its bonds:

```python
from pycrystalgen.graph.coloring import color_refinement

hashes = color_refinement(structures, atomic_radius(structures))   # (n_struct, 4) uint32
```

!!! warning
    Graphs built before the fix of distances in non orthogonal cells (see
    [Graph conventions](../design/graphs.md)) were wrong for these cells. Hashes computed
    from them must be computed again.
