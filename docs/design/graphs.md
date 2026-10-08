# Graph conventions

## Lattice

The **rows** of `lattice` are the lattice vectors. A fractional vector `f` is the cartesian
vector `f @ lattice`:

```python
cartesian = torch.bmm(fractional.unsqueeze(1), lattice[batch]).squeeze(1)
```

## Edges

An edge goes from `edge_index[0]` (source, in the cell) to `edge_index[1]` (target).
`edge_shift` is the periodic image of the target: its vector is

```python
fractional = x[target] + edge_shift - x[source]
```

The shift is relative to the coordinates as given, which can be outside `[0, 1)`: `cutoff`
wraps atoms in the cell to search neighbours, then expresses shifts for the given coordinates.

Graphs of `cutoff` and `atomic_radius` contain both directions of each pair. Edges are grouped
by structure, in the order of the batch.

## What is attached to the structures

A graph built by a transform (`torchms.graph.transforms`) is a field of `Structures`, as the
tuples of its edges: `graph`, `line_graph`, `triplets`, `quadruplets`. The fields are
described in the [guide](../guide/graphs.md#fields-attached-to-the-structures).

One object goes through the transforms, the loader and the model. The alternative, a graph
holding its structures, was the convention of the functions (`cutoff` returns a `Graph` with a
reference to its structures) and is kept for them. It does not chain: a line graph would hold
a graph which holds structures, and every step would have another type.

There is no reference cycle. `structures.graph.structures` is a `Structures` sharing the same
tensors, with nothing attached: `graph.edges_vectors()` works on an attached graph as on any
other one.

### Batching

Indices are global in the batch at each level, so `collate` shifts each level by the size of
the level below.

| Attached | Rule in `Structures.collate` |
|---|---|
| `graph.edge_index` | shifted by the number of **atoms** of the elements before |
| `graph.edge_shift`, `graph.num_atoms`, `graph.num_edges`, `graph.edge_vector`, `graph.edge_distance` | concatenated |
| `graph.edge_batch`, `graph.edge_ptr` | rebuilt from `num_edges` |
| `graph.structures` | the new batch |
| `index` of `line_graph`, `triplets`, `quadruplets` | shifted by the number of **edges** of the elements before |
| their `num_edges`, `num_tuples`, `angle` | concatenated |
| their `batch`, `ptr` | rebuilt from `num_tuples` |

- An element of the list given to `collate` can be a batch itself. The shift of an element is
  the total of the elements before it, atoms for the graph and edges for the tuples.
- Tuples carry `num_edges`, the number of edges of the graph per structure, which is all
  their collate needs: it does not look at the graph.
- A field is in every element or in none. A mix raises a `ValueError`, for the graph, for each
  kind of tuples and for their geometry: dropping the field silently would give a batch which
  looks valid and trains a model on less than expected.
- `separate` removes the same shifts, so `collate(separate(batch))` is the batch.

There are two ways to a batch with graphs, which give the same graphs:

- `HDF5Dataset` reads a batch and applies its transform to it: the graphs are built on the
  batch, and the collate of the loader receives one element.
- A dataset returning one structure per index (`CifDataset`) applies its transform to each
  structure: the graphs are built one by one and really collated.

The first one is faster, see [Loader and graph construction](loader.md).

### Augmentations

The five actions of `torchms.data.augmentation` return structures with nothing attached. A
permutation changes the indices of the atoms, a supercell their number, a translation with
wrapping and a change of basis the periodic images. Only the rotation would leave a graph
valid, and its vectors would still be wrong. Removing everything in every case is one rule to
remember: augmentations first, graphs last.

## Self images

An atom is never its own neighbour at a distance of zero. Its periodic images are neighbours
when the cell is smaller than the radius: `cutoff(..., self_images=True)` includes them,
the default does not. The energy model needs them.

## Two bugs fixed, and what they affected

Both were found by comparing with the neighbour list of ASE while porting the energy model,
and are covered by tests on triclinic cells since.

**Distances in non orthogonal cells.** Edge distances multiplied the fractional vector by the
lattice on the wrong side (`lattice @ f` instead of `f @ lattice`). The two are equal only
when the lattice matrix is symmetric, which is the case of cubic and orthorhombic cells in the
standard orientation. For every other cell, the graphs of `cutoff` and `atomic_radius` and
`Graph.edges_distances` were wrong.

!!! warning
    Graphs, and hashes from `color_refinement`, computed before this fix on triclinic,
    monoclinic, hexagonal or rhombohedral cells must be computed again.

**Shift of an atom at 1.0.** The shift of an edge was read from the integer part of the
coordinate of the target in the supercell, which is off by one for a fractional coordinate of
exactly 1.0. Such coordinates exist in the datasets because of the rounding to 32 bits (33,624
of 148 million in LeMat-Bulk). Shifts are now the difference between the coordinates in the
supercell and in the cell.

**Hash of a structure in a batch.** `color_refinement` stops when the partition of the atoms
of every structure is stable, and drops the last round. A structure which was stable before
the others kept the round which found it stable, so it was refined one more time in a batch
than alone and its hash depended on the other structures of the batch. The round which finds
a structure stable is now dropped for this structure.

!!! warning
    Hashes from `color_refinement` computed in batches of several structures before this fix
    must be computed again.

**Refinement which never stopped.** The color of an atom was replaced by a hash of the colors
of its neighbours alone. The color of the atom itself was lost: lithium and sodium bonded to
the same oxygen got the same color, and on some structures the partition of the atoms
alternated between two groupings, so the refinement never stopped (27 of the first 576
structures of LeMat-Bulk). The new color is now a hash of the color of the atom and of the
colors of its neighbours, as in the Weisfeiler-Lehman test: a round only splits the classes.

!!! warning
    This changes every hash of `color_refinement` and every color of `refined_coloring`:
    the ones computed before this fix must be computed again.

## What is tested

Against the neighbour list of ASE, on triclinic cells, alone and in batches:

- `cutoff`: same edges, shifts and distances, with coordinates inside and outside the cell;
- `atomic_radius`: same edges, shifts and distances;
- `nearest_neighbors`: the k shortest edges of each atom.

`color_refinement` and `refined_coloring`: same hashes and colors for a structure alone and in
a batch, same hashes after the augmentations of `torchms.data.augmentation`, on cubic and
triclinic cells.

Graph transforms and batching, on triclinic cells of different sizes, one of them smaller than
the radius, for ten chains of transforms:

- the chain applied to a batch gives the same graph and the same tuples as the batch of the
  chain applied to each structure, and as a batch of batches;
- after `collate`, `separate` and `to`, every index points to an atom or an edge of its
  structure, and the distances, angles and dihedral angles recomputed from the indices are
  the ones attached;
- `collate(separate(batch))` is the batch;
- the transforms give the indices and the geometry of `LineGraphNetwork.build_graph`;
- a `DataLoader` on an HDF5 file and on CIF files gives the graphs of the chain applied
  directly, with workers started by spawn.
