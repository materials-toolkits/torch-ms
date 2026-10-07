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

## What is tested

Against the neighbour list of ASE, on triclinic cells, alone and in batches:

- `cutoff`: same edges, shifts and distances, with coordinates inside and outside the cell;
- `atomic_radius`: same edges, shifts and distances;
- `nearest_neighbors`: the k shortest edges of each atom.

Not tested on non orthogonal cells: `color_refinement`, and `edges_distances` after
`Graph.collate` or `Graph.separate`.
