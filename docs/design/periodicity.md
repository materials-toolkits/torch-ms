# Periodicity

A batch can hold crystals, molecules, or both. `Structures.pbc`, shape (n_struct, 3), says
along which of its lattice vectors each structure is periodic: all `True` for a crystal, the
default, all `False` for a molecule.

## One rule for the positions

The position of an atom in Å is `x @ lattice`, for every structure:

- for a crystal, `lattice` is its cell and `x` its fractional coordinates;
- a molecule has no cell. Its `lattice` is only a frame, **the identity**, and `x` is then its
  positions in Å.

```python
molecule = Structures.molecule(positions, z)   # identity lattice, pbc False
molecule.positions()                           # x @ lattice, for any structure
```

The alternative was a second field of cartesian positions for the molecules. Every function
which computes a vector between two atoms would then have two cases. With one rule, the edge
vectors, the line graph, the triplets and the models are the same code for a crystal and a
molecule, and a mixed batch needs no branch.

The frame of a molecule stops being the identity when it is rotated: `LatticeRotation` rotates
the lattice of a molecule as the one of a crystal, and `x` is unchanged. Read positions with
`positions()`, not `x`.

## Graphs

`cutoff` and `atomic_radius` connect an atom to the periodic images of the others only along
the periodic lattice vectors. The edges of a molecule are between its atoms and their
`edge_shift` is zero.

In `periodic_pairs`, the images of a pair of atoms are a box of `count` cells from `low` along
each lattice vector. Along a vector which is not periodic, `low = 0` and `count = 1`: this is
the whole change, a `torch.where` on tensors which were already per structure and per axis.
There is no loop and no test on what the batch holds. The coordinates of a molecule are not
wrapped before the search.

A structure periodic along some of its lattice vectors only (a slab, a wire) follows the same
rule, and is tested against ASE. No dataset published has one.

## Augmentations

| Transform | On a molecule |
|---|---|
| `AtomPermutation` | as a crystal |
| `LatticeRotation` | rotates the molecule |
| `AtomTranslation` | translates it by at most 1 Å along each axis of its frame, **without wrapping** |
| `LatticeBasisChange` | unchanged: it has no lattice to describe in another basis |
| `Supercell` | unchanged: it cannot be repeated |

`make_supercell` raises an error when asked to repeat a structure along a vector which is not
periodic.

## Files

The HDF5 files have a `pbc` array. **A file without it holds crystals**: the files written
before molecules existed are read as they were.

## What is not covered

- The energy estimator and the convex hull are fitted on crystals. The pretrained models run
  on a molecule (its graph has no periodic image), but nothing here says their energies are
  good for molecules, and the correction and the hull of LeMat-Bulk do not apply.
- The radius graph lists all the pairs of atoms of each structure. This is fast for molecules
  and for the usual crystals, and heavy for cells of thousands of atoms.
