"""Hash of a structure from the colors of its atoms in a graph of its bonds.

The colors come from the color refinement of Weisfeiler and Lehman: an atom
starts with a color for its element and its number of bonds, then repeatedly
takes a color for its own color and the colors of its neighbours, until the
groups of atoms with the same color stop changing. The hash of a structure is
a hash of the colors of its atoms.

Two structures have the same hash when they have the same elements, the same
number of atoms and graphs of bonds the refinement cannot tell apart. The
hash does not see distances: it does not replace a comparison of structures.

How often structures share a hash was measured on LeMat-Bulk (compatible_pbe)
on 2026-10-07, with the graphs of atomic_radius and its default arguments, by
scripts/test_hash_collision.py:

| Split of the dataset              | all       | unique    |
|-----------------------------------|-----------|-----------|
| Structures                        | 5,335,299 | 4,927,038 |
| Different hashes                  | 4,694,471 | 4,564,507 |
| Structures minus different hashes |   640,828 |   362,531 |
| Rate                              |   12.01 % |    7.36 % |

"unique" keeps one structure per fingerprint of LeMat-Bulk. The 408,261
structures it drops are duplicates for LeMat-Bulk: about 278,300 of them
share a hash with a structure which is kept, about 130,000 do not. In
"unique", 362,531 structures which LeMat-Bulk holds for different share a
hash with another one. Which of them are the same crystal, crystals with the
same bonds and other distances, or different crystals the refinement confuses,
was not measured.
"""

import torch
from jaxtyping import UInt32

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.sparse import equivalent_partition, mix32, scatter

# odd constant which mixes the colors of the neighbours with the one of the atom
OWN_COLOR_MULTIPLIER = 0x9E3779B1


def initial_coloring(
    structures: Structures,
    graph: Graph,
) -> UInt32[torch.Tensor, "n_atoms 4"]:
    """Assign a 128-bit hash color to each atom based on atomic number and degree.

    Treats {z, degree} as an unordered 2-element set and hashes it into four
    independent uint32 values: mix32 each element, sum [mix^1, mix^2, mix^3, mix^4]
    across the set (power-sum construction, inspired by scatter hash), then
    mix32 each component.

    Two atoms with the same atomic number and degree receive the same color.
    Collisions between distinct (z, degree) pairs are possible but negligible.

    Args:
        structures: Batched Structures containing atomic numbers z, shape (n_atoms,).
        graph: Graph built from structures; must cover all atoms.

    Returns:
        Hash color per atom, shape (n_atoms, 4), dtype uint32.

    Examples:
        >>> from pycrystalgen.data.batching import Structures
        >>> from pycrystalgen.graph.atomic_radius import atomic_radius
        >>> structures = Structures.zincblende()
        >>> graph = atomic_radius(structures, scaling=1.1)
        >>> colors = initial_coloring(structures, graph)
        >>> colors.shape
        torch.Size([8, 4])
        >>> colors.unique(dim=0).shape[0]  # Zn and S both have degree 4 → 2 colors
        2
    """
    n_atoms = int(structures.num_atoms.sum())
    device = structures.device()

    degree = torch.bincount(graph.edge_index[1], minlength=n_atoms)

    mz = mix32(structures.z.to(torch.uint32)).to(torch.int64)
    md = mix32(degree.to(torch.uint32)).to(torch.int64)

    pz = torch.cumprod(mz.unsqueeze(-1).expand(-1, 4), dim=-1) & 0xFFFFFFFF
    pd = torch.cumprod(md.unsqueeze(-1).expand(-1, 4), dim=-1) & 0xFFFFFFFF

    return mix32(((pz + pd) & 0xFFFFFFFF).to(torch.uint32))


def _xor4(c: torch.Tensor) -> torch.Tensor:
    c = c.to(torch.int64)
    return c[:, 0] ^ c[:, 1] ^ c[:, 2] ^ c[:, 3]


def refined_coloring(
    structures: Structures,
    graph: Graph,
) -> UInt32[torch.Tensor, "n_atoms 4"]:
    """Run Weisfeiler-Lehman color refinement and return the color of each atom.

    Iteratively replaces each node's color with a hash of its own color and of
    its neighbors' colors, restricted to structures whose partition is still
    changing. A round only splits the classes of nodes with the same color:
    the refinement stops when no structure's partition changes, after at most
    as many rounds as atoms in a structure.

    Two atoms with the same environment in the graph receive the same color. The
    colors do not depend on the size of the cell: an atom of a supercell has the
    color of the atom it is a copy of. They do not depend on the other
    structures of the batch.

    Args:
        structures: Batched Structures, used for atomic numbers and batch indices.
        graph: Graph built from structures.

    Returns:
        128-bit hash color per atom, shape (n_atoms, 4), dtype uint32.

    Examples:
        >>> from pycrystalgen.data.batching import Structures
        >>> from pycrystalgen.graph.atomic_radius import atomic_radius
        >>> structures = Structures.zincblende()
        >>> graph = atomic_radius(structures)
        >>> colors = refined_coloring(structures, graph)
        >>> colors.shape
        torch.Size([8, 4])
        >>> colors.unique(dim=0).shape[0]  # one color for Zn, one for S
        2
    """
    colors = initial_coloring(structures, graph)
    mask = torch.ones_like(structures.num_atoms, dtype=torch.bool)

    while True:
        mask_edge = mask[graph.edge_batch]
        mask_nodes = mask[structures.batch]

        # indexing is done on 64 bits: it is not implemented for uint32 on CUDA
        neighbors = scatter(
            colors.to(torch.int64)[graph.edge_index[1, mask_edge]],
            graph.edge_index[0, mask_edge],
            colors.shape[0],
            reduce="hash",
        ).to(torch.int64)

        # hash of the pair (color of the atom, colors of its neighbours): two
        # atoms with different colors never get the same one, a round only
        # splits the classes and the refinement stops
        next_color = mix32(
            (
                (neighbors * OWN_COLOR_MULTIPLIER + mix32(colors).to(torch.int64))
                & 0xFFFFFFFF
            ).to(torch.uint32)
        )
        next_color = torch.where(
            mask_nodes.unsqueeze(-1),
            next_color.to(torch.int64),
            colors.to(torch.int64),
        ).to(torch.uint32)

        equivalent = equivalent_partition(
            _xor4(colors), _xor4(next_color), structures.batch
        )

        if equivalent.all():
            break

        # a round which does not change the partition of a structure is dropped,
        # as the last one: its colors do not depend on the rest of the batch
        colors = torch.where(
            equivalent[structures.batch].unsqueeze(-1),
            colors.to(torch.int64),
            next_color.to(torch.int64),
        ).to(torch.uint32)
        mask = ~equivalent

    return colors


def color_refinement(
    structures: Structures,
    graph: Graph,
) -> UInt32[torch.Tensor, "n_structures 4"]:
    """Hash each structure from the colors of its atoms after color refinement.

    The hash does not depend on the order of the atoms. It depends on their
    number: a supercell has another hash than its cell, compare the colors of
    refined_coloring instead.

    Args:
        structures: Batched Structures, used for atomic numbers and batch indices.
        graph: Graph built from structures.

    Returns:
        128-bit hash per structure, shape (n_structures, 4), dtype uint32.

    Examples:
        >>> from pycrystalgen.data.batching import Structures
        >>> from pycrystalgen.graph.atomic_radius import atomic_radius
        >>> structures = Structures.zincblende()
        >>> graph = atomic_radius(structures)
        >>> h = color_refinement(structures, graph)
        >>> h.shape
        torch.Size([1, 4])
    """
    colors = refined_coloring(structures, graph)

    return scatter(
        colors, structures.batch, structures.num_atoms.shape[0], reduce="hash"
    )
