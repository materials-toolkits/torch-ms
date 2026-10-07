import torch
from jaxtyping import UInt32

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.sparse import equivalent_partition, mix32, scatter


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

    Iteratively replaces each node's color with a hash of its neighbors' colors,
    restricted to structures whose partition is still changing. Stops when no
    structure's partition changes.

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

        next_color = scatter(
            colors[graph.edge_index[1, mask_edge]],
            graph.edge_index[0, mask_edge],
            colors.shape[0],
            reduce="hash",
        ).to(torch.uint32)
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
