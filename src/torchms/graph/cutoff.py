import torch
import torch.nn.functional as F
from jaxtyping import Float

from torchms.data.batching import Structures
from torchms.graph.geometry import cartesian, image_order, periodic_pairs
from torchms.graph.graph import Graph
from torchms.graph.sparse import count_by


def cutoff(structures: Structures, radius: float, self_images: bool = False) -> Graph:
    """Build a radius-cutoff graph for a batch of structures.

    A structure is connected to its periodic images only along the lattice
    vectors along which it is periodic: the edges of a molecule are between
    its atoms, and crystals and molecules can be in the same batch.

    Args:
        structures: Batched Structures object. x must be fractional coordinates,
            inside or outside the cell.
        radius: Cutoff radius in Ångströms.
        self_images: If True, an atom is connected to its own periodic images
            closer than the radius, which exist when the cell is smaller than
            the radius. If False, an atom is never connected to itself.

    Returns:
        Graph with edge_index, edge_shift, and batch bookkeeping tensors.

    Examples:
        Compute nearest-neighbour distances in zincblende ZnS (a = 5.41 Å).
        The Zn–S bond length is a√3/4 ≈ 2.34 Å; each atom has 4 neighbours.

        >>> from torchms.data.batching import Structures
        >>> structure = Structures.zincblende()
        >>> graph = cutoff(structure, 3.0)
        >>> graph.edges_distances().unique().round(decimals=2)
        tensor([2.3400])
        >>> graph.num_edges
        tensor([32])

        A simple cubic cell (a = 3 Å) with one atom: its 6 nearest neighbours
        are its own periodic images.

        >>> structure = Structures(
        ...     lattice=torch.eye(3).unsqueeze(0) * 3.0,
        ...     x=torch.zeros(1, 3),
        ...     z=torch.tensor([29]),
        ... )
        >>> cutoff(structure, 3.5).num_edges
        tensor([0])
        >>> cutoff(structure, 3.5, self_images=True).num_edges
        tensor([6])

        A molecule of water: its two O-H bonds, in both directions. The two
        hydrogens are 1.51 Å apart.

        >>> graph = cutoff(Structures.water(), 1.2)
        >>> graph.edge_index
        tensor([[0, 0, 1, 2],
                [1, 2, 0, 0]])
        >>> graph.edge_shift.abs().max().item()
        0
    """
    # neighbours are searched from atoms wrapped in the cell. Edges are then
    # expressed for the given coordinates. Atoms are not wrapped along a
    # lattice vector which is not periodic.
    cell = torch.where(structures.pbc[structures.batch], structures.x.floor(), 0.0)
    wrapped = structures.x - cell

    source, target, shift, pair_batch = periodic_pairs(
        wrapped, structures.lattice, structures.ptr, radius, structures.pbc
    )

    fractional = (wrapped[target] + shift.to(wrapped.dtype)) - wrapped[source]
    dists = cartesian(fractional, structures.lattice, pair_batch).norm(dim=-1)

    if self_images:
        # only the atom itself, at a distance of zero, is not a neighbour
        mask = ((source != target) | (dists > 1e-6)) & (dists <= radius)
    else:
        mask = (source != target) & (dists <= radius)

    # the edges kept, sorted for each atom by image then by neighbour. The
    # mask is read once: on a GPU each reading makes the CPU wait for it.
    kept = mask.nonzero().squeeze(1)
    source, target, shift = source[kept], target[kept], shift[kept]
    order = image_order(source, shift)
    source, target, shift = source[order], target[order], shift[order]

    # image of the target, moved from the wrapped to the given coordinates
    edge_shift = shift + (cell[source] - cell[target]).long()

    return Graph.from_edges(
        structures,
        structures.num_atoms,
        torch.stack([source, target]),
        edge_shift,
        edge_batch=structures.batch[source],
    )


def nearest_neighbors(graph: Graph, k: int) -> Graph:
    """Keep the k shortest edges starting from each atom.

    Args:
        graph: Graph with a reference to its structures.
        k: Maximum number of edges per source atom. Among edges of the same
            length, the first ones of the graph are kept.

    Returns:
        Graph with at most k edges per source atom, sorted by source atom then
        by length.

    Examples:
        The 4 nearest neighbours of each of the 8 atoms of diamond:

        >>> from torchms.data.batching import Structures
        >>> graph = nearest_neighbors(cutoff(Structures.diamond(), 4.0), 4)
        >>> graph.num_edges
        tensor([32])
        >>> graph.edges_distances().unique().round(decimals=2)
        tensor([1.5400])
    """
    device = graph.device()
    n_atoms = int(graph.num_atoms.sum())
    source = graph.edge_index[0]

    # sort by (source, distance): stable sort by distance, then by source
    by_distance = torch.argsort(graph.edges_distances(), stable=True)
    order = by_distance[torch.argsort(source[by_distance], stable=True)]

    # rank of each edge among the edges of its source atom
    degree = count_by(source, n_atoms)
    source_ptr = F.pad(torch.cumsum(degree, dim=0), (1, 0), value=0)
    rank = torch.arange(len(order), device=device) - source_ptr[source[order]]

    return graph.select_edges(order[rank < k])


def near_images(graph: Graph, max_shift: int = 1) -> Graph:
    """Keep the edges to the nearest periodic images.

    Args:
        graph: Graph of the atoms.
        max_shift: Largest number of cells, along each lattice vector, between
            the cell of the sender and the image of the receiver.

    Returns:
        Graph with the edges whose shift is at most max_shift in absolute
        value along each lattice vector, in the same order.

    Examples:
        One atom in a cubic cell of 3 Å: its images up to 7 Å, then the ones
        of the 26 cells around its own.

        >>> from torchms.data.batching import Structures
        >>> lattice = 3.0 * torch.eye(3).unsqueeze(0)
        >>> cube = Structures(lattice=lattice, x=[[0.0, 0.0, 0.0]], z=[29])
        >>> graph = cutoff(cube, 7.0, self_images=True)
        >>> graph.num_edges, near_images(graph).num_edges
        (tensor([56]), tensor([26]))
    """
    near = (graph.edge_shift.abs() <= max_shift).all(dim=1)

    return graph.select_edges(near.nonzero().squeeze(1))


def mirror_edges(graph: Graph) -> Graph:
    """Make a graph symmetric from the edges going to an atom of lower index.

    An edge is kept if its receiver has a lower index than its sender, or, for
    an edge between an atom and one of its images, if the first non-zero
    component of its shift is negative. The reverse of each edge kept is
    added. After nearest_neighbors, where an edge can be kept without its
    reverse, such an edge gets its reverse if it is kept and is removed
    otherwise. This is how GemNet makes its graphs symmetric.

    Args:
        graph: Graph of the atoms.

    Returns:
        Graph where every edge has its reverse, ordered by structure, without
        geometry.

    Examples:
        Edges 1 → 0, kept and mirrored, and 1 → 2, removed.

        >>> from torchms.data.batching import Structures
        >>> structures = Structures.water()
        >>> graph = Graph.from_edges(
        ...     structures,
        ...     structures.num_atoms,
        ...     torch.tensor([[1, 1], [0, 2]]),
        ...     torch.zeros(2, 3, dtype=torch.int64),
        ...     edge_batch=torch.zeros(2, dtype=torch.int64),
        ... )
        >>> mirror_edges(graph).edge_index
        tensor([[1, 0],
                [0, 1]])
    """
    sender, receiver = graph.edge_index
    shift = graph.edge_shift

    # first non-zero component of the shift negative
    earlier = (
        (shift[:, 0] < 0)
        | ((shift[:, 0] == 0) & (shift[:, 1] < 0))
        | ((shift[:, 0] == 0) & (shift[:, 1] == 0) & (shift[:, 2] < 0))
    )
    kept = ((receiver < sender) | ((receiver == sender) & earlier)).nonzero().squeeze(1)

    edge_index = graph.edge_index[:, kept]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_shift = torch.cat([shift[kept], -shift[kept]])
    edge_batch = graph.edge_batch[kept].repeat(2)

    order = torch.argsort(edge_batch, stable=True)

    return Graph.from_edges(
        graph.structures,
        graph.num_atoms,
        edge_index[:, order],
        edge_shift[order],
        edge_batch=edge_batch[order],
    )
