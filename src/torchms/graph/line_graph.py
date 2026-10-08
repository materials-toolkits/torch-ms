"""Higher order of a graph: pairs and chains of edges, for angles and dihedrals.

The line graph of a graph has one node per edge, and one edge per pair of edges
sharing an atom. Triplets of atoms (angles) are the edges of the line graph of
the atoms, quadruplets (dihedrals) the edges of the line graph of this line
graph.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Int

from torchms.graph.graph import Graph
from torchms.graph.sparse import batched_join


def reverse_edges(graph: Graph) -> Int[torch.Tensor, "n_edges"]:
    """Index of the reverse of each edge.

    The reverse of the edge from i to the image s of j goes from j to the image
    -s of i. Every edge of a cutoff graph has a reverse, not the ones of a graph
    of nearest neighbours.

    Args:
        graph: Graph without duplicated edges.

    Returns:
        Index of the reverse edge, -1 when it is not in the graph, shape (n_edges,).

    Examples:
        >>> from torchms.data.batching import Structures
        >>> from torchms.graph.cutoff import cutoff
        >>> graph = cutoff(Structures.diamond(), 2.0)
        >>> reverse = reverse_edges(graph)
        >>> bool((graph.edge_index[:, reverse] == graph.edge_index.flip(0)).all())
        True
        >>> bool((graph.edge_shift[reverse] == -graph.edge_shift).all())
        True
    """
    device = graph.device()
    n_edges = graph.edge_index.shape[1]
    n_atoms = int(graph.num_atoms.sum())
    sender, receiver = graph.edge_index

    if n_edges == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    # one integer per edge: its atoms and its image, as the digits of a number
    largest = int(graph.edge_shift.abs().max())
    base = 2 * largest + 1

    if n_atoms * n_atoms * base**3 >= 2**63:
        return _reverse_edges_by_rows(graph)

    def key(first: torch.Tensor, second: torch.Tensor, shift: torch.Tensor):
        image = shift + largest
        atoms = first * n_atoms + second

        return ((atoms * base + image[:, 0]) * base + image[:, 1]) * base + image[:, 2]

    edge_key = key(sender, receiver, graph.edge_shift)
    reverse_key = key(receiver, sender, -graph.edge_shift)

    # position of the key of each reverse among the sorted keys of the edges
    order = torch.argsort(edge_key, stable=True)
    sorted_key = edge_key[order]
    position = torch.searchsorted(sorted_key, reverse_key).clamp(max=n_edges - 1)

    return torch.where(sorted_key[position] == reverse_key, order[position], -1)


def _reverse_edges_by_rows(graph: Graph) -> Int[torch.Tensor, "n_edges"]:
    """Reverse of each edge by comparing rows, see reverse_edges.

    Slower, for graphs too large for one integer per edge.
    """
    device = graph.device()
    n_edges = graph.edge_index.shape[1]
    sender, receiver = graph.edge_index

    edges = torch.cat([sender[:, None], receiver[:, None], graph.edge_shift], dim=1)
    reverse = torch.cat([receiver[:, None], sender[:, None], -graph.edge_shift], dim=1)

    # the same identifier for an edge and for the reverse of its reverse
    unique, identifier = torch.unique(
        torch.cat([edges, reverse], dim=0), dim=0, return_inverse=True
    )
    edge_id, reverse_id = identifier[:n_edges], identifier[n_edges:]

    # identifier → edge, -1 for the reverses which are not edges
    edge_of = torch.full((len(unique),), -1, dtype=torch.int64, device=device)
    edge_of[edge_id] = torch.arange(n_edges, device=device)

    return edge_of[reverse_id]


def line_graph(
    edge_index: Int[torch.Tensor, "2 n_edges"],
    num_nodes: int,
    reverse: Int[torch.Tensor, "n_edges"] | None = None,
) -> Int[torch.Tensor, "2 n_pairs"]:
    """Pairs of edges head to tail: u→v followed by v→w.

    The result is the edge index of the line graph, whose nodes are the edges.
    It is itself an edge index: its line graph gives the chains of three edges.

    Args:
        edge_index: Sender and receiver of each edge, shape (2, n_edges).
        num_nodes: Number of nodes of the graph.
        reverse: Index of the reverse of each edge, see reverse_edges. If given,
            an edge is not followed by its reverse: the chain does not go back
            to the atom it comes from.

    Returns:
        Indices of the first and second edge of each pair, sorted by the node
        they share, shape (2, n_pairs).

    Examples:
        A chain 0 → 1 → 2 with its reverse edges.

        >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        >>> line_graph(edge_index, 3)
        tensor([[1, 0, 0, 3, 3, 2],
                [0, 1, 2, 1, 2, 3]])
        >>> reverse = torch.tensor([1, 0, 3, 2])
        >>> line_graph(edge_index, 3, reverse)
        tensor([[0, 3],
                [2, 1]])
    """
    pairs, _, _ = batched_join(edge_index[1], edge_index[0], num_nodes)

    if reverse is not None:
        pairs = pairs[reverse[pairs[:, 0]] != pairs[:, 1]]

    return pairs.t()


def _selected(
    graph: Graph, edge_mask: Bool[torch.Tensor, "n_edges"] | None
) -> Int[torch.Tensor, "n_selected"]:
    """Indices of the edges of the mask, all the edges without mask."""
    if edge_mask is None:
        return torch.arange(graph.edge_index.shape[1], device=graph.device())

    return edge_mask.nonzero().squeeze(1)


def _batch_of(
    graph: Graph, edges: Int[torch.Tensor, "n"]
) -> tuple[Int[torch.Tensor, "n"], Int[torch.Tensor, "n_struct+1"]]:
    """Structure of each tuple of edges from its first edge, and CSR pointers."""
    batch = graph.edge_batch[edges]
    count = torch.bincount(batch, minlength=len(graph.num_atoms))

    return batch, F.pad(torch.cumsum(count, dim=0), (1, 0), value=0)


def edge_pairs(
    graph: Graph,
    edge_mask: Bool[torch.Tensor, "n_edges"] | None = None,
    backtrack: bool = False,
) -> tuple[
    Int[torch.Tensor, "2 n_pairs"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Pairs of edges head to tail u→v, v→w: the edges of the line graph.

    Args:
        graph: Graph of the atoms.
        edge_mask: Edges which can be part of a pair, shape (n_edges,). All of
            them by default.
        backtrack: If False, an edge is not followed by its reverse: w is
            another atom than u, or another image of u.

    Returns:
        Indices of the first and second edge of each pair, shape (2, n_pairs),
        sorted by the atom they share, structure index of each pair, shape
        (n_pairs,), and CSR pointers of the pairs of each structure, shape
        (n_struct+1,).

    Examples:
        Each of the 32 edges of diamond is followed by the 3 other edges of the
        atom it arrives at.

        >>> from torchms.data.batching import Structures
        >>> from torchms.graph.cutoff import cutoff
        >>> graph = cutoff(Structures.diamond(), 2.0)
        >>> pair_index, pair_batch, pair_ptr = edge_pairs(graph)
        >>> pair_ptr
        tensor([ 0, 96])
        >>> first, second = pair_index
        >>> bool((graph.edge_index[1, first] == graph.edge_index[0, second]).all())
        True
        >>> edge_pairs(graph, backtrack=True)[2]
        tensor([  0, 128])
    """
    device = graph.device()
    edges = _selected(graph, edge_mask)
    n_edges = graph.edge_index.shape[1]

    reverse = None
    if not backtrack:
        # reverse of each selected edge among the selected edges
        position = torch.full((n_edges + 1,), -1, dtype=torch.int64, device=device)
        position[edges] = torch.arange(len(edges), device=device)
        reverse = position[reverse_edges(graph)[edges]]

    pairs = line_graph(graph.edge_index[:, edges], int(graph.num_atoms.sum()), reverse)

    pair_index = edges[pairs]
    pair_batch, pair_ptr = _batch_of(graph, pair_index[0])

    return pair_index, pair_batch, pair_ptr


def triplets(
    graph: Graph, edge_mask: Bool[torch.Tensor, "n_edges"] | None = None
) -> tuple[
    Int[torch.Tensor, "2 n_triplets"],
    Int[torch.Tensor, "n_triplets"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Pairs of different edges starting from the same atom.

    A triplet (i→j, i→k) is the angle at the atom i between two of its
    neighbours. Both orders of the two edges are listed: an atom with d edges
    has d (d - 1) triplets. Two images of the same atom are different
    neighbours.

    Args:
        graph: Graph of the atoms.
        edge_mask: Edges which can be part of a triplet, shape (n_edges,). All
            of them by default. Used for a cutoff of the angles shorter than
            the one of the edges.

    Returns:
        Indices of the two edges of each triplet, shape (2, n_triplets), sorted
        by their atom, structure index of each triplet, shape (n_triplets,),
        and CSR pointers of the triplets of each structure, shape (n_struct+1,).

    Examples:
        Each atom of diamond has 4 neighbours, 4 × 3 triplets.

        >>> from torchms.data.batching import Structures
        >>> from torchms.graph.cutoff import cutoff
        >>> from torchms.graph.geometry import triplet_cosines
        >>> graph = cutoff(Structures.diamond(), 2.0)
        >>> triplet_index, triplet_batch, triplet_ptr = triplets(graph)
        >>> triplet_ptr
        tensor([ 0, 96])
        >>> cosines = triplet_cosines(graph.edges_vectors(), triplet_index)
        >>> cosines.unique().round(decimals=4)  # tetrahedral angle, 109.47°
        tensor([-0.3333])
    """
    edges = _selected(graph, edge_mask)
    sender = graph.edge_index[0, edges]

    pairs, _, _ = batched_join(sender, sender, int(graph.num_atoms.sum()))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    triplet_index = edges[pairs].t()
    triplet_batch, triplet_ptr = _batch_of(graph, triplet_index[0])

    return triplet_index, triplet_batch, triplet_ptr


def quadruplets(
    graph: Graph, edge_mask: Bool[torch.Tensor, "n_edges"] | None = None
) -> tuple[
    Int[torch.Tensor, "3 n_quadruplets"],
    Int[torch.Tensor, "n_quadruplets"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Chains of three edges u→v, v→w, w→x which never go back along an edge.

    A quadruplet is the dihedral angle around the bond v-w. In a graph where
    every edge has a reverse, a chain and the same chain from x to u are both
    listed.

    Args:
        graph: Graph of the atoms.
        edge_mask: Edges which can be part of a chain, shape (n_edges,). All of
            them by default. The number of chains grows with the cube of the
            number of neighbours: use a short cutoff.

    Returns:
        Indices of the three edges of each chain, shape (3, n_quadruplets),
        structure index of each chain, shape (n_quadruplets,), and CSR
        pointers of the chains of each structure, shape (n_struct+1,).

    Examples:
        In diamond, each of the 4 neighbours of an atom has 3 other neighbours,
        which have 3 other neighbours: 8 × 4 × 3 × 3 chains.

        >>> from torchms.data.batching import Structures
        >>> from torchms.graph.cutoff import cutoff
        >>> graph = cutoff(Structures.diamond(), 2.0)
        >>> quadruplet_index, quadruplet_batch, quadruplet_ptr = quadruplets(graph)
        >>> quadruplet_ptr
        tensor([  0, 288])
    """
    device = graph.device()
    edges = _selected(graph, edge_mask)
    n_edges = graph.edge_index.shape[1]

    # reverse of each selected edge among the selected edges
    position = torch.full((n_edges + 1,), -1, dtype=torch.int64, device=device)
    position[edges] = torch.arange(len(edges), device=device)
    reverse = position[reverse_edges(graph)[edges]]

    # chains of two edges, then chains of two chains sharing their middle edge
    pairs = line_graph(graph.edge_index[:, edges], int(graph.num_atoms.sum()), reverse)
    chains = line_graph(pairs, len(edges))

    quadruplet_index = edges[
        torch.stack([pairs[0, chains[0]], pairs[1, chains[0]], pairs[1, chains[1]]])
    ]
    quadruplet_batch, quadruplet_ptr = _batch_of(graph, quadruplet_index[0])

    return quadruplet_index, quadruplet_batch, quadruplet_ptr
