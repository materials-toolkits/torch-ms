import torch

from torchms.data.batching import Structures
from torchms.graph.graph import Graph
from torchms.graph.sparse import batched_cartesian_product_2


def complete(structures: Structures, self_loops: bool = False) -> Graph:
    """Build the complete graph of each structure of a batch.

    An edge goes from each atom to every atom of its structure, in the same
    cell: no edge goes to a periodic image and the lattice is not read. It is
    the graph of the models which look at every pair of atoms and read the
    difference of their fractional coordinates modulo the cell, and of small
    molecules.

    Args:
        structures: Batched Structures object.
        self_loops: If True, an atom is also connected to itself.

    Returns:
        Graph with the n² edges of each structure of n atoms, n (n - 1)
        without self loops, sorted by sender then by receiver. Every
        edge_shift is zero.

    Examples:
        A molecule of water: its three atoms are connected to each other.

        >>> from torchms.data.batching import Structures
        >>> graph = complete(Structures.water())
        >>> graph.edge_index
        tensor([[0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1]])
        >>> complete(Structures.water(), self_loops=True).num_edges
        tensor([9])

        A batch: the atoms of two structures are not connected.

        >>> batch = Structures.collate([Structures.water(), Structures.diamond()])
        >>> complete(batch, self_loops=True).num_edges
        tensor([ 9, 64])
    """
    device = structures.device()

    pairs, edge_batch, _ = batched_cartesian_product_2(structures.ptr, structures.ptr)

    if not self_loops:
        other = pairs[:, 0] != pairs[:, 1]
        pairs, edge_batch = pairs[other], edge_batch[other]

    return Graph.from_edges(
        structures,
        structures.num_atoms,
        pairs.t().contiguous(),
        torch.zeros((len(pairs), 3), dtype=torch.int64, device=device),
        edge_batch=edge_batch,
    )
