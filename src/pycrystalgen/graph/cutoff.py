import torch
from jaxtyping import Float

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.geometry import centered_supercell, edges_distances
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.sparse import batched_cartesian_product_2


def cutoff(structures: Structures, radius: float) -> Graph:
    """Build a radius-cutoff graph for a batch of crystal structures.

    Args:
        structures: Batched Structures object. x must be fractional coordinates.
        radius: Cutoff radius in Ångströms.

    Returns:
        Graph with edge_index, edge_shift, and batch bookkeeping tensors.
    """
    device = structures.device()
    n_struct = len(structures.num_atoms)

    supercell, atom_index = centered_supercell(structures, radius)

    pair_idx, pair_batch, _ = batched_cartesian_product_2(structures.ptr, supercell.ptr)
    pair_idx = pair_idx.t()

    dists = edges_distances(
        structures.x,
        supercell.x,
        structures.lattice,
        pair_batch,
        pair_idx,
    )

    print(pair_idx)
    global_i = pair_idx[0]
    global_j = atom_index[pair_idx[1]]
    print(global_j)

    mask = (global_i != global_j) & (dists <= radius)

    senders = global_i[mask]
    receivers = global_j[mask]
    edge_batch = structures.batch[senders]
    num_edges = torch.bincount(edge_batch, minlength=n_struct)

    edge_shift = supercell.x[pair_idx[1][mask]].floor().long()

    return Graph(
        structures=structures,
        edge_index=torch.stack([senders, receivers]),
        edge_shift=edge_shift,
        num_atoms=structures.num_atoms,
        num_edges=num_edges,
        _validate=False,
    )
