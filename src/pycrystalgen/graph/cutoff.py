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

    Example:
        Compute nearest-neighbour distances in zincblende ZnS (a = 5.41 Å).
        The Zn–S bond length is a√3/4 ≈ 2.34 Å; each atom has 4 neighbours.

        >>> from pycrystalgen.data.batching import Structures
        >>> structure = Structures.zincblende()
        >>> graph = cutoff(structure, 3.0)
        >>> graph.edges_distances().unique().round(decimals=2)
        tensor([2.3400])
        >>> graph.num_edges
        tensor([32])
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

    real_target = atom_index[pair_idx[1]]
    mask = (pair_idx[0] != real_target) & (dists <= radius)

    edge_shift = supercell.x[pair_idx[1][mask]].floor().long()

    pair_idx[1] = real_target

    pair_idx = pair_idx[:, mask]
    edge_batch = structures.batch[pair_idx[0]]
    num_edges = torch.bincount(edge_batch, minlength=n_struct)

    return Graph(
        structures=structures,
        edge_index=pair_idx,
        edge_shift=edge_shift,
        num_atoms=structures.num_atoms,
        num_edges=num_edges,
        _validate=False,
    )
