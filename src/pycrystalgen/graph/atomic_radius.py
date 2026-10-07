import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pycrystalgen.data.batching import Structures
from pycrystalgen.data.constants import get_element
from pycrystalgen.graph.geometry import centered_supercell, edges_distances
from pycrystalgen.graph.graph import Graph
from pycrystalgen.graph.sparse import batched_cartesian_product_2


def _build_radii_tensor() -> torch.Tensor:
    # Index 0 unused (no element with Z=0). Radii in pm → Å.
    radii = torch.zeros(119, dtype=torch.float32)
    for z in range(1, 119):
        radii[z] = get_element(z).atomic_radius / 100.0
    return radii


_ATOMIC_RADII = _build_radii_tensor()


def _first_shell_mask(
    dists: Float[torch.Tensor, "n_edges"],
    batch: Int[torch.Tensor, "n_edges"],
    n_struct: int,
    ensure_gap: float,
) -> torch.Tensor:
    """Boolean mask keeping only the first distance shell per structure.

    Sorts edges by distance within each structure, finds the first consecutive
    pair with a gap >= ensure_gap, and drops all edges at or after that gap.
    """
    if len(dists) == 0:
        return dists.new_zeros(0, dtype=torch.bool)

    device = dists.device
    n_sorted = len(dists)

    # Lexsort by (batch, dist): stable-sort by dist first, then by batch.
    sort1 = torch.argsort(dists, stable=True)
    sort2 = torch.argsort(batch[sort1], stable=True)
    sort_idx = sort1[sort2]

    sb = batch[sort_idx]  # sorted batch indices
    sd = dists[sort_idx]  # sorted distances

    # Local position of each edge within its structure (0-based).
    struct_counts = torch.bincount(sb, minlength=n_struct)
    struct_ptr = F.pad(struct_counts.cumsum(0), (1, 0), value=0)
    local_pos = torch.arange(n_sorted, device=device) - struct_ptr[sb]

    # gap_after[i]: True if the jump from edge i to i+1 (same structure) >= ensure_gap.
    same_next = torch.cat([sb[:-1] == sb[1:], sb.new_zeros(1, dtype=torch.bool)])
    d_next = torch.cat([sd[1:], sd.new_zeros(1)])
    gap_after = same_next & ((d_next - sd) >= ensure_gap)

    # First gap local position per structure. Sentinel n_sorted means no gap.
    first_gap_local = torch.full((n_struct,), n_sorted, device=device, dtype=torch.long)
    if gap_after.any():
        first_gap_local.scatter_reduce_(
            0, sb[gap_after], local_pos[gap_after], reduce="amin", include_self=True
        )

    # Keep edge if its local position is before (or at) the first gap boundary.
    keep_sorted = local_pos <= first_gap_local[sb]

    # Invert the sort to recover the original edge ordering.
    keep = torch.empty(n_sorted, dtype=torch.bool, device=device)
    keep[sort_idx] = keep_sorted
    return keep


def atomic_radius(
    structures: Structures, scaling: float = 2.0, ensure_gap: float = 0.5
) -> Graph:
    """Build a graph whose per-edge cutoff is scaling * (r_i + r_j).

    Uses empirical atomic radii from the periodic table constants. An edge
    (i, j) is kept when dist(i, j) <= scaling * (r_i + r_j).

    When ensure_gap > 0, edges after the first gap of that size in the
    per-structure distance distribution are also dropped, ensuring a clean
    shell boundary.

    Args:
        structures: Batched Structures object. x must be fractional coordinates.
        scaling: Multiplier applied to the sum of atomic radii.
        ensure_gap: Minimum gap in Å required between the last kept bond and
            the next bond. Edges at or after the first such gap per structure
            are dropped. Set to 0 to disable. Default: 0.5.

    Returns:
        Undirected Graph, where for every edge (i, j, s) the reverse (j, i, -s)
        is also present, giving a symmetric sparse matrix.

    Examples:
        Nearest-neighbour graph for zincblende ZnS with scaling=1.1.
        Zn (r=1.42 Å) + S (r=0.88 Å) → threshold 2.53 Å captures the
        2.34 Å bond while excluding the 3.83 Å same-species distance.

        >>> from pycrystalgen.data.batching import Structures
        >>> structure = Structures.zincblende()
        >>> graph = atomic_radius(structure, 1.1)
        >>> graph.edges_distances().unique().round(decimals=2)
        tensor([2.3400])
        >>> graph.num_edges
        tensor([32])

        With a wider scaling, ensure_gap drops the second shell:

        >>> graph2 = atomic_radius(structure, 2.0, ensure_gap=0.5)
        >>> graph2.edges_distances().unique().round(decimals=2)
        tensor([2.3400])
        >>> graph2.num_edges
        tensor([32])
    """
    device = structures.device()
    n_struct = len(structures.num_atoms)

    radii = _ATOMIC_RADII.to(device)[structures.z]  # (n_atoms,)
    max_radius = float(radii.max()) * scaling * 2

    supercell, atom_index = centered_supercell(structures, max_radius)

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
    r_i = radii[pair_idx[0]]
    r_j = radii[real_target]
    threshold = scaling * (r_i + r_j)

    mask = (pair_idx[0] != real_target) & (dists <= threshold)

    # image of the target: its coordinates in the supercell minus the ones in
    # the cell, also exact for coordinates outside [0, 1)
    edge_shift = (
        (supercell.x[pair_idx[1][mask]] - structures.x[real_target[mask]])
        .round()
        .long()
    )
    pair_idx[1] = real_target
    pair_idx = pair_idx[:, mask]

    filtered_dists = dists[mask]
    edge_batch = structures.batch[pair_idx[0]]

    if ensure_gap > 0.0:
        gap_mask = _first_shell_mask(filtered_dists, edge_batch, n_struct, ensure_gap)
        pair_idx = pair_idx[:, gap_mask]
        edge_shift = edge_shift[gap_mask]
        edge_batch = edge_batch[gap_mask]

    # Keep only bidirectional edges so the sparse matrix is symmetric.
    # An edge (i, j, s) is kept iff (j, i, -s) also exists in the edge list.
    n_edges = pair_idx.shape[1]
    fwd = torch.cat([pair_idx.t(), edge_shift], dim=1)
    rev = torch.cat([pair_idx[[1, 0]].t(), -edge_shift], dim=1)
    _, uid = torch.unique(torch.cat([fwd, rev], dim=0), dim=0, return_inverse=True)
    bidi_mask = torch.isin(uid[:n_edges], uid[n_edges:])
    pair_idx = pair_idx[:, bidi_mask]
    edge_shift = edge_shift[bidi_mask]
    edge_batch = edge_batch[bidi_mask]

    num_edges = torch.bincount(edge_batch, minlength=n_struct)

    return Graph(
        structures=structures,
        edge_index=pair_idx,
        edge_shift=edge_shift,
        num_atoms=structures.num_atoms,
        num_edges=num_edges,
        _validate=False,
    )
