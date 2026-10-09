import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from torchms.data.batching import Structures
from torchms.data.constants import get_element
from torchms.graph.geometry import cartesian, image_order, periodic_pairs
from torchms.graph.graph import Graph
from torchms.graph.line_graph import reverse_index
from torchms.graph.sparse import count_by


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
    struct_counts = count_by(sb, n_struct)
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

        >>> from torchms.data.batching import Structures
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
    n_struct = len(structures.num_atoms)
    n_atoms = len(structures.z)

    radii = _ATOMIC_RADII.to(structures.device())[structures.z]  # (n_atoms,)

    # the images of each pair which can be within the sum of its two radii,
    # exact for coordinates inside and outside the cell
    source, target, shift, pair_batch = periodic_pairs(
        structures.x,
        structures.lattice,
        structures.ptr,
        scaling * radii,
        structures.pbc,
    )

    fractional = (structures.x[target] + shift.to(structures.x.dtype)) - (
        structures.x[source]
    )
    dists = cartesian(fractional, structures.lattice, pair_batch).norm(dim=-1)

    mask = (source != target) & (dists <= scaling * (radii[source] + radii[target]))

    # the edges kept, sorted for each atom by image then by neighbour. Each
    # mask is read once: on a GPU each reading makes the CPU wait for it.
    kept = mask.nonzero().squeeze(1)
    kept = kept[image_order(source[kept], shift[kept])]
    source, target, shift, dists = source[kept], target[kept], shift[kept], dists[kept]

    if ensure_gap > 0.0:
        kept = _first_shell_mask(dists, structures.batch[source], n_struct, ensure_gap)
        kept = kept.nonzero().squeeze(1)
        source, target, shift = source[kept], target[kept], shift[kept]

    # Keep only bidirectional edges so the sparse matrix is symmetric.
    # An edge (i, j, s) is kept iff (j, i, -s) also exists in the edge list.
    edge_index = torch.stack([source, target])
    kept = (reverse_index(edge_index, shift, n_atoms) >= 0).nonzero().squeeze(1)

    return Graph.from_edges(
        structures,
        structures.num_atoms,
        edge_index[:, kept],
        shift[kept],
        edge_batch=structures.batch[source[kept]],
    )
