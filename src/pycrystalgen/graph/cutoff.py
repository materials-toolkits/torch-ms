import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.scatter import batched_cartesian_product_3


def lattice_heights(
    lattice: Float[torch.Tensor, "n 3 3"],
) -> Float[torch.Tensor, "n 3"]:
    """Compute the perpendicular height of unit cells along each lattice direction.

    h_i = |a_i · n̂_i| where n̂_i = normalize(a_j × a_k) is the unit normal to the opposite face.

    Args:
        lattice: Lattice vectors with shape (n, 3, 3). Rows are lattice vectors.

    Returns:
        Heights in Ångströms with shape (n, 3).
    """

    normal = torch.cross(lattice[:, [1, 0, 0]], lattice[:, [2, 2, 1]], dim=-1)
    unit = F.normalize(normal, dim=-1)
    heights = (lattice * unit).sum(-1).abs()

    return heights


def centered_supercell(
    structures: Structures,
    radius: float,
) -> tuple[
    Float[torch.Tensor, "total 3"],
    Int[torch.Tensor, "total 3"],
    Int[torch.Tensor, "total"],
]:
    """Generate a supercell with all borders at distance >= radius from the central cell.

    Args:
        structures: Batched Structures object. x must be fractional coordinates.
        radius: Minimum border distance in Ångströms.

    Returns:
        positions: Cartesian positions of all tiled atoms, shape (total, 3).
        offsets: Integer image shifts per atom, shape (total, 3).
        atom_index: Index into structures.x for each output atom, shape (total,).
    """
    device = structures.device()

    # repetitions per axis per structure: (n_struct, 3)
    n = torch.ceil(radius / lattice_heights(structures.lattice).clamp_min(1e-8)).long()

    # all shift combinations, 0-based, then centered
    sizes = 2 * n + 1                                                    # (n_struct, 3)
    shift_idx, tile_batch, tile_ptr = batched_cartesian_product_3(
        sizes[:, 0], sizes[:, 1], sizes[:, 2]
    )                                                                    # (total_tiles, 3)
    shifts = shift_idx - n[tile_batch]                                   # (total_tiles, 3)

    # fractional → Cartesian: (n_atoms, 3)
    cart = torch.bmm(
        structures.x.unsqueeze(1),
        structures.lattice[structures.batch],
    ).squeeze(1)

    # Cartesian displacement per tile: (total_tiles, 3)
    shift_cart = torch.bmm(
        shifts.float().unsqueeze(1),
        structures.lattice[tile_batch],
    ).squeeze(1)

    # expand: tiles (outer) × atoms (inner)
    total_tiles = int(tile_ptr[-1])
    num_atoms_per_tile = structures.num_atoms[tile_batch]                # (total_tiles,)
    output_ptr = F.pad(torch.cumsum(num_atoms_per_tile, dim=0), (1, 0), value=0)
    total_output = int(output_ptr[-1])

    atom_tile  = torch.arange(total_tiles, device=device).repeat_interleave(num_atoms_per_tile)
    local_atom = torch.arange(total_output, device=device) - output_ptr[atom_tile]
    atom_index = local_atom + structures.ptr[tile_batch[atom_tile]]      # global atom index

    positions = cart[atom_index] + shift_cart[atom_tile]
    offsets   = shifts[atom_tile]

    return positions, offsets, atom_index


def cutoff(structures: Structures) -> torch.LongTensor:
    pass
