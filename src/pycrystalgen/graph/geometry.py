import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.sparse import batched_cartesian_product_3


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
) -> tuple[Structures, Int[torch.Tensor, "total"]]:
    """Generate a supercell with all borders at distance >= radius from the central cell.

    Args:
        structures: Batched Structures object. x must be fractional coordinates.
        radius: Minimum border distance in Ångströms.

    Returns:
        The supercell, a Structures whose x are fractional coords of the tiled
        atoms (may be outside [0, 1) for periodic images) and whose num_atoms
        gives the count of tiled atoms per structure, and the index into
        structures.x of each supercell atom, shape (total,).
    """
    device = structures.device()

    # repetitions per axis per structure: (n_struct, 3)
    n = torch.ceil(radius / lattice_heights(structures.lattice).clamp_min(1e-8)).long()

    # all shift combinations, 0-based, then centered
    sizes = 2 * n + 1  # (n_struct, 3)
    shift_ptr_x = F.pad(torch.cumsum(sizes[:, 0], dim=0), (1, 0), value=0)
    shift_ptr_y = F.pad(torch.cumsum(sizes[:, 1], dim=0), (1, 0), value=0)
    shift_ptr_z = F.pad(torch.cumsum(sizes[:, 2], dim=0), (1, 0), value=0)
    shift_idx, tile_batch, tile_ptr = batched_cartesian_product_3(
        shift_ptr_x,
        shift_ptr_y,
        shift_ptr_z,
    )  # (total_tiles, 3)
    # subtract per-structure start and half-width to center around (0,0,0)
    shifts = (
        shift_idx
        - torch.stack(
            [shift_ptr_x[tile_batch], shift_ptr_y[tile_batch], shift_ptr_z[tile_batch]],
            dim=1,
        )
        - n[tile_batch]
    )  # (total_tiles, 3)

    # expand: tiles (outer) × atoms (inner)
    total_tiles = int(tile_ptr[-1])
    num_atoms_per_tile = structures.num_atoms[tile_batch]  # (total_tiles,)
    output_ptr = F.pad(torch.cumsum(num_atoms_per_tile, dim=0), (1, 0), value=0)
    total_output = int(output_ptr[-1])

    atom_tile = torch.arange(total_tiles, device=device).repeat_interleave(
        num_atoms_per_tile
    )
    local_atom = torch.arange(total_output, device=device) - output_ptr[atom_tile]
    atom_index = local_atom + structures.ptr[tile_batch[atom_tile]]  # global atom index

    # supercell fractional coords: original frac + integer shift (may be outside [0, 1))
    supercell_x = structures.x[atom_index] + shifts[atom_tile].float()

    # per-structure supercell atom counts
    num_per_struct = (tile_ptr[1:] - tile_ptr[:-1]) * structures.num_atoms

    supercell = Structures(
        lattice=structures.lattice,
        x=supercell_x,
        z=structures.z[atom_index],
        num_atoms=num_per_struct,
    )
    return supercell, atom_index


def edges_distances(
    source_x: Float[torch.Tensor, "n_atoms 3"],
    target_x: Float[torch.Tensor, "n_atoms_2 3"],
    lattice: Float[torch.Tensor, "n_structures 3 3"],
    edges_batch: Int[torch.Tensor, "n_edges"],
    edges_index: Int[torch.Tensor, "2 n_edges"],
) -> Float[torch.Tensor, "n_edges"]:
    edges_frac = target_x[edges_index[1]] - source_x[edges_index[0]]
    # rows of the lattice are the lattice vectors: cartesian = fractional @ lattice
    return torch.bmm(edges_frac.unsqueeze(1), lattice[edges_batch]).norm(dim=(-2, -1))
