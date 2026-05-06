import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pycrystalgen.data.batching import Structures


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
) -> tuple[Float[torch.Tensor, "tiles_atoms 3"], Int[torch.Tensor, "tiles_atoms 3"]]:
    """Generate a supercell with all borders at distance >= radius from the central cell.

    Args:
        structures: Single-structure Structures object. x must be fractional coordinates.
        radius: Minimum border distance in Ångströms.

    Returns:
        positions: Cartesian positions of all tiled atoms, shape (n_tiles * n_atoms, 3).
        offsets: Integer image shifts per atom, shape (n_tiles * n_atoms, 3).
    """

    device = structures.device()
    L = structures.lattice[0]           # (3, 3)
    n_atoms = int(structures.num_atoms[0])
    cart = structures.x @ L             # (n_atoms, 3) fractional → Cartesian

    n = torch.ceil(radius / lattice_heights(structures.lattice)[0].clamp_min(1e-8)).long()  # (3,)

    shifts = torch.cartesian_prod(
        torch.arange(-n[0], n[0] + 1, device=device),
        torch.arange(-n[1], n[1] + 1, device=device),
        torch.arange(-n[2], n[2] + 1, device=device),
    )  # (n_tiles, 3)

    positions = (cart[None] + (shifts.float() @ L)[:, None]).reshape(-1, 3)
    offsets = shifts.repeat_interleave(n_atoms, dim=0)

    return positions, offsets


def cutoff(structures: Structures) -> torch.LongTensor:
    pass
