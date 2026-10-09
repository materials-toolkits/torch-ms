import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from torchms.data.batching import Structures
from torchms.graph.sparse import (
    batched_cartesian_product_2,
    batched_cartesian_product_3,
)


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
    # b × c, c × a, a × b: the normals of the faces opposite to a, b and c
    normal = torch.cross(lattice.roll(-1, dims=1), lattice.roll(-2, dims=1), dim=-1)
    unit = F.normalize(normal, dim=-1)
    heights = (lattice * unit).sum(-1).abs()

    return heights


def centered_supercell(
    structures: Structures,
    radius: float,
) -> tuple[Structures, Int[torch.Tensor, "total"]]:
    """Generate a supercell with all borders at distance >= radius from the central cell.

    A structure is not repeated along a lattice vector along which it is not
    periodic.

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
    n = torch.where(structures.pbc, n, 0)

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
        pbc=structures.pbc,
    )
    return supercell, atom_index


def cartesian(
    fractional: Float[torch.Tensor, "n 3"],
    lattice: Float[torch.Tensor, "n_struct 3 3"],
    batch: Int[torch.Tensor, "n"],
) -> Float[torch.Tensor, "n 3"]:
    """Cartesian vectors from fractional ones, each in the lattice of its structure.

    The rows of a lattice are its vectors: cartesian = fractional @ lattice. It
    is written with products and a sum. A batched product of matrices this
    small is split on a GPU into many kernels, which took two thirds of the
    time of a radius graph on large batches.

    Args:
        fractional: Vectors in fractional coordinates, shape (n, 3).
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).
        batch: Structure index of each vector, shape (n,).

    Returns:
        Vectors in Ångströms, shape (n, 3).

    Examples:
        >>> lattice = torch.tensor([[[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]])
        >>> fractional = torch.tensor([[1.0, 0.0, 0.0], [0.5, 1.0, 0.5]])
        >>> cartesian(fractional, lattice, torch.tensor([0, 0]))
        tensor([[2., 0., 0.],
                [2., 3., 2.]])
    """
    return (fractional.unsqueeze(-1) * lattice[batch]).sum(dim=1)


def periodic_pairs(
    x: Float[torch.Tensor, "n_atoms 3"],
    lattice: Float[torch.Tensor, "n_struct 3 3"],
    ptr: Int[torch.Tensor, "n_struct+1"],
    radius: float | Float[torch.Tensor, "n_atoms"],
    pbc: Bool[torch.Tensor, "n_struct 3"] | None = None,
) -> tuple[
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs 3"],
    Int[torch.Tensor, "n_pairs"],
]:
    """Pairs of an atom and a periodic image of an atom which can be within a radius.

    Two points whose fractional coordinates differ by d along a lattice vector
    are at least d times the height of the cell apart. For each pair of atoms
    of a structure, only the images of the second one closer than
    radius / height along each of the three lattice vectors are listed: a box
    around the first atom, which holds every image within the radius.

    Along a lattice vector along which a structure is not periodic, an atom
    has no image: only the atom itself is listed. The pairs of a molecule are
    the pairs of its atoms.

    Args:
        x: Fractional coordinates of the atoms, shape (n_atoms, 3).
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).
        ptr: Index of the first atom of each structure, shape (n_struct+1,).
        radius: Radius in Ångströms. Or a radius per atom, shape (n_atoms,):
            the radius of a pair is then the sum of the ones of its two atoms.
        pbc: Whether each structure is periodic along each lattice vector,
            shape (n_struct, 3). Every structure is periodic by default.

    Returns:
        Index of the first atom, index of the second atom, image of the second
        atom in lattice vectors, shape (n_pairs, 3), and structure index, of
        each pair. Sorted by first atom, then second atom, then image. Every
        pair closer than the radius is listed, with others which are farther.

    Examples:
        One atom in a cubic cell of 3 Å: its images at most one cell away, in a
        box of 3 × 3 × 3 cells.

        >>> source, target, shift, batch = periodic_pairs(
        ...     torch.zeros(1, 3), 3.0 * torch.eye(3)[None], torch.tensor([0, 1]), 3.5
        ... )
        >>> shift.shape, shift.min().item(), shift.max().item()
        (torch.Size([27, 3]), -1, 1)

        The same atom without periodicity along the last lattice vector: a box
        of 3 × 3 × 1 cells.

        >>> pbc = torch.tensor([[True, True, False]])
        >>> source, target, shift, batch = periodic_pairs(
        ...     torch.zeros(1, 3), 3.0 * torch.eye(3)[None], torch.tensor([0, 1]), 3.5, pbc
        ... )
        >>> shift.shape, shift[:, 2].abs().max().item()
        (torch.Size([9, 3]), 0)
    """
    device = x.device

    pairs, pair_batch, _ = batched_cartesian_product_2(ptr, ptr)
    source, target = pairs[:, 0], pairs[:, 1]

    # largest difference of fractional coordinates, with a margin for rounding
    width = 1.0 / lattice_heights(lattice).clamp_min(1e-8)[pair_batch]
    if isinstance(radius, torch.Tensor):
        width = width * (radius[source] + radius[target]).unsqueeze(1) + 1e-4
    else:
        width = width * radius + 1e-4

    # images s of the target with |x[target] + s - x[source]| <= width
    delta = x[source] - x[target]
    low = torch.ceil(delta - width).long()
    count = (torch.floor(delta + width).long() - low + 1).clamp(min=0)

    if pbc is not None:
        # no image along a lattice vector which is not periodic
        periodic = pbc[pair_batch]
        low = torch.where(periodic, low, 0)
        count = torch.where(periodic, count, 1)

    # all the images of each pair: a flat index is a mixed-radix number
    num_images = count.prod(dim=1)
    image_ptr = F.pad(torch.cumsum(num_images, dim=0), (1, 0), value=0)
    pair = torch.arange(len(num_images), device=device).repeat_interleave(num_images)

    local = torch.arange(len(pair), device=device) - image_ptr[pair]

    count_y, count_z = count[pair, 1], count[pair, 2]
    shift = low[pair] + torch.stack(
        [local // (count_y * count_z), (local // count_z) % count_y, local % count_z],
        dim=1,
    )

    return source[pair], target[pair], shift, pair_batch[pair]


def image_order(
    source: Int[torch.Tensor, "n_edges"], shift: Int[torch.Tensor, "n_edges 3"]
) -> Int[torch.Tensor, "n_edges"]:
    """Order of edges by sender, then by periodic image of the receiver.

    Edges with the same sender and image stay in their order: pairs from
    periodic_pairs are then sorted by sender, image and receiver.

    Args:
        source: Sender of each edge, shape (n_edges,).
        shift: Periodic image of the receiver of each edge, shape (n_edges, 3).

    Returns:
        Indices sorting the edges, shape (n_edges,).

    Examples:
        >>> source = torch.tensor([0, 0, 1, 0])
        >>> shift = torch.tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0], [-1, 0, 0]])
        >>> image_order(source, shift)
        tensor([3, 1, 0, 2])
    """
    if len(source) == 0:
        return source

    # one integer per sender and image. It is far from 64 bits: the pairs of a
    # cell with so many images cannot be held in memory.
    largest = int(shift.abs().max())
    base = 2 * largest + 1
    image = shift + largest
    key = ((source * base + image[:, 0]) * base + image[:, 1]) * base + image[:, 2]

    return torch.argsort(key, stable=True)


def triplet_cosines(
    vectors: Float[torch.Tensor, "n_edges 3"],
    triplet_index: Int[torch.Tensor, "2 n_triplets"],
) -> Float[torch.Tensor, "n_triplets"]:
    """Cosine of the angle between the two edges of each triplet.

    Args:
        vectors: Cartesian vector of each edge, shape (n_edges, 3).
        triplet_index: Indices of the two edges of each triplet, which start
            from the same atom, shape (2, n_triplets).

    Returns:
        Cosine of the angle at the shared atom, in [-1, 1], shape (n_triplets,).

    Examples:
        >>> vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]])
        >>> triplet_cosines(vectors, torch.tensor([[0, 0], [1, 2]]))
        tensor([ 0., -1.])
    """
    a = F.normalize(vectors[triplet_index[0]], dim=-1)
    b = F.normalize(vectors[triplet_index[1]], dim=-1)

    return (a * b).sum(dim=-1).clamp(-1.0, 1.0)


def dihedral_angles(
    vectors: Float[torch.Tensor, "n_edges 3"],
    quadruplet_index: Int[torch.Tensor, "3 n_quadruplets"],
) -> Float[torch.Tensor, "n_quadruplets"]:
    """Dihedral angle of each chain of three edges.

    For a chain of atoms u, v, w, x, it is the angle between the planes
    (u, v, w) and (v, w, x) around the bond v-w: 0 when u and x are on the
    same side (cis), pi on opposite sides (trans). Its sign changes in the
    mirror image of the chain.

    Args:
        vectors: Cartesian vector of each edge, shape (n_edges, 3).
        quadruplet_index: Indices of the edges u→v, v→w and w→x of each chain,
            shape (3, n_quadruplets).

    Returns:
        Angle in radians, in [-pi, pi], shape (n_quadruplets,).

    Examples:
        >>> vectors = torch.tensor(
        ...     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
        ... )
        >>> index = torch.tensor([[0, 0], [1, 1], [2, 3]])
        >>> dihedral_angles(vectors, index)  # a quarter turn, then cis
        tensor([1.5708, 0.0000])
    """
    b1, b2, b3 = vectors[quadruplet_index].unbind(dim=0)

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    axis = F.normalize(b2, dim=-1)

    sine = (torch.cross(n1, n2, dim=-1) * axis).sum(dim=-1)
    cosine = (n1 * n2).sum(dim=-1)

    return torch.atan2(sine, cosine)


def edges_distances(
    source_x: Float[torch.Tensor, "n_atoms 3"],
    target_x: Float[torch.Tensor, "n_atoms_2 3"],
    lattice: Float[torch.Tensor, "n_structures 3 3"],
    edges_batch: Int[torch.Tensor, "n_edges"],
    edges_index: Int[torch.Tensor, "2 n_edges"],
) -> Float[torch.Tensor, "n_edges"]:
    edges_frac = target_x[edges_index[1]] - source_x[edges_index[0]]

    return cartesian(edges_frac, lattice, edges_batch).norm(dim=-1)
