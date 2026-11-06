from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from torchms.data.batching import Structures
from torchms.graph.sparse import (
    batched_cartesian_product_2,
    batched_cartesian_product_3,
)

if TYPE_CHECKING:
    from torchms.graph.graph import EdgeTuples


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


def lattice_from_parameters(
    lengths: Float[torch.Tensor, "n_struct 3"],
    angles: Float[torch.Tensor, "n_struct 3"],
) -> Float[torch.Tensor, "n_struct 3 3"]:
    """Lattice vectors of cells from their lengths and their angles.

    The orientation is the one of pymatgen: c is along z and a in the plane
    (x, z).

    Args:
        lengths: Lengths a, b and c in Å, shape (n_struct, 3).
        angles: Angles alpha, beta and gamma in degrees, shape (n_struct, 3).

    Returns:
        Lattice vectors as rows, shape (n_struct, 3, 3).

    Examples:
        >>> lengths = torch.tensor([[2.0, 3.0, 4.0]])
        >>> lattice = lattice_from_parameters(lengths, torch.full((1, 3), 90.0))
        >>> lattice.round(decimals=4) + 0.0
        tensor([[[2., 0., 0.],
                 [0., 3., 0.],
                 [0., 0., 4.]]])
    """
    radians = torch.deg2rad(angles)
    cos, sin = torch.cos(radians), torch.sin(radians)
    zero = torch.zeros_like(lengths[:, 0])

    # angle between the planes (a, c) and (b, c)
    gamma_star = torch.arccos(
        ((cos[:, 0] * cos[:, 1] - cos[:, 2]) / (sin[:, 0] * sin[:, 1])).clamp(-1.0, 1.0)
    )

    vector_a = torch.stack([lengths[:, 0] * sin[:, 1], zero, lengths[:, 0] * cos[:, 1]])
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sin[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sin[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * cos[:, 0],
        ]
    )
    vector_c = torch.stack([zero, zero, lengths[:, 2]])

    return torch.stack([vector_a.t(), vector_b.t(), vector_c.t()], dim=1)


def lattice_parameters(
    lattice: Float[torch.Tensor, "n_struct 3 3"],
) -> tuple[Float[torch.Tensor, "n_struct 3"], Float[torch.Tensor, "n_struct 3"]]:
    """Lengths and angles of cells, the inverse of lattice_from_parameters.

    Args:
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).

    Returns:
        Lengths a, b and c in Å, and angles alpha (between b and c), beta
        (between a and c) and gamma (between a and b) in degrees, each of
        shape (n_struct, 3).

    Examples:
        >>> lengths = torch.tensor([[3.0, 4.0, 5.0]])
        >>> angles = torch.tensor([[80.0, 95.0, 110.0]])
        >>> found = lattice_parameters(lattice_from_parameters(lengths, angles))
        >>> [values.round(decimals=3) + 0.0 for values in found]
        [tensor([[3., 4., 5.]]), tensor([[ 80.,  95., 110.]])]
    """
    lengths = lattice.norm(dim=-1)

    # the angle opposite to each vector: between the two other ones
    others = lattice.roll(-1, dims=1), lattice.roll(-2, dims=1)
    norms = lengths.roll(-1, dims=1) * lengths.roll(-2, dims=1)
    cosines = (others[0] * others[1]).sum(dim=-1) / norms
    angles = torch.rad2deg(torch.arccos(cosines.clamp(-1.0, 1.0)))

    return lengths, angles


def polar_logarithm(
    lattice: Float[torch.Tensor, "n_struct 3 3"],
) -> Float[torch.Tensor, "n_struct 3 3"]:
    """Logarithm S of the symmetric part of the polar decomposition of cells.

    The symmetric part P = (L L^T)^(1/2) of a lattice L is exp(S): S is
    symmetric, and does not change with a rotation of the cell.

    Args:
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).

    Returns:
        S, shape (n_struct, 3, 3).

    Examples:
        >>> cell = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
        >>> torch.allclose(torch.matrix_exp(polar_logarithm(cell[None])), cell)
        True
    """
    gram = lattice @ lattice.transpose(-1, -2)
    # pylint: disable-next=not-callable
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    log = torch.diag_embed(eigenvalues.log()) / 2

    return eigenvectors @ log @ eigenvectors.transpose(-1, -2)


def polar_lattice_parameters(
    lattice: Float[torch.Tensor, "n_struct 3 3"],
) -> Float[torch.Tensor, "n_struct 6"]:
    """Six numbers of a cell which do not change with its rotation, as in CrystalFlow.

    The symmetric part of the polar decomposition of the lattice,
    P = (L L^T)^(1/2), is exp(S) for a symmetric matrix S: k holds S_01, S_02,
    S_12, (S_00 - S_11) / 2, (S_00 + S_11 - 2 S_22) / 6 and the mean of its
    diagonal, the logarithm of the cubic root of the volume. See
    lattice_from_polar_parameters for the inverse.

    Args:
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).

    Returns:
        k, shape (n_struct, 6).

    Examples:
        >>> k = polar_lattice_parameters(2.0 * torch.eye(3).unsqueeze(0))
        >>> k.round(decimals=4) + 0.0
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6931]])
    """
    s = polar_logarithm(lattice)

    return torch.stack(
        [
            s[:, 0, 1],
            s[:, 0, 2],
            s[:, 1, 2],
            (s[:, 0, 0] - s[:, 1, 1]) / 2,
            (s[:, 0, 0] + s[:, 1, 1] - 2 * s[:, 2, 2]) / 6,
            (s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]) / 3,
        ],
        dim=1,
    )


def lattice_from_polar_parameters(
    k: Float[torch.Tensor, "n_struct 6"],
) -> Float[torch.Tensor, "n_struct 3 3"]:
    """The symmetric lattice exp(S) of six numbers k, see polar_lattice_parameters.

    The lattice has the lengths and the angles of the cell k was computed
    from, in another orientation: a symmetric matrix.

    Examples:
        >>> vectors = [[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.2, 5.0]]
        >>> lattice = torch.tensor([vectors])
        >>> rebuilt = lattice_from_polar_parameters(polar_lattice_parameters(lattice))
        >>> torch.allclose(rebuilt @ rebuilt.mT, lattice @ lattice.mT, atol=1e-4)
        True
        >>> torch.allclose(rebuilt, rebuilt.mT)
        True
    """
    diagonal_0 = k[:, 3] + k[:, 4] + k[:, 5]
    diagonal_1 = -k[:, 3] + k[:, 4] + k[:, 5]
    diagonal_2 = -2 * k[:, 4] + k[:, 5]
    s = torch.stack(
        [
            torch.stack([diagonal_0, k[:, 0], k[:, 1]], dim=1),
            torch.stack([k[:, 0], diagonal_1, k[:, 2]], dim=1),
            torch.stack([k[:, 1], k[:, 2], diagonal_2], dim=1),
        ],
        dim=1,
    )

    return torch.matrix_exp(s)


def symmetric_lattice(
    lattice: Float[torch.Tensor, "n_struct 3 3"],
) -> Float[torch.Tensor, "n_struct 3 3"]:
    """Symmetric lattice of the same cell, rotated: its polar decomposition.

    A matrix L is a rotation U times a symmetric matrix: the symmetric matrix
    U P U^T, with P = (L^T L)^(1/2), has the lengths and the angles of L. It
    is computed with a singular value decomposition, as in MatterGen, whose
    diffusion keeps its lattices symmetric.

    Args:
        lattice: Lattice vectors as rows, shape (n_struct, 3, 3).

    Returns:
        Symmetric lattice vectors, shape (n_struct, 3, 3).

    Examples:
        >>> rows = [[0.0, 2.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 0.0, 4.0]]
        >>> lattice = torch.tensor([rows])
        >>> symmetric = symmetric_lattice(lattice)
        >>> torch.allclose(symmetric, symmetric.transpose(1, 2), atol=1e-6)
        True
        >>> symmetric.norm(dim=-1).round(decimals=4)
        tensor([[2., 3., 4.]])
    """
    w, s, v_transposed = torch.linalg.svd(lattice)
    v = v_transposed.transpose(1, 2)
    rotation = w @ v_transposed
    positive = v @ torch.diag_embed(s) @ v_transposed

    return rotation @ positive @ rotation.transpose(1, 2)


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
    max_shift: int | None = None,
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
        max_shift: If given, only the images at most this number of cells
            away along each lattice vector are listed. A flat cell has images
            within the radius very far along the lattice vector of its small
            height: their number is then bounded before they are listed.

    Returns:
        Index of the first atom, index of the second atom, image of the second
        atom in lattice vectors, shape (n_pairs, 3), and structure index, of
        each pair. Sorted by first atom, then second atom, then image; in a
        batch of crystals and molecules, the pairs of the molecules come after
        the ones of the crystals. Every pair closer than the radius is listed,
        with others which are farther.

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

        A flat cell, 0.001 Å high: its images up to 2 cells away along the
        third lattice vector, not the 3,500 within the radius.

        >>> flat = torch.diag(torch.tensor([3.0, 3.0, 1e-3]))[None]
        >>> source, target, shift, batch = periodic_pairs(
        ...     torch.zeros(1, 3), flat, torch.tensor([0, 1]), 3.5, max_shift=2
        ... )
        >>> shift.shape, shift[:, 2].abs().max().item()
        (torch.Size([45, 3]), 2)
    """
    device = x.device

    pairs, pair_batch, _ = batched_cartesian_product_2(ptr, ptr)
    source, target = pairs[:, 0], pairs[:, 1]

    # A pair of atoms of a molecule has no image to list: these pairs are set
    # apart, which saves most of the time of the graphs of molecules. The
    # number of molecules is read once from the device.
    molecules: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    if pbc is not None:
        molecule = ~pbc.any(dim=1)
        num_molecules = int(molecule.sum())

        if num_molecules == len(molecule):
            shift = torch.zeros((len(source), 3), dtype=torch.int64, device=device)
            return source, target, shift, pair_batch

        if num_molecules > 0:
            in_molecule = molecule[pair_batch]
            alone = in_molecule.nonzero().squeeze(1)
            others = (~in_molecule).nonzero().squeeze(1)

            molecules = (source[alone], target[alone], pair_batch[alone])
            source, target = source[others], target[others]
            pair_batch = pair_batch[others]

    # largest difference of fractional coordinates, with a margin for rounding
    width = 1.0 / lattice_heights(lattice).clamp_min(1e-8)[pair_batch]
    if isinstance(radius, torch.Tensor):
        width = width * (radius[source] + radius[target]).unsqueeze(1) + 1e-4
    else:
        width = width * radius + 1e-4

    # images s of the target with |x[target] + s - x[source]| <= width
    delta = x[source] - x[target]
    low = torch.ceil(delta - width).long()
    high = torch.floor(delta + width).long()
    if max_shift is not None:
        low, high = low.clamp(min=-max_shift), high.clamp(max=max_shift)
    count = (high - low + 1).clamp(min=0)

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

    source, target, pair_batch = source[pair], target[pair], pair_batch[pair]

    if molecules is not None:
        source = torch.cat([source, molecules[0]])
        target = torch.cat([target, molecules[1]])
        pair_batch = torch.cat([pair_batch, molecules[2]])
        shift = F.pad(shift, (0, 0, 0, len(molecules[0])))

    return source, target, shift, pair_batch


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
    triplet_index: Int[torch.Tensor, "2 n_triplets"] | EdgeTuples,
) -> Float[torch.Tensor, "n_triplets"]:
    """Cosine of the angle between the two edges of each triplet.

    Args:
        vectors: Cartesian vector of each edge, shape (n_edges, 3).
        triplet_index: Triplets as EdgeTuples, or indices of the two edges of
            each triplet, which start from the same atom, shape (2, n_triplets).

    Returns:
        Cosine of the angle at the shared atom, in [-1, 1], shape (n_triplets,).

    Examples:
        >>> vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]])
        >>> triplet_cosines(vectors, torch.tensor([[0, 0], [1, 2]]))
        tensor([ 0., -1.])
    """
    if not isinstance(triplet_index, torch.Tensor):
        triplet_index = triplet_index.index

    a = F.normalize(vectors[triplet_index[0]], dim=-1)
    b = F.normalize(vectors[triplet_index[1]], dim=-1)

    return (a * b).sum(dim=-1).clamp(-1.0, 1.0)


def pair_cosines(
    vectors: Float[torch.Tensor, "n_edges 3"],
    pair_index: Int[torch.Tensor, "2 n_pairs"] | EdgeTuples,
) -> Float[torch.Tensor, "n_pairs"]:
    """Cosine of the angle of each pair of edges head to tail u→v, v→w.

    Args:
        vectors: Cartesian vector of each edge, shape (n_edges, 3).
        pair_index: Pairs of a line graph as EdgeTuples, or indices of the two
            edges of each pair, the first one arrives at the atom the second
            one leaves, shape (2, n_pairs).

    Returns:
        Cosine of the angle at v between u and w, in [-1, 1], shape (n_pairs,).
        It is 1 for an edge followed by its reverse.

    Examples:
        >>> vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]])
        >>> pair_cosines(vectors, torch.tensor([[0, 0], [1, 2]]))
        tensor([-0., 1.])
    """
    return -triplet_cosines(vectors, pair_index)


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
