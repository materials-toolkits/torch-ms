"""Group actions on batched structures.

Every function takes the element of the group to apply, one per structure, and
returns new structures describing the same crystals. They are deterministic:
the elements are drawn in torchms.data.augmentation.sampling.

Rotations, permutations and translations apply to molecules as to crystals.
The actions on the cell, its basis and its repetition, leave a structure which
is not periodic unchanged.
"""

import dataclasses

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from torchms.data.batching import Structures
from torchms.graph.sparse import batched_cartesian_product_3


def wrap(x: Float[torch.Tensor, "n 3"]) -> Float[torch.Tensor, "n 3"]:
    """Wrap fractional coordinates to [0, 1).

    The remainder of a tiny negative number is rounded to 1.0, it is wrapped a
    second time.

    Examples:
        >>> wrap(torch.tensor([[1.25, -0.25, -1e-12]]))
        tensor([[0.2500, 0.7500, 0.0000]])
    """
    return torch.remainder(torch.remainder(x, 1.0), 1.0)


def wrap_periodic(
    structures: Structures, x: Float[torch.Tensor, "n_atoms 3"]
) -> Float[torch.Tensor, "n_atoms 3"]:
    """Wrap coordinates to [0, 1) along the periodic lattice vectors only.

    Args:
        structures: Structures of the atoms, for their periodicity.
        x: Coordinates of their atoms, shape (n_atoms, 3).

    Examples:
        >>> crystal, molecule = Structures.diamond(), Structures.water()
        >>> wrap_periodic(crystal, crystal.x + 0.75)[0]
        tensor([0.7500, 0.7500, 0.7500])
        >>> wrap_periodic(molecule, molecule.x + 1.5)[0]
        tensor([1.5000, 1.5000, 1.5000])
    """
    return torch.where(structures.pbc[structures.batch], wrap(x), x)


def permute_atoms(
    structures: Structures, permutation: Int[torch.Tensor, "n_atoms"]
) -> Structures:
    """Reorder the atoms inside each structure.

    Args:
        structures: Batched Structures object.
        permutation: Index into the atoms of the batch of the atom moved at
            each position, shape (n_atoms,). An atom must stay in its structure.

    Returns:
        Structures with reordered x and z.

    Raises:
        ValueError: If the permutation moves an atom to another structure.

    Examples:
        >>> structures = Structures(
        ...     lattice=torch.eye(3).unsqueeze(0), x=torch.zeros(3, 3), z=[1, 6, 8]
        ... )
        >>> permute_atoms(structures, torch.tensor([2, 0, 1])).z
        tensor([8, 1, 6])
    """
    if (structures.batch[permutation] != structures.batch).any():
        raise ValueError("atoms must be permuted inside their structure")

    return dataclasses.replace(
        structures.without_graphs(),
        x=structures.x[permutation],
        z=structures.z[permutation],
    )


def rotate_lattice(
    structures: Structures, rotation: Float[torch.Tensor, "n_struct 3 3"]
) -> Structures:
    """Apply an element of O(3) to the lattice of each structure.

    Every lattice vector v becomes R v. The fractional coordinates are unchanged,
    the cartesian positions are rotated with the lattice.

    Args:
        structures: Batched Structures object.
        rotation: Orthogonal matrices, rotations or rotoreflections, with shape
            (n_struct, 3, 3).

    Returns:
        Structures with the transformed lattice.

    Examples:
        A rotation of 90° around the z axis sends the x axis on the y axis.

        >>> structures = Structures.diamond(a=1.0)
        >>> rotation = torch.tensor(
        ...     [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        ... )
        >>> rotate_lattice(structures, rotation).lattice[0, 0]
        tensor([0., 1., 0.])
    """
    # rows of the lattice are the lattice vectors
    lattice = torch.bmm(structures.lattice, rotation.transpose(1, 2))

    return dataclasses.replace(structures.without_graphs(), lattice=lattice)


def translate_atoms(
    structures: Structures, translation: Float[torch.Tensor, "n_struct 3"]
) -> Structures:
    """Translate all the atoms of each structure, the lattice is unchanged.

    Args:
        structures: Batched Structures object.
        translation: Translation in fractional coordinates, shape (n_struct, 3).

    Returns:
        Structures with translated x, wrapped to [0, 1) along the periodic
        lattice vectors.

    Examples:
        >>> structures = Structures.diamond()
        >>> translate_atoms(structures, torch.tensor([[0.5, 0.75, 1.0]])).x[0]
        tensor([0.5000, 0.7500, 0.0000])
    """
    x = wrap_periodic(structures, structures.x + translation[structures.batch])

    return dataclasses.replace(structures.without_graphs(), x=x)


def unimodular_inverse(
    basis: Int[torch.Tensor, "n_struct 3 3"],
) -> Int[torch.Tensor, "n_struct 3 3"]:
    """Exact inverse of integer matrices of determinant 1.

    Raises:
        ValueError: If a determinant is not 1.

    Examples:
        >>> unimodular_inverse(torch.tensor([[[1, 2, 0], [0, 1, 0], [0, 0, 1]]]))
        tensor([[[ 1, -2,  0],
                 [ 0,  1,  0],
                 [ 0,  0,  1]]])
    """
    # the adjugate: its columns are the cross products of the rows
    adjugate = torch.cross(basis[:, [1, 2, 0]], basis[:, [2, 0, 1]], dim=-1)
    det = (basis[:, 0] * adjugate[:, 0]).sum(-1)

    if (det != 1).any():
        raise ValueError("the determinant of the change of basis must be 1")

    return adjugate.transpose(1, 2)


def change_lattice_basis(
    structures: Structures, basis: Int[torch.Tensor, "n_struct 3 3"]
) -> Structures:
    """Apply an element of SL(3, Z) to the lattice of each structure.

    The new lattice vectors are integer combinations of the previous ones,
    they generate the same lattice: the cell changes, not the crystal. The
    cartesian positions are unchanged, up to a vector of the lattice.

    A structure which is not periodic along its three lattice vectors has no
    such lattice: it is left unchanged.

    Args:
        structures: Batched Structures object.
        basis: Integer matrices of determinant 1 with shape (n_struct, 3, 3).
            Row i gives the new lattice vector i from the previous ones.

    Returns:
        Structures with the new lattice and x expressed in it, wrapped to [0, 1).

    Raises:
        ValueError: If a determinant is not 1.

    Examples:
        Replace the first lattice vector a by a + b.

        >>> structures = Structures.diamond(a=1.0)
        >>> basis = torch.tensor([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
        >>> sheared = change_lattice_basis(structures, basis)
        >>> sheared.lattice[0, 0]
        tensor([1., 1., 0.])
        >>> sheared.x[1]  # (0.5, 0.5, 0) in the previous cell
        tensor([0.5000, 0.0000, 0.0000])
    """
    identity = torch.eye(3, dtype=basis.dtype, device=basis.device)
    basis = torch.where(structures.pbc.all(dim=1)[:, None, None], basis, identity)

    inverse = unimodular_inverse(basis).to(structures.x.dtype)

    # cartesian = x @ lattice = (x @ basis^-1) @ (basis @ lattice)
    lattice = torch.bmm(basis.to(structures.lattice.dtype), structures.lattice)
    x = torch.bmm(structures.x.unsqueeze(1), inverse[structures.batch]).squeeze(1)

    return dataclasses.replace(
        structures.without_graphs(), lattice=lattice, x=wrap_periodic(structures, x)
    )


def make_supercell(
    structures: Structures, repeat: Int[torch.Tensor, "n_struct 3"]
) -> Structures:
    """Repeat the cell of each structure along its lattice vectors.

    The atoms are ordered by cell, the atoms of a cell keep their order. The
    total energy is the one of the supercell, the other properties are
    intensive and unchanged, the ones of Structures.properties included.

    With a change of basis before and after, every supercell is reached (Smith
    normal form of its integer matrix).

    Args:
        structures: Batched Structures object.
        repeat: Number of cells along each lattice vector, at least 1, with
            shape (n_struct, 3).

    Returns:
        Structures with prod(repeat) times the atoms of each structure.

    Raises:
        ValueError: If a number of cells is lower than 1, or greater than 1
            along a lattice vector along which the structure is not periodic.

    Examples:
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende()]
        ... )
        >>> supercell = make_supercell(structures, torch.tensor([[2, 1, 1], [1, 3, 2]]))
        >>> supercell.num_atoms
        tensor([16, 48])
        >>> supercell.lattice[0].diagonal()
        tensor([7.1340, 3.5670, 3.5670])
    """
    if (repeat < 1).any():
        raise ValueError("a supercell must repeat the cell at least once")

    if ((repeat > 1) & ~structures.pbc).any():
        raise ValueError(
            "a structure cannot be repeated along a lattice vector along which "
            "it is not periodic"
        )

    device = structures.device()

    # all the cells of the supercells: (total_cells, 3)
    ptr_a = F.pad(torch.cumsum(repeat[:, 0], dim=0), (1, 0), value=0)
    ptr_b = F.pad(torch.cumsum(repeat[:, 1], dim=0), (1, 0), value=0)
    ptr_c = F.pad(torch.cumsum(repeat[:, 2], dim=0), (1, 0), value=0)
    cell_idx, cell_batch, cell_ptr = batched_cartesian_product_3(ptr_a, ptr_b, ptr_c)
    shifts = cell_idx - torch.stack(
        [ptr_a[cell_batch], ptr_b[cell_batch], ptr_c[cell_batch]], dim=1
    )

    # expand: cells (outer) × atoms (inner)
    num_atoms_per_cell = structures.num_atoms[cell_batch]
    output_ptr = F.pad(torch.cumsum(num_atoms_per_cell, dim=0), (1, 0), value=0)

    atom_cell = torch.arange(len(cell_batch), device=device).repeat_interleave(
        num_atoms_per_cell
    )
    local_atom = (
        torch.arange(int(output_ptr[-1]), device=device) - output_ptr[atom_cell]
    )
    atom_struct = cell_batch[atom_cell]
    atom_index = local_atom + structures.ptr[atom_struct]

    num_cells = cell_ptr[1:] - cell_ptr[:-1]
    num_atoms = num_cells * structures.num_atoms

    x = (structures.x[atom_index] + shifts[atom_cell]) / repeat[atom_struct]

    return dataclasses.replace(
        structures.without_graphs(),
        lattice=structures.lattice * repeat.unsqueeze(-1),
        x=x.to(structures.x.dtype),
        z=structures.z[atom_index],
        num_atoms=num_atoms,
        batch=atom_struct,
        ptr=F.pad(torch.cumsum(num_atoms, dim=0), (1, 0), value=0),
        total_energy=structures.total_energy * num_cells,
    )
