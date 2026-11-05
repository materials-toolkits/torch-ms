"""Symmetry operations on crystal structures."""

from __future__ import annotations

import torch

from torchms.data.batching import Structures
from torchms.data.constants import get_symmetry_operations

# atoms closer than this in fractional coordinates are one, as in
# ase.spacegroup.crystal (symprec)
MERGE_TOLERANCE = 1e-3


def generate_symmetric_structure(
    lattice: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    spacegroup: int | str,
    tol: float = MERGE_TOLERANCE,
) -> Structures:
    """Generate full symmetric crystal by applying spacegroup operations.

    The operations are the ones of the standard setting of the group, see
    torchms.data.constants.get_symmetry_operations.

    Args:
        lattice: Lattice vectors with shape (3, 3) in Angstroms.
        positions: Atomic positions in fractional coordinates [0, 1) with shape (N, 3).
        atomic_numbers: Atomic numbers with shape (N,).
        spacegroup: Space group number (int) or symbol (str).
        tol: Atoms closer than this, in fractional coordinates and across the
            faces of the cell, are merged.

    Returns:
        Structures object with all symmetry-equivalent atoms.
    """
    operations = get_symmetry_operations(spacegroup).to(positions.dtype)

    return expand_symmetry(lattice, positions, atomic_numbers, operations, tol)


def expand_symmetry(
    lattice: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    operations: torch.Tensor,
    tol: float = MERGE_TOLERANCE,
) -> Structures:
    """Apply symmetry operations to the atoms of an asymmetric unit.

    Args:
        lattice: Lattice vectors with shape (3, 3) in Angstroms.
        positions: Fractional coordinates of the atoms, shape (N, 3).
        atomic_numbers: Atomic numbers, shape (N,).
        operations: Affine operations on fractional coordinates, shape
            (n_ops, 4, 4): [R|t; 0 0 0 1].
        tol: Images closer than this, in fractional coordinates and across the
            faces of the cell, are one atom: the image of the first atom of
            the asymmetric unit, whatever the elements, as in
            ase.spacegroup.crystal.

    Returns:
        Structures with one atom per distinct image, in the order of the atoms
        then of the operations.

    Examples:
        >>> identity = torch.eye(4)
        >>> inversion = torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0]))
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        >>> crystal = expand_symmetry(
        ...     torch.eye(3), x, torch.tensor([11, 17]),
        ...     torch.stack([identity, inversion]),
        ... )
        >>> crystal.x.tolist()
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
        >>> crystal.z.tolist()
        [11, 17, 17]
    """
    rot = operations[:, :3, :3].to(positions.dtype)
    trans = operations[:, :3, 3].to(positions.dtype)

    # images of each atom in turn
    new_positions = torch.einsum("gmn,xn->xgm", rot, positions) + trans
    all_positions = torch.remainder(new_positions.reshape(-1, 3), 1.0)
    all_atomic_numbers = atomic_numbers.repeat_interleave(operations.shape[0])

    unique_mask = _find_unique_atoms(all_positions, tol)

    return Structures(
        lattice=lattice.unsqueeze(0),
        x=all_positions[unique_mask],
        z=all_atomic_numbers[unique_mask],
        _validate=True,
    )


def _find_unique_atoms(
    positions: torch.Tensor, tol: float = MERGE_TOLERANCE
) -> torch.Tensor:
    """Mask of the first atom of each group of atoms at the same place.

    Two atoms are at the same place when their fractional coordinates differ
    by less than tol, modulo 1: 0.0 and 0.9999 are the same coordinate.

    Args:
        positions: Fractional positions with shape (N, 3).
        tol: Tolerance on the distance in fractional coordinates.

    Returns:
        Boolean mask of the atoms kept, shape (N,).
    """
    difference = positions.unsqueeze(1) - positions.unsqueeze(0)
    difference = difference - torch.round(difference)
    same_place = torch.sum(difference**2, dim=-1) < tol * tol

    # an atom is dropped when an atom before it is at the same place
    duplicate = torch.tril(same_place, diagonal=-1).any(dim=1)

    return ~duplicate
