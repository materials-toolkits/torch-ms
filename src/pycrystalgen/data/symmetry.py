"""Symmetry operations on crystal structures."""

from __future__ import annotations

import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.data.constants import get_symmetry_operations


def generate_symmetric_structure(
    lattice: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    spacegroup: int | str,
    tol: float = 1e-5,
) -> Structures:
    """Generate full symmetric crystal by applying spacegroup operations.

    Args:
        lattice: Lattice vectors with shape (3, 3) in Angstroms.
        positions: Atomic positions in fractional coordinates [0, 1) with shape (N, 3).
        atomic_numbers: Atomic numbers with shape (N,).
        spacegroup: Space group number (int) or symbol (str).
        wrap_tolerance: Tolerance for wrapping fractional coordinates.

    Returns:
        Structures object with all symmetry-equivalent atoms.
    """
    ops = get_symmetry_operations(spacegroup)

    num_atoms = positions.shape[0]
    num_ops = ops.shape[0]

    rot = ops[:, :3, :3]
    trans = ops[:, :3, 3]

    new_positions = torch.einsum("gmn,xn->gxm", rot, positions) + trans[:, None]
    new_positions = new_positions.reshape(-1, 3)

    new_positions = torch.remainder(new_positions, 1.0)

    new_atomic_numbers = atomic_numbers.repeat(num_ops)

    all_positions = new_positions.view(-1, 3)
    all_atomic_numbers = new_atomic_numbers.view(-1)

    unique_mask = _find_unique_atoms(all_positions, all_atomic_numbers, tol)
    final_positions = all_positions[unique_mask]
    final_atomic_numbers = all_atomic_numbers[unique_mask]

    return Structures(
        lattice=lattice.unsqueeze(0),
        x=final_positions,
        z=final_atomic_numbers,
        _validate=True,
    )


def _find_unique_atoms(
    positions: torch.Tensor, atomic_numbers: torch.Tensor, tol: float = 1e-5
) -> torch.Tensor:
    """Find indices of unique atoms, considering both position and atomic number.

    Uses sorting for O(n log n) complexity instead of O(n²) loop.

    Args:
        positions: Fractional positions with shape (N, 3).
        atomic_numbers: Atomic numbers with shape (N,).
        tol: Tolerance for position comparison.

    Returns:
        Boolean mask of unique atoms.
    """
    n = positions.shape[0]
    if n <= 1:
        return torch.ones(n, dtype=torch.bool, device=positions.device)

    sorted_idx = torch.argsort(atomic_numbers)
    sorted_z = atomic_numbers[sorted_idx]
    sorted_pos = positions[sorted_idx]

    pos_diff = sorted_pos.unsqueeze(1) - sorted_pos.unsqueeze(0)
    dist_sq = torch.sum(pos_diff**2, dim=-1)

    z_diff = sorted_z.unsqueeze(1) - sorted_z.unsqueeze(0)
    same_z = z_diff == 0

    same_pos = dist_sq < tol * tol
    duplicate = same_pos & same_z

    mask = torch.ones(n, dtype=torch.bool, device=positions.device)
    for i in range(n):
        if duplicate[i, :i].any():
            mask[sorted_idx[i]] = False

    return mask
