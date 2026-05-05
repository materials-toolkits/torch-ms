import torch

from pycrystalgen.data.batching import Structures


def compute_cell_parameters(lattice: torch.Tensor) -> dict:
    """Compute cell parameters (a, b, c, alpha, beta, gamma) from lattice vectors.

    Args:
        lattice: Lattice vectors with shape (3, 3).

    Returns:
        Dict with keys 'a', 'b', 'c', 'alpha', 'beta', 'gamma'.
    """
    a_vec = lattice[0]
    b_vec = lattice[1]
    c_vec = lattice[2]

    a = torch.norm(a_vec).item()
    b = torch.norm(b_vec).item()
    c = torch.norm(c_vec).item()

    cos_alpha = torch.dot(b_vec, c_vec) / (b * c)
    cos_beta = torch.dot(a_vec, c_vec) / (a * c)
    cos_gamma = torch.dot(a_vec, b_vec) / (a * b)

    alpha = torch.rad2deg(torch.acos(cos_alpha)).item()
    beta = torch.rad2deg(torch.acos(cos_beta)).item()
    gamma = torch.rad2deg(torch.acos(cos_gamma)).item()

    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def compare_cell_parameters(
    actual: dict[str, float], expected: dict[str, float]
) -> list[str]:
    errors = []

    for key in ["a", "b", "c"]:
        if abs(actual[key] - expected[key]) >= 0.01:
            errors.append(f"cell {key} mismatch: " f"{actual[key]} != {expected[key]}")

    for key in ["alpha", "beta", "gamma"]:
        if abs(actual[key] - expected[key]) >= 0.1:
            errors.append(f"cell {key} mismatch: " f"{actual[key]} != {expected[key]}")

    return errors


def compare_structures(
    actual: Structures, expected: Structures, coord_tol: float = 1e-2
) -> list[str]:
    """Compare two Structures objects.

    Args:
        actual: The Structures object to validate.
        expected: The expected Structures object.
        coord_tol: Tolerance for coordinate comparison.

    Returns:
        List of error messages (empty if all match).
    """
    errors = []

    n_struct_actual = len(actual.lattice)
    n_struct_expected = len(expected.lattice)

    if n_struct_actual != n_struct_expected:
        errors.append(
            f"Structure count mismatch: {n_struct_actual} != {n_struct_expected}"
        )
        return errors

    if actual.x.shape[0] != expected.x.shape[0]:
        errors.append(
            f"Atom count mismatch: {actual.x.shape[0]} != {expected.x.shape[0]}"
        )

    for i in range(n_struct_actual):
        cell_actual = compute_cell_parameters(actual.lattice[i])
        cell_expected = compute_cell_parameters(expected.lattice[i])

        for key in ["a", "b", "c"]:
            if abs(cell_actual[key] - cell_expected[key]) >= 0.01:
                errors.append(
                    f"Structure {i} cell {key} mismatch: "
                    f"{cell_actual[key]} != {cell_expected[key]}"
                )

        for key in ["alpha", "beta", "gamma"]:
            if abs(cell_actual[key] - cell_expected[key]) >= 0.1:
                errors.append(
                    f"Structure {i} cell {key} mismatch: "
                    f"{cell_actual[key]} != {cell_expected[key]}"
                )

    errors.extend(compare_atoms(actual, expected, coord_tol))

    return errors


def compare_atoms(
    actual: Structures,
    expected: Structures,
    coord_tol: float = 1e-3,
) -> list[str]:
    errors = []

    if actual.x.shape[0] != expected.x.shape[0]:
        return ["Atom count mismatch"]

    expected_supercell = _generate_centered_supercell(expected)

    distance = torch.linalg.norm(actual.x[None] - expected_supercell.x[:, None], dim=2)

    min_dist = distance.min(dim=0).values

    for i, dist in enumerate(min_dist):

        if dist > coord_tol:
            errors.append(f"structures {i} failled")
            continue

    return errors


def _generate_centered_supercell(
    structures: Structures,
    repeat: tuple[int, int, int] = (3, 3, 3),
    wrap_tolerance: float = 1e-5,
) -> Structures:
    """Generate a supercell by tiling the unit cell.

    Args:
        structures: Single-structure Structures object to expand.
        repeat: Number of repeats along a, b, c axes (default: (3, 3, 1)).
        wrap_tolerance: Tolerance for wrapping fractional coordinates.

    Returns:
        Structures object with expanded positions in fractional coordinates [0, repeat).
    """
    lattice = structures.lattice[0]
    positions = structures.x
    atomic_numbers = structures.z

    i_count, j_count, k_count = repeat
    i_offset, j_offset, k_offset = (-i_count // 2, -j_count // 2, -k_count // 2)

    i_vals = torch.arange(
        i_offset, i_count + i_offset, dtype=torch.float32, device=lattice.device
    )
    j_vals = torch.arange(
        j_offset, j_count + j_offset, dtype=torch.float32, device=lattice.device
    )
    k_vals = torch.arange(
        k_offset, k_count + k_offset, dtype=torch.float32, device=lattice.device
    )

    translations = torch.cartesian_prod(i_vals, j_vals, k_vals)

    num_original = positions.shape[0]
    num_tiles = translations.shape[0]

    tiled_positions = positions.unsqueeze(0).expand(num_tiles, -1, -1)
    tiled_positions = tiled_positions + translations.unsqueeze(1)

    tiled_positions = torch.remainder(tiled_positions + wrap_tolerance, float(i_count))

    all_positions = tiled_positions.view(-1, 3)
    all_atomic_numbers = atomic_numbers.repeat(num_tiles)

    return Structures(
        lattice=lattice,
        x=all_positions,
        z=all_atomic_numbers,
        _validate=True,
    )
