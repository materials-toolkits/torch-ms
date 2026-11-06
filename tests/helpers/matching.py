"""The permutation between two orders of the same rows of integer keys."""

from __future__ import annotations

import torch


def permutation(found: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Index which puts rows in the order of other rows with the same keys.

    Args:
        found: Integer keys of each row, shape (n, k): the pair of edges of a
            triplet, (sender, receiver, shift) of an edge.
        expected: The same keys in another order, shape (n, k).

    Returns:
        index such that found[index] equals expected, shape (n,).

    Raises:
        AssertionError: If the rows are not the same, or a key is repeated.

    Examples:
        >>> found = torch.tensor([[0, 1], [2, 3], [1, 0]])
        >>> expected = torch.tensor([[1, 0], [0, 1], [2, 3]])
        >>> index = permutation(found, expected)
        >>> index.tolist(), torch.equal(found[index], expected)
        ([2, 0, 1], True)
    """
    if found.shape != expected.shape:
        raise AssertionError(
            f"{found.shape[0]} rows, expected {expected.shape[0]} "
            f"(shapes {tuple(found.shape)} and {tuple(expected.shape)})"
        )

    order_found = _lexicographic_order(found)
    order_expected = _lexicographic_order(expected)
    if not torch.equal(found[order_found], expected[order_expected]):
        raise AssertionError("the rows are not the expected ones")
    if (found[order_found][1:] == found[order_found][:-1]).all(dim=1).any():
        raise AssertionError("a key is repeated: the order is not defined")

    index = torch.empty_like(order_found)
    index[order_expected] = order_found

    return index


def _lexicographic_order(keys: torch.Tensor) -> torch.Tensor:
    """Order of rows sorted by their first column, then the second..."""
    order = torch.arange(len(keys), device=keys.device)
    for column in reversed(range(keys.shape[1])):
        order = order[torch.sort(keys[order, column], stable=True).indices]

    return order
