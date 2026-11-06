"""Comparison of a result with what an original code computed."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from torchms.data import Structures


def assert_close(
    found: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    name: str = "values",
    scale: float | None = None,
) -> None:
    """Assert values are the expected ones up to the rounding of float32.

    The largest difference is under rtol times the largest expected value, as
    in .claude/skills/port-model/SKILL.md: small values are held to the same
    relative precision as large ones (scores of 1e-3, not only 1e-5 absolute).

    A sum whose terms cancel, the move of an atom summed over its edges, is
    held to the size of its terms instead: give it as scale.

    Raises:
        AssertionError: If the shapes differ, or the difference is larger,
            with the largest absolute and relative differences.

    Examples:
        >>> assert_close(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0 + 1e-6]))
        >>> assert_close(torch.tensor([1.0]), torch.tensor([2.0]), name="e")
        Traceback (most recent call last):
        ...
        AssertionError: e: largest difference 1.0e+00 (0.5 of the scale 2), ...
    """
    if found.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape {tuple(found.shape)}, expected {tuple(expected.shape)}"
        )

    if scale is None:
        largest = float(expected.abs().max()) if expected.numel() else 0.0
        scale = largest if largest > 0.0 else 1.0
    difference = float((found.double() - expected.double()).abs().max())
    if difference >= rtol * scale:
        raise AssertionError(
            f"{name}: largest difference {difference:.1e} "
            f"({difference / scale:.2g} of the scale {scale:.2g}), "
            f"tolerance {rtol * scale:.1e}"
        )


def assert_close_in_the_cell(
    found: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5, name: str = "x"
) -> None:
    """Assert fractional coordinates are the expected ones, modulo the cell.

    Examples:
        >>> assert_close_in_the_cell(torch.tensor([0.0, 0.5]), torch.tensor([0.999999, 0.5]))
    """
    difference = (found - expected + 0.5) % 1.0 - 0.5
    largest = float(difference.abs().max()) if difference.numel() else 0.0
    if largest >= atol:
        raise AssertionError(
            f"{name}: largest difference in the cell {largest:.1e}, tolerance {atol:.0e}"
        )


def as_near_as_original(
    found: torch.Tensor, single: torch.Tensor, double: torch.Tensor
) -> bool:
    """Whether values are as near to the original in double precision as it is.

    For a network which amplifies the rounding of its inputs (MatterGen): the
    original in single precision is itself far from the original in double
    precision, and so is the rewrite, which rounds differently. The mean
    difference is at most 1.5 times the original's, the largest 3 times.

    Examples:
        >>> double = torch.tensor([1.0, 2.0], dtype=torch.float64)
        >>> as_near_as_original(double + 1e-3, double + 2e-3, double)
        True
    """
    ours, theirs = (found.double() - double).abs(), (single.double() - double).abs()

    return bool(ours.mean() < 1.5 * theirs.mean() and ours.max() < 3 * theirs.max())


def rounding_spread(
    evaluate: Callable[[Structures], tuple[torch.Tensor, ...]],
    structures: Structures,
    count: int = 6,
) -> list[torch.Tensor]:
    """How much the outputs of a network move when its inputs move by their last bits.

    The lattice and the coordinates are multiplied by 1 + k 2^-23, k drawn in
    -2 to 2 for each value, count times.

    Args:
        evaluate: Outputs of the network for structures without graph, a tuple
            of tensors.
        structures: Structures without graph.

    Returns:
        For each output, the largest move of each value over the draws.
    """
    random = torch.Generator().manual_seed(1)
    found = evaluate(structures)
    spread = [torch.zeros_like(values) for values in found]
    for _ in range(count):
        lattice = structures.lattice * (
            1
            + torch.randint(-2, 3, structures.lattice.shape, generator=random) * 2**-23
        )
        x = structures.x * (
            1 + torch.randint(-2, 3, structures.x.shape, generator=random) * 2**-23
        )
        moved = evaluate(replace(structures, lattice=lattice, x=x))
        spread = [
            torch.maximum(old, (new - values).abs())
            for old, new, values in zip(spread, moved, found)
        ]

    return spread


def within_rounding(
    found: torch.Tensor, expected: torch.Tensor, spread: torch.Tensor
) -> bool:
    """Whether values differ from the expected ones as little as rounding moves them.

    The largest and the mean difference are at most 3 times the largest and
    the mean move of rounding_spread.

    Examples:
        >>> within_rounding(torch.tensor([1.0]), torch.tensor([1.1]), torch.tensor([0.05]))
        True
    """
    difference = (found - expected).abs()

    return bool(
        difference.max() <= 3 * spread.max() and difference.mean() <= 3 * spread.mean()
    )
