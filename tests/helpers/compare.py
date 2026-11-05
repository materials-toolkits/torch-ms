"""Comparison of a result with what an original code computed."""

from __future__ import annotations

import torch


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
