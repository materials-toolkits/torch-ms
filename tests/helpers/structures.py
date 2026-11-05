"""Structures and random numbers the tests of the models share."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from helpers.paths import CIF
from torchms.data import Structures, parse_cif


def generator(seed: int = 0) -> torch.Generator:
    """A generator of random numbers on the CPU, seeded.

    Examples:
        >>> torch.rand(2, generator=generator(3)).equal(torch.rand(2, generator=generator(3)))
        True
    """
    return torch.Generator().manual_seed(seed)


def make_structures(
    first: Callable[[], Structures] = Structures.zincblende,
    cif: str = "triclinic1.cif",
) -> list[Structures]:
    """Two crystals the tests of the models run on: a cubic one and a triclinic one.

    Args:
        first: The first crystal, zincblende by default.
        cif: The second one, a file of tests/cif.

    Examples:
        >>> [len(structures.z) for structures in make_structures()]
        [8, 40]
    """
    return [first(), parse_cif((CIF / cif).read_text())]


def make_batch(
    first: Callable[[], Structures] = Structures.zincblende,
    cif: str = "triclinic1.cif",
) -> Structures:
    """The structures of make_structures as a batch."""
    return Structures.collate(make_structures(first, cif))


def noisy(structures: Structures, seed: int = 0) -> Structures:
    """Structures with their atoms moved at random by 0.05 of the cell, in the cell.

    Examples:
        >>> moved = noisy(Structures.diamond())
        >>> bool((moved.x != Structures.diamond().x).any()), bool((moved.x < 1).all())
        (True, True)
    """
    noise = 0.05 * torch.randn(structures.x.shape, generator=generator(seed))

    return replace(structures.without_graphs(), x=(structures.x + noise) % 1.0)


def tilted(scale: float = 1.0) -> Structures:
    """A triclinic cell of three atoms: sodium, chlorine and oxygen.

    With a scale of 1 sodium and chlorine are close enough for the pair
    repulsion of orb-v3, not with a larger one.

    Examples:
        >>> tilted().z.tolist()
        [11, 17, 8]
    """
    return Structures(
        lattice=scale
        * torch.tensor([[[4.1, 0.0, 0.0], [0.6, 4.4, 0.0], [0.2, 0.5, 5.0]]]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.25, 0.7, 0.1]]),
        z=torch.tensor([11, 17, 8]),
    )
