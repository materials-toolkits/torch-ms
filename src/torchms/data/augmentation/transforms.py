"""Random augmentations of batched crystal structures.

Each transform draws one element of a group per structure and applies it. They
are callables from Structures to Structures, to give as the transform of a
dataset, alone or chained with Compose.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import torch

from torchms.data.augmentation import actions, sampling
from torchms.data.batching import Structures

Transform = Callable[[Structures], Structures]


@dataclass
class Compose:
    """Chain transforms, applied in order.

    Args:
        transforms: Transforms to apply one after the other.

    Examples:
        >>> augment = Compose(
        ...     [Supercell(max_repeat=2), LatticeBasisChange(), LatticeRotation()]
        ... )
        >>> augment(Structures.diamond()).lattice.shape
        torch.Size([1, 3, 3])
    """

    transforms: Sequence[Transform]

    def __call__(self, structures: Structures) -> Structures:
        for transform in self.transforms:
            structures = transform(structures)

        return structures


@dataclass
class AtomPermutation:
    """Shuffle the order of the atoms inside each structure.

    Args:
        generator: Random number generator, on the device of the structures.

    Examples:
        >>> structures = Structures.zincblende()
        >>> AtomPermutation()(structures).z.sort().values
        tensor([16, 16, 16, 16, 30, 30, 30, 30])
    """

    generator: Optional[torch.Generator] = None

    def __call__(self, structures: Structures) -> Structures:
        permutation = sampling.random_permutation(structures.batch, self.generator)

        return actions.permute_atoms(structures, permutation)


@dataclass
class LatticeRotation:
    """Apply a uniform random element of O(3), or of SO(3), to each lattice.

    Args:
        proper: If True, only rotations are drawn, in SO(3). Otherwise half of
            the draws are rotoreflections, which mirror a chiral crystal.
        generator: Random number generator, on the device of the structures.

    Examples:
        The lengths of the lattice vectors are preserved.

        >>> rotated = LatticeRotation()(Structures.diamond())
        >>> rotated.lattice.norm(dim=-1)
        tensor([[3.5670, 3.5670, 3.5670]])

        With rotations only, the orientation of the lattice is preserved too.

        >>> rotated = LatticeRotation(proper=True)(Structures.diamond(a=2.0))
        >>> torch.linalg.det(rotated.lattice).round()
        tensor([8.])
    """

    proper: bool = False
    generator: Optional[torch.Generator] = None

    def __call__(self, structures: Structures) -> Structures:
        rotation = sampling.random_orthogonal(
            len(structures.num_atoms),
            structures.device(),
            self.generator,
            self.proper,
        )

        return actions.rotate_lattice(structures, rotation)


@dataclass
class AtomTranslation:
    """Translate the atoms of each structure by a uniform random vector.

    Args:
        generator: Random number generator, on the device of the structures.

    Examples:
        >>> translated = AtomTranslation()(Structures.diamond())
        >>> bool(((translated.x >= 0) & (translated.x < 1)).all())
        True
    """

    generator: Optional[torch.Generator] = None

    def __call__(self, structures: Structures) -> Structures:
        translation = sampling.random_translation(
            len(structures.num_atoms), structures.device(), self.generator
        )

        return actions.translate_atoms(structures, translation)


@dataclass
class LatticeBasisChange:
    """Apply a random element of SL(3, Z) to each lattice.

    Args:
        num_shears: Number of random elementary matrices multiplied to get the
            change of basis. The cells get more skewed as it grows.
        generator: Random number generator, on the device of the structures.

    Examples:
        The volume of the cell is preserved.

        >>> sheared = LatticeBasisChange(num_shears=4)(Structures.diamond(a=2.0))
        >>> torch.linalg.det(sheared.lattice).abs().round()
        tensor([8.])
    """

    num_shears: int = 2
    generator: Optional[torch.Generator] = None

    def __call__(self, structures: Structures) -> Structures:
        basis = sampling.random_unimodular(
            len(structures.num_atoms),
            self.num_shears,
            structures.device(),
            self.generator,
        )

        return actions.change_lattice_basis(structures, basis)


@dataclass
class Supercell:
    """Repeat the cell of each structure a random number of times.

    Args:
        max_repeat: Maximum number of cells along a lattice vector, drawn
            uniformly from 1 to max_repeat for each lattice vector.
        max_atoms: Maximum number of atoms of a supercell. A structure whose
            supercell is larger is left unchanged.
        generator: Random number generator, on the device of the structures.

    Examples:
        >>> Supercell(max_repeat=3, max_atoms=8)(Structures.diamond()).num_atoms
        tensor([8])
    """

    max_repeat: int = 2
    max_atoms: Optional[int] = None
    generator: Optional[torch.Generator] = None

    def __call__(self, structures: Structures) -> Structures:
        repeat = sampling.random_repeat(
            structures.num_atoms, self.max_repeat, self.max_atoms, self.generator
        )

        return actions.make_supercell(structures, repeat)
