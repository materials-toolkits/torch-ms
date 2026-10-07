"""Data augmentation of crystal structures.

A crystal has many descriptions: the order of its atoms, the orientation and
the origin of its cell, the basis of its lattice and the size of its cell are
arbitrary. Each augmentation moves from one description to another one of the
same crystal.
"""

from .actions import (
    change_lattice_basis,
    make_supercell,
    permute_atoms,
    rotate_lattice,
    translate_atoms,
)
from .sampling import (
    random_orthogonal,
    random_permutation,
    random_repeat,
    random_translation,
    random_unimodular,
)
from .transforms import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
    Supercell,
)

__all__ = [
    "AtomPermutation",
    "AtomTranslation",
    "Compose",
    "LatticeBasisChange",
    "LatticeRotation",
    "Supercell",
    "change_lattice_basis",
    "make_supercell",
    "permute_atoms",
    "rotate_lattice",
    "translate_atoms",
    "random_orthogonal",
    "random_permutation",
    "random_repeat",
    "random_translation",
    "random_unimodular",
]
