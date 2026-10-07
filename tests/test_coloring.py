import torch

from pycrystalgen.data import Structures, parse_cif
from pycrystalgen.graph.atomic_radius import atomic_radius
from pycrystalgen.graph.coloring import color_refinement, refined_coloring


def make_structures() -> list[Structures]:
    """Structures whose refinement stops after a different number of rounds."""
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.diamond(), triclinic, Structures.zincblende()]


def hashes(structures: Structures) -> torch.Tensor:
    return color_refinement(structures, atomic_radius(structures))


def colors(structures: Structures) -> torch.Tensor:
    return refined_coloring(structures, atomic_radius(structures))


def test_hash_does_not_depend_on_the_batch():
    structures = make_structures()
    alone = torch.cat([hashes(structure) for structure in structures])

    assert (hashes(Structures.collate(structures)) == alone).all()


def test_colors_do_not_depend_on_the_batch():
    structures = make_structures()
    alone = torch.cat([colors(structure) for structure in structures])

    assert (colors(Structures.collate(structures)) == alone).all()


def test_hash_does_not_depend_on_the_order_of_the_batch():
    structures = make_structures()
    batch = hashes(Structures.collate(structures))
    reversed_batch = hashes(Structures.collate(structures[::-1]))

    assert (reversed_batch == batch.flip(0)).all()
    # the structures are different
    assert batch.unique(dim=0).shape[0] == len(structures)
