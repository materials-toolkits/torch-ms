import pytest
import torch

from pycrystalgen.data import Structures, parse_cif
from pycrystalgen.graph.atomic_radius import atomic_radius
from pycrystalgen.graph.coloring import color_refinement, refined_coloring
from pycrystalgen.graph.graph import Graph


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


def chain(z: list[int]) -> tuple[Structures, Graph]:
    """Atoms bonded in a chain, in the order of the list."""
    n = len(z)
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 100.0,
        x=torch.linspace(0.0, 0.5, n).unsqueeze(-1).repeat(1, 3),
        z=z,
    )
    forward = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
    edge_index = torch.cat([forward, forward.flip(0)], dim=1)
    graph = Graph(
        structures=structures,
        edge_index=edge_index,
        edge_shift=torch.zeros(edge_index.shape[1], 3, dtype=torch.int64),
        num_atoms=structures.num_atoms,
        num_edges=torch.tensor([edge_index.shape[1]]),
    )
    return structures, graph


def test_refinement_stops_on_a_chain():
    # with colors computed from the neighbours alone, the partition of this
    # chain alternates between two groupings and the refinement never stops
    structures, graph = chain([3, 3, 3, 3, 8])

    atom_colors = refined_coloring(structures, graph)

    # every atom is at a different distance of the oxygen
    assert atom_colors.unique(dim=0).shape[0] == 5
    assert color_refinement(structures, graph).shape == (1, 4)


def test_refinement_keeps_different_atoms_apart():
    # lithium and sodium have the same neighbour: they must keep their own color
    structures, graph = chain([3, 8, 11])

    atom_colors = refined_coloring(structures, graph)

    assert atom_colors.unique(dim=0).shape[0] == 3
    assert (atom_colors[0] != atom_colors[2]).any()


def test_refinement_only_splits_the_classes():
    # atoms with a different element or number of bonds never share a color
    structures = Structures.collate(make_structures())
    graph = atomic_radius(structures)

    degree = torch.bincount(graph.edge_index[0], minlength=len(structures.z))
    initial = torch.stack([structures.batch, structures.z, degree])
    final = torch.cat(
        [structures.batch.unsqueeze(-1), refined_coloring(structures, graph).long()],
        dim=1,
    )

    # each final class is inside one initial class
    joint = torch.cat([initial.t(), final], dim=1).unique(dim=0).shape[0]
    assert joint == final.unique(dim=0).shape[0]
    assert joint >= initial.unique(dim=1).shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_same_hashes_and_colors_on_cuda():
    structures = Structures.collate(make_structures())
    on_cuda = structures.to("cuda")

    atom_colors = refined_coloring(on_cuda, atomic_radius(on_cuda))
    structure_hashes = color_refinement(on_cuda, atomic_radius(on_cuda))

    assert atom_colors.device.type == "cuda" and structure_hashes.device.type == "cuda"
    assert (atom_colors.cpu() == colors(structures)).all()
    assert (structure_hashes.cpu() == hashes(structures)).all()
