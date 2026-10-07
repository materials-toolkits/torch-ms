import math

import pytest
import torch

from pycrystalgen.data import Structures, parse_cif
from pycrystalgen.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
    random_orthogonal,
    rotate_lattice,
)
from pycrystalgen.graph.cutoff import cutoff, nearest_neighbors
from pycrystalgen.graph.geometry import dihedral_angles, triplet_cosines
from pycrystalgen.graph.line_graph import (
    line_graph,
    quadruplets,
    reverse_edges,
    triplets,
)
from pycrystalgen.graph.sparse import batched_join

RADIUS = 2.6


def make_structures() -> list[Structures]:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.diamond(), triclinic, Structures.zincblende()]


def make_batch() -> Structures:
    return Structures.collate(make_structures())


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def as_set(index: torch.Tensor) -> set[tuple[int, ...]]:
    """Columns of an index as a set, which fails on duplicates."""
    columns = [tuple(column) for column in index.t().tolist()]
    assert len(set(columns)) == len(columns)
    return set(columns)


def test_batched_join_against_loops():
    gen = generator()
    key_a = torch.randint(0, 7, (40,), generator=gen)
    key_b = torch.randint(0, 7, (25,), generator=gen)

    pairs, key, ptr = batched_join(key_a, key_b, num_keys=7)

    expected = {
        (a, b)
        for a in range(len(key_a))
        for b in range(len(key_b))
        if key_a[a] == key_b[b]
    }
    assert as_set(pairs.t()) == expected
    assert (key_a[pairs[:, 0]] == key).all()
    assert (key_b[pairs[:, 1]] == key).all()
    assert (key[:-1] <= key[1:]).all()
    assert (torch.bincount(key, minlength=7) == ptr[1:] - ptr[:-1]).all()


def test_batched_join_empty_keys():
    pairs, key, ptr = batched_join(
        torch.tensor([2, 2]), torch.tensor([0, 1]), num_keys=3
    )

    assert pairs.shape == (0, 2)
    assert key.shape == (0,)
    assert ptr.tolist() == [0, 0, 0, 0]


def test_edges_vectors():
    batch = make_batch()
    graph = cutoff(batch, RADIUS, self_images=True)
    vectors = graph.edges_vectors()

    assert vectors.shape == (graph.edge_index.shape[1], 3)
    assert torch.allclose(vectors.norm(dim=-1), graph.edges_distances())

    # the reverse edge has the opposite vector
    reverse = reverse_edges(graph)
    assert torch.allclose(vectors[reverse], -vectors, atol=1e-5)


def test_reverse_edges_of_a_cutoff_graph():
    graph = cutoff(make_batch(), RADIUS, self_images=True)
    reverse = reverse_edges(graph)

    assert (reverse >= 0).all()
    assert (reverse[reverse] == torch.arange(len(reverse))).all()
    assert (graph.edge_index[:, reverse] == graph.edge_index.flip(0)).all()
    assert (graph.edge_shift[reverse] == -graph.edge_shift).all()


def test_reverse_edges_missing():
    # the nearest neighbour of an atom does not always have it as nearest
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 10.0,
        x=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
        z=[1, 1, 1],
    )
    graph = nearest_neighbors(cutoff(structures, 4.0), 1)
    reverse = reverse_edges(graph)

    assert graph.edge_index.t().tolist() == [[0, 1], [1, 2], [2, 1]]
    assert reverse.tolist() == [-1, 2, 1]


def test_triplets_against_loops():
    batch = make_batch()
    graph = cutoff(batch, RADIUS, self_images=True)
    triplet_index, triplet_batch, triplet_ptr = triplets(graph)

    sender = graph.edge_index[0].tolist()
    expected = {
        (a, b)
        for a in range(len(sender))
        for b in range(len(sender))
        if a != b and sender[a] == sender[b]
    }
    assert len(expected) > 0
    assert as_set(triplet_index) == expected

    degree = torch.bincount(graph.edge_index[0], minlength=len(batch.z))
    assert triplet_index.shape[1] == int((degree * (degree - 1)).sum())

    # sorted by atom, hence by structure
    center = graph.edge_index[0, triplet_index[0]]
    assert (center[:-1] <= center[1:]).all()
    assert (triplet_batch == batch.batch[center]).all()
    assert (torch.bincount(triplet_batch, minlength=3) == triplet_ptr.diff()).all()


def test_triplets_with_a_shorter_cutoff():
    batch = make_batch()
    graph = cutoff(batch, 4.0, self_images=True)
    short = graph.edges_distances() <= RADIUS

    masked, _, masked_ptr = triplets(graph, short)
    assert short[masked].all()

    # same triplets as the graph of the shorter cutoff
    reference = cutoff(batch, RADIUS, self_images=True)
    _, _, reference_ptr = triplets(reference)
    assert (masked_ptr == reference_ptr).all()

    cosines = triplet_cosines(graph.edges_vectors(), masked).sort().values
    expected = (
        triplet_cosines(reference.edges_vectors(), triplets(reference)[0]).sort().values
    )
    assert torch.allclose(cosines, expected, atol=1e-4)


@pytest.mark.parametrize(
    "structure, radius",
    [(Structures.diamond(), 2.0), (Structures.zincblende(), 2.6)],
)
def test_tetrahedral_angle(structure, radius):
    graph = cutoff(structure, radius)
    triplet_index, _, triplet_ptr = triplets(graph)
    cosines = triplet_cosines(graph.edges_vectors(), triplet_index)

    assert triplet_ptr.tolist() == [0, 8 * 4 * 3]
    assert torch.allclose(cosines, torch.full_like(cosines, -1.0 / 3.0), atol=1e-5)


def test_line_graph_gives_the_triplets_of_a_symmetric_graph():
    graph = cutoff(make_batch(), RADIUS, self_images=True)
    reverse = reverse_edges(graph)
    pairs = line_graph(graph.edge_index, int(graph.num_atoms.sum()), reverse)

    # u→v then v→w is the triplet (v→u, v→w)
    assert (graph.edge_index[1, pairs[0]] == graph.edge_index[0, pairs[1]]).all()
    from_center = torch.stack([reverse[pairs[0]], pairs[1]])
    assert as_set(from_center) == as_set(triplets(graph)[0])


def test_quadruplets_against_loops():
    batch = make_batch()
    graph = cutoff(batch, 2.0, self_images=True)
    quadruplet_index, quadruplet_batch, quadruplet_ptr = quadruplets(graph)

    sender, receiver = graph.edge_index.tolist()
    reverse = reverse_edges(graph).tolist()
    n_edges = len(sender)
    by_sender: dict[int, list[int]] = {}
    for edge in range(n_edges):
        by_sender.setdefault(sender[edge], []).append(edge)

    expected = {
        (a, b, c)
        for a in range(n_edges)
        for b in by_sender.get(receiver[a], [])
        if b != reverse[a]
        for c in by_sender.get(receiver[b], [])
        if c != reverse[b]
    }
    assert len(expected) > 0
    assert as_set(quadruplet_index) == expected

    assert (quadruplet_batch == graph.edge_batch[quadruplet_index[0]]).all()
    assert (quadruplet_batch[:-1] <= quadruplet_batch[1:]).all()
    assert (
        torch.bincount(quadruplet_batch, minlength=3) == quadruplet_ptr.diff()
    ).all()


def test_dihedral_angles_of_diamond():
    graph = cutoff(Structures.diamond(), 2.0)
    quadruplet_index, _, _ = quadruplets(graph)
    angles = dihedral_angles(graph.edges_vectors(), quadruplet_index)

    # staggered bonds: 60°, -60° and 180°, each for a third of the chains
    degrees = torch.rad2deg(angles).abs().round()
    assert sorted(degrees.unique().tolist()) == [60.0, 180.0]
    assert int((degrees == 180.0).sum()) * 3 == len(degrees)
    gauche = angles[degrees == 60.0]
    assert int((gauche > 0).sum()) == int((gauche < 0).sum())


def test_dihedral_angles_convention():
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    cis = dihedral_angles(vectors, torch.tensor([[0], [1], [2]]))
    trans = dihedral_angles(vectors, torch.tensor([[0], [1], [3]]))

    assert torch.allclose(cis, torch.zeros(1), atol=1e-6)
    assert torch.allclose(trans.abs(), torch.full((1,), math.pi), atol=1e-6)


def test_dihedral_angles_change_sign_in_a_mirror():
    batch = make_batch()
    graph = cutoff(batch, 2.0, self_images=True)
    quadruplet_index, _, _ = quadruplets(graph)
    vectors = graph.edges_vectors()
    angles = dihedral_angles(vectors, quadruplet_index)

    mirror = vectors * torch.tensor([1.0, 1.0, -1.0])
    rotation = random_orthogonal(1, torch.device("cpu"), generator(), proper=True)[0]

    flat = angles.abs() > math.pi - 1e-3
    mirrored = dihedral_angles(mirror, quadruplet_index)
    rotated = dihedral_angles(vectors @ rotation.T, quadruplet_index)

    assert torch.allclose(mirrored[~flat], -angles[~flat], atol=1e-3)
    assert torch.allclose(rotated[~flat], angles[~flat], atol=1e-3)


Labelled = tuple[torch.Tensor, torch.Tensor]


def labelled(species: torch.Tensor, values: torch.Tensor) -> Labelled:
    """Species and values sorted by species, then by value.

    Args:
        species: Atomic numbers of the atoms of each tuple, shape (k, n).
        values: Value of each tuple, shape (n,).
    """
    # one number per tuple of species, in base 1000
    label = torch.zeros_like(species[0])
    for z in species:
        label = label * 1000 + z

    values, order = values.sort()
    label, by_label = label[order].sort(stable=True)

    return label, values[by_label]


def features(structures: Structures) -> list[tuple[Labelled, Labelled]]:
    """Angles and dihedral angles of each structure, with their species.

    An angle is labelled by the elements of its atom and of the two
    neighbours, a dihedral angle by the elements of the four atoms of its
    chain. The values are cosines, sorted for each label.
    """
    graph = cutoff(structures, RADIUS, self_images=True)
    vectors = graph.edges_vectors()
    sender, receiver = structures.z[graph.edge_index]

    # i→j and i→k: the atom i, then its neighbours j and k
    triplet_index, triplet_batch, _ = triplets(graph)
    first, second = triplet_index
    triplet_species = torch.stack([sender[first], receiver[first], receiver[second]])
    cosines = triplet_cosines(vectors, triplet_index)

    # first neighbours of diamond (1.54 Å) and of zincblende (2.34 Å)
    short = graph.edges_distances() <= 2.4
    quadruplet_index, quadruplet_batch, _ = quadruplets(graph, short)
    first, middle, last = quadruplet_index
    # u→v, v→w, w→x: the atoms u, v, w and x
    quadruplet_species = torch.stack(
        [sender[first], sender[middle], receiver[middle], receiver[last]]
    )
    dihedrals = dihedral_angles(vectors, quadruplet_index).cos()

    result = []
    for i in range(len(structures.num_atoms)):
        in_triplets, in_quadruplets = triplet_batch == i, quadruplet_batch == i
        result.append(
            (
                labelled(triplet_species[:, in_triplets], cosines[in_triplets]),
                labelled(
                    quadruplet_species[:, in_quadruplets], dihedrals[in_quadruplets]
                ),
            )
        )

    return result


def assert_same_labelled(a: Labelled, b: Labelled, n: int = 1, atol: float = 2e-3):
    """The tuples of b are the ones of a, each one n times: species and values."""
    (label_a, value_a), (label_b, value_b) = a, b

    assert len(label_a) > 0
    assert len(label_b) == n * len(label_a)
    assert (label_b.view(-1, n) == label_a.unsqueeze(1)).all()
    assert torch.allclose(value_b.view(-1, n), value_a.unsqueeze(1), atol=atol)


def assert_same_features(a: Structures, b: Structures, num_cells=None):
    if num_cells is None:
        num_cells = [1] * len(a.num_atoms)

    for (angles_a, dihedrals_a), (angles_b, dihedrals_b), n in zip(
        features(a), features(b), num_cells
    ):
        assert_same_labelled(angles_a, angles_b, n)
        assert_same_labelled(dihedrals_a, dihedrals_b, n)


def test_features_are_labelled_by_species():
    # same angles in ZnS and in CdS, between other elements
    zinc, cadmium = Structures.zincblende(), Structures.zincblende(cation=48)
    (angles_zn, dihedrals_zn), (angles_cd, dihedrals_cd) = (
        features(zinc)[0],
        features(cadmium)[0],
    )

    assert torch.allclose(angles_zn[1].sort().values, angles_cd[1].sort().values)
    assert torch.allclose(dihedrals_zn[1].sort().values, dihedrals_cd[1].sort().values)
    assert not torch.equal(angles_zn[0], angles_cd[0])
    assert not torch.equal(dihedrals_zn[0], dihedrals_cd[0])

    # an angle at a zinc atom between two sulfur atoms, and the reverse
    assert sorted(angles_zn[0].unique().tolist()) == [16_030_030, 30_016_016]
    # chains alternate zinc and sulfur
    assert sorted(dihedrals_zn[0].unique().tolist()) == [
        16_030_016_030,
        30_016_030_016,
    ]

    with pytest.raises(AssertionError):
        assert_same_labelled(angles_zn, angles_cd)


def test_same_alone_and_in_a_batch():
    alone = [features(structure)[0] for structure in make_structures()]

    for (angles_a, dihedrals_a), (angles_b, dihedrals_b) in zip(
        alone, features(make_batch())
    ):
        # same triplets and chains, up to the rounding of batched products
        assert_same_labelled(angles_a, angles_b, atol=1e-5)
        assert_same_labelled(dihedrals_a, dihedrals_b, atol=1e-5)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_unchanged_by_the_augmentations(transform):
    batch = make_batch()

    for seed in range(3):
        assert_same_features(batch, transform(generator=generator(seed))(batch))


def test_repeated_in_a_supercell():
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 1], [1, 2, 2]])

    assert_same_features(
        batch, make_supercell(batch, repeat), repeat.prod(dim=1).tolist()
    )


def test_rotation_of_the_lattice_rotates_the_vectors():
    batch = make_batch()
    rotation = random_orthogonal(3, torch.device("cpu"), generator())
    graph = cutoff(batch, RADIUS, self_images=True)
    rotated = cutoff(rotate_lattice(batch, rotation), RADIUS, self_images=True)

    assert (graph.edge_index == rotated.edge_index).all()
    expected = torch.einsum(
        "nij,nj->ni", rotation[graph.edge_batch], graph.edges_vectors()
    )
    assert torch.allclose(rotated.edges_vectors(), expected, atol=1e-4)
