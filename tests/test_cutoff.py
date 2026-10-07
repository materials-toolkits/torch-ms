import pytest
import torch

from torchms.data.batching import Structures
from torchms.graph.atomic_radius import atomic_radius


@pytest.fixture
def zincblende():
    return Structures.zincblende()


def test_atomic_radius_scaling_1_1_no_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=1.1, ensure_gap=0.0)
    assert graph.num_edges.tolist() == [32]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426]), atol=1e-3)


def test_atomic_radius_scaling_2_0_with_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=2.0, ensure_gap=0.5)
    assert graph.num_edges.tolist() == [32]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426]), atol=1e-3)


def test_atomic_radius_scaling_2_0_no_gap(zincblende):
    graph = atomic_radius(zincblende, scaling=2.0, ensure_gap=0.0)
    assert graph.num_edges.tolist() == [176]
    dists = graph.edges_distances().unique()
    assert torch.allclose(dists, torch.tensor([2.3426, 3.8254, 4.4857]), atol=1e-3)


# radius graph with periodic images of the atom itself, and nearest neighbours


def simple_cubic(a: float = 3.0, z: int = 29) -> Structures:
    return Structures(
        lattice=torch.eye(3).unsqueeze(0) * a, x=torch.zeros(1, 3), z=torch.tensor([z])
    )


def ase_edges(structures: Structures, radius: float):
    """Reference neighbour list of a single structure, as sorted (i, j, shift) rows."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=structures.z.numpy(),
        scaled_positions=structures.x.double().numpy(),
        cell=structures.lattice[0].double().numpy(),
        pbc=True,
    )
    i, j, shift = neighbor_list("ijS", atoms, radius)
    rows = torch.cat(
        [torch.tensor(i)[:, None], torch.tensor(j)[:, None], torch.tensor(shift)], 1
    )
    return sorted(map(tuple, rows.tolist()))


def graph_edges(graph):
    rows = torch.cat([graph.edge_index.t(), graph.edge_shift], dim=1)
    return sorted(map(tuple, rows.tolist()))


def test_cutoff_default_has_no_self_image():
    from torchms.graph.cutoff import cutoff

    assert cutoff(simple_cubic(), 3.5).num_edges.tolist() == [0]
    assert cutoff(Structures.zincblende(), 3.0).num_edges.tolist() == [32]


def test_cutoff_self_images_single_atom_cell():
    from torchms.graph.cutoff import cutoff

    graph = cutoff(simple_cubic(), 3.5, self_images=True)
    assert graph.num_edges.tolist() == [6]
    assert graph.edge_index.unique().tolist() == [0]
    assert sorted(graph.edge_shift.abs().sum(dim=1).tolist()) == [1] * 6
    assert torch.allclose(graph.edges_distances(), torch.full((6,), 3.0))

    # second shell at a * sqrt(2)
    assert cutoff(simple_cubic(), 4.5, self_images=True).num_edges.tolist() == [18]


@pytest.mark.parametrize("radius", [3.0, 6.0])
@pytest.mark.parametrize(
    "structures",
    [Structures.zincblende(), Structures.diamond(), simple_cubic(2.6)],
    ids=["zincblende", "diamond", "cubic"],
)
def test_cutoff_self_images_matches_ase(structures, radius):
    from torchms.graph.cutoff import cutoff

    graph = cutoff(structures, radius, self_images=True)

    assert graph_edges(graph) == ase_edges(structures, radius)


def test_cutoff_self_images_batch():
    from torchms.graph.cutoff import cutoff

    parts = [Structures.diamond(), simple_cubic(2.6), Structures.zincblende()]
    batch = Structures.collate(parts)

    graph = cutoff(batch, 5.0, self_images=True)
    expected = [cutoff(part, 5.0, self_images=True).num_edges.item() for part in parts]

    assert graph.num_edges.tolist() == expected
    assert (graph.edge_batch[:-1] <= graph.edge_batch[1:]).all()


def test_nearest_neighbors_keeps_the_closest():
    from torchms.graph.cutoff import cutoff, nearest_neighbors

    graph = cutoff(Structures.diamond(), 6.0, self_images=True)
    degree = torch.bincount(graph.edge_index[0], minlength=8)
    assert (degree > 16).all()

    # 4 first neighbours and 12 second neighbours: 16 is a complete shell
    capped = nearest_neighbors(graph, 16)
    distances = capped.edges_distances()

    assert torch.bincount(capped.edge_index[0], minlength=8).tolist() == [16] * 8
    assert capped.num_edges.tolist() == [128]
    assert torch.allclose(distances.max(), torch.tensor(3.567 / 2**0.5), atol=1e-3)
    assert set(graph_edges(capped)) <= set(graph_edges(graph))
    assert (capped.edge_batch == 0).all()


def test_nearest_neighbors_noop_and_batch():
    from torchms.graph.cutoff import cutoff, nearest_neighbors

    batch = Structures.collate(
        [Structures.diamond(), simple_cubic(2.6), Structures.zincblende()]
    )
    graph = cutoff(batch, 5.0, self_images=True)

    same = nearest_neighbors(graph, 10_000)
    assert graph_edges(same) == graph_edges(graph)
    assert same.num_edges.tolist() == graph.num_edges.tolist()

    capped = nearest_neighbors(graph, 4)
    assert torch.bincount(capped.edge_index[0], minlength=17).tolist() == [4] * 17
    assert capped.num_edges.tolist() == [32, 4, 32]
    assert (capped.edge_batch[:-1] <= capped.edge_batch[1:]).all()
    assert capped.edge_ptr.tolist() == [0, 32, 36, 68]
    # the 4 nearest of diamond and zincblende are the first shell
    per_structure = capped.edges_distances().split(capped.num_edges.tolist())
    assert torch.allclose(
        per_structure[0], torch.full((32,), 3.567 * 3**0.5 / 4), atol=1e-3
    )
    assert torch.allclose(
        per_structure[2], torch.full((32,), 5.41 * 3**0.5 / 4), atol=1e-3
    )


# anti-regression: distances in cells whose lattice matrix is not symmetric


def triclinic(name: str) -> Structures:
    from pathlib import Path

    from torchms.data import parse_cif

    return parse_cif((Path(__file__).parent / "cif" / f"{name}.cif").read_text())


def ase_distances(structures: Structures, radius: float) -> torch.Tensor:
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=structures.z.numpy(),
        scaled_positions=structures.x.double().numpy(),
        cell=structures.lattice[0].double().numpy(),
        pbc=True,
    )
    return torch.tensor(neighbor_list("d", atoms, radius)).sort().values


@pytest.mark.parametrize("name", ["triclinic1", "triclinic2", "triclinic3"])
def test_cutoff_triclinic_matches_ase(name):
    from torchms.graph.cutoff import cutoff

    structures = triclinic(name)
    lattice = structures.lattice[0]
    assert not torch.allclose(lattice, lattice.T), "the cell must not be symmetric"

    graph = cutoff(structures, 4.0, self_images=True)

    assert graph_edges(graph) == ase_edges(structures, 4.0)
    assert torch.allclose(
        graph.edges_distances().sort().values.double(),
        ase_distances(structures, 4.0),
        atol=1e-4,
    )


def test_atomic_radius_triclinic_distances_match_ase():
    structures = triclinic("triclinic1")

    graph = atomic_radius(structures, scaling=1.5, ensure_gap=0.0)
    distances = graph.edges_distances().sort().values.double()

    # every edge of the graph is a pair of atoms at this distance
    reference = ase_distances(structures, float(distances.max()) + 1e-3)
    assert len(distances) > 0
    position = torch.searchsorted(reference, distances).clamp(max=len(reference) - 1)
    nearest = torch.minimum(
        (reference[position] - distances).abs(),
        (reference[(position - 1).clamp(min=0)] - distances).abs(),
    )
    assert nearest.max() < 1e-4


# anti-regression: fractional coordinates outside [0, 1)


def unwrapped() -> Structures:
    """Rock salt like cell with atoms at 1.0 and outside the cell."""
    return Structures(
        lattice=torch.tensor([[[4.0, 0.0, 0.0], [0.5, 4.2, 0.0], [0.0, 0.3, 4.4]]]),
        x=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, -0.25],
                [1.5, 2.0, 0.1],
            ]
        ),
        z=torch.tensor([11, 17, 17, 11, 8]),
    )


@pytest.mark.parametrize("radius", [3.0, 5.0])
def test_cutoff_coordinates_outside_the_cell(radius):
    from torchms.graph.cutoff import cutoff

    structures = unwrapped()
    graph = cutoff(structures, radius, self_images=True)

    assert graph_edges(graph) == ase_edges(structures, radius)
    assert torch.allclose(
        graph.edges_distances().sort().values.double(),
        ase_distances(structures, radius),
        atol=1e-4,
    )


def test_atomic_radius_coordinates_outside_the_cell():
    structures = unwrapped()

    graph = atomic_radius(structures, scaling=1.2, ensure_gap=0.0)
    distances = graph.edges_distances()

    assert len(distances) > 0
    # an edge with a wrong shift is longer than the threshold of the graph
    assert distances.max() < 1.2 * 2 * 1.9
    assert set(graph_edges(graph)) <= set(
        ase_edges(structures, float(distances.max()) + 1e-3)
    )


# non orthogonal cells: complete set of edges, batches and nearest neighbours


def ase_neighbors(structures: Structures, radius: float):
    """Reference neighbours of a single structure: (i, j, shift, distance) rows."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=structures.z.numpy(),
        scaled_positions=structures.x.double().numpy(),
        cell=structures.lattice[0].double().numpy(),
        pbc=True,
    )
    i, j, shift, distance = neighbor_list("ijSd", atoms, radius)
    return i, j, shift, distance


@pytest.mark.parametrize("name", ["triclinic1", "triclinic2", "triclinic3"])
@pytest.mark.parametrize("scaling", [1.0, 1.6])
def test_atomic_radius_triclinic_edges_match_ase(name, scaling):
    from torchms.data.constants import get_element

    structures = triclinic(name)
    radii = torch.tensor(
        [get_element(int(z)).atomic_radius / 100.0 for z in structures.z]
    )

    graph = atomic_radius(structures, scaling=scaling, ensure_gap=0.0)

    # an atom is connected to the other atoms closer than the scaled sum of radii
    i, j, shift, distance = ase_neighbors(structures, 2 * scaling * float(radii.max()))
    threshold = scaling * (radii[i] + radii[j]).numpy()
    kept = (i != j) & (distance <= threshold)
    expected = sorted(
        (int(a), int(b), *map(int, s)) for a, b, s in zip(i[kept], j[kept], shift[kept])
    )

    # triclinic2 has no pair this close with the smallest scaling
    assert len(expected) > 0 or (name, scaling) == ("triclinic2", 1.0)
    assert graph_edges(graph) == expected
    assert torch.allclose(
        graph.edges_distances().sort().values.double(),
        torch.tensor(distance[kept]).sort().values,
        atol=1e-4,
    )


def test_cutoff_batch_of_non_orthogonal_cells():
    from torchms.graph.cutoff import cutoff

    parts = [
        triclinic("triclinic2"),
        Structures.diamond(),
        triclinic("triclinic1"),
        unwrapped(),
        triclinic("triclinic3"),
    ]
    batch = Structures.collate(parts)
    radius = 4.5

    graph = cutoff(batch, radius, self_images=True)

    assert graph.num_edges.tolist() == [len(ase_edges(part, radius)) for part in parts]
    assert (graph.edge_batch[:-1] <= graph.edge_batch[1:]).all()

    edges = torch.cat([graph.edge_index.t(), graph.edge_shift], dim=1)
    distances = graph.edges_distances().double()
    for k, part in enumerate(parts):
        rows = edges[graph.edge_batch == k].clone()
        rows[:, :2] -= batch.ptr[k]

        assert sorted(map(tuple, rows.tolist())) == ase_edges(part, radius)
        assert torch.allclose(
            distances[graph.edge_batch == k].sort().values,
            ase_distances(part, radius),
            atol=1e-4,
        )


@pytest.mark.parametrize("name", ["triclinic1", "triclinic3"])
@pytest.mark.parametrize("k", [1, 6, 20])
def test_nearest_neighbors_triclinic_matches_ase(name, k):
    from torchms.graph.cutoff import cutoff, nearest_neighbors

    structures = triclinic(name)
    radius = 6.0

    graph = nearest_neighbors(cutoff(structures, radius, self_images=True), k)
    distances = graph.edges_distances().double()

    i, _, _, reference = ase_neighbors(structures, radius)
    for atom in range(int(structures.num_atoms)):
        expected = torch.tensor(reference[i == atom]).sort().values[:k]
        found = distances[graph.edge_index[0] == atom].sort().values

        assert len(expected) == k
        assert torch.allclose(found, expected, atol=1e-4)

    # edges are sorted by source atom, then by length
    source = graph.edge_index[0]
    assert (source[:-1] <= source[1:]).all()
    same = source[:-1] == source[1:]
    assert (distances[:-1][same] <= distances[1:][same] + 1e-9).all()
