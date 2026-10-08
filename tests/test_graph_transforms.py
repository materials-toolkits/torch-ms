"""Graph construction as transforms, and batching of what they attach."""

import math

import pytest
import torch
from torch.utils.data import default_collate

from torchms.data import Structures
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
    Supercell,
)
from torchms.data.batching import TUPLES_FIELDS, InvalidStructure
from torchms.graph.atomic_radius import atomic_radius
from torchms.graph.cutoff import cutoff, nearest_neighbors
from torchms.graph.geometry import dihedral_angles, triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph, InvalidGraph
from torchms.graph.line_graph import edge_pairs, quadruplets, triplets
from torchms.graph.transforms import (
    AddAtomicRadiusGraph,
    AddLineGraph,
    AddQuadruplets,
    AddRadiusGraph,
    AddTriplets,
    KeepNearestNeighbors,
    KeepRadius,
)
from torchms.models import LineGraphNetwork


def random_structure(generator: torch.Generator, n_atoms: int) -> Structures:
    """A triclinic cell with atoms at random positions."""
    lattice = 4.0 * torch.eye(3) + torch.rand(3, 3, generator=generator)

    return Structures(
        lattice=lattice.unsqueeze(0),
        x=torch.rand(n_atoms, 3, generator=generator),
        z=torch.randint(1, 30, (n_atoms,), generator=generator),
        total_energy=[float(n_atoms)],
    )


@pytest.fixture
def structures() -> list[Structures]:
    """Structures of different sizes, one of them smaller than the radius."""
    generator = torch.Generator().manual_seed(0)
    small = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 2.5, x=torch.zeros(1, 3), z=[29]
    )

    return [
        Structures.diamond(),
        random_structure(generator, 5),
        small,
        Structures.zincblende(),
        random_structure(generator, 3),
    ]


CHAINS = {
    "radius": [AddRadiusGraph(3.0)],
    "radius-geometry": [AddRadiusGraph(3.0, self_images=True, geometry=True)],
    "atomic-radius": [AddAtomicRadiusGraph(1.1, geometry=True)],
    "nearest": [AddRadiusGraph(4.0, geometry=True), KeepNearestNeighbors(3)],
    "keep-radius": [AddRadiusGraph(4.0), KeepRadius(2.8)],
    "line-graph": [AddRadiusGraph(3.0), AddLineGraph(geometry=True)],
    "line-graph-backtrack": [AddRadiusGraph(3.0), AddLineGraph(backtrack=True)],
    "triplets": [AddRadiusGraph(3.5, self_images=True), AddTriplets(2.8, True)],
    "quadruplets": [AddRadiusGraph(3.0), AddQuadruplets(2.6, geometry=True)],
    "all": [
        AddRadiusGraph(3.5, self_images=True, geometry=True),
        AddLineGraph(3.0, geometry=True),
        AddTriplets(3.0, geometry=True),
        AddQuadruplets(2.6, geometry=True),
    ],
}


@pytest.fixture(params=sorted(CHAINS))
def chain(request) -> Compose:
    return Compose(CHAINS[request.param])


def sorted_columns(columns: torch.Tensor) -> torch.Tensor:
    """Columns in lexicographic order, to compare sets of edges or tuples."""
    if columns.shape[1] == 0:
        return columns

    return torch.unique(columns, dim=1)


def assert_same_tuples(a: EdgeTuples | None, b: EdgeTuples | None, edges_a, edges_b):
    """Same tuples of edges, the edges being described by their atoms and image."""
    assert (a is None) == (b is None)
    if a is None or b is None:
        return

    assert torch.equal(a.num_edges, b.num_edges)
    assert torch.equal(a.num_tuples, b.num_tuples)
    assert torch.equal(a.ptr, b.ptr)
    assert torch.equal(a.batch, b.batch)
    assert (a.angle is None) == (b.angle is None)

    # a tuple is described by the atoms and the images of its edges
    described_a = edges_a[:, a.index].flatten(0, 1).double()
    described_b = edges_b[:, b.index].flatten(0, 1).double()
    if a.angle is not None and b.angle is not None:
        described_a = torch.cat([described_a, a.angle.double().round(decimals=3)[None]])
        described_b = torch.cat([described_b, b.angle.double().round(decimals=3)[None]])

    assert torch.equal(sorted_columns(described_a), sorted_columns(described_b))


def assert_same_attached(a: Structures, b: Structures):
    """Same graph and tuples of edges, whatever the order inside a structure."""
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.ptr, b.ptr)
    assert (a.graph is None) == (b.graph is None)
    if a.graph is None or b.graph is None:
        return

    for name in ("num_atoms", "num_edges", "edge_ptr", "edge_batch"):
        assert torch.equal(getattr(a.graph, name), getattr(b.graph, name)), name

    edges_a = torch.cat([a.graph.edge_index, a.graph.edge_shift.t()])
    edges_b = torch.cat([b.graph.edge_index, b.graph.edge_shift.t()])
    assert torch.equal(sorted_columns(edges_a), sorted_columns(edges_b))

    assert (a.graph.edge_distance is None) == (b.graph.edge_distance is None)
    assert (a.graph.edge_vector is None) == (b.graph.edge_vector is None)

    for name in TUPLES_FIELDS:
        assert_same_tuples(getattr(a, name), getattr(b, name), edges_a, edges_b)


def assert_consistent(structures: Structures):
    """Indices point to the right atoms and edges: geometry is recomputed from them."""
    graph = structures.graph
    assert graph is not None
    assert graph.structures is not None
    assert graph.structures.graph is None
    assert graph.structures.x is structures.x

    n_atoms, n_edges = len(structures.z), graph.edge_index.shape[1]
    assert torch.equal(graph.num_atoms, structures.num_atoms)
    assert n_edges == int(graph.num_edges.sum())
    if n_edges > 0:
        assert int(graph.edge_index.min()) >= 0
        assert int(graph.edge_index.max()) < n_atoms
    # both atoms of an edge are in the structure of the edge
    assert (structures.batch[graph.edge_index] == graph.edge_batch).all()
    assert torch.equal(graph.edge_ptr[1:] - graph.edge_ptr[:-1], graph.num_edges)

    vectors = graph.edges_vectors()
    if graph.edge_vector is not None:
        assert graph.edge_distance is not None
        assert torch.allclose(graph.edge_vector, vectors, atol=1e-5)
        assert torch.allclose(graph.edge_distance, vectors.norm(dim=-1), atol=1e-5)

    for name in TUPLES_FIELDS:
        tuples: EdgeTuples | None = getattr(structures, name)
        if tuples is None:
            continue

        assert torch.equal(tuples.num_edges, graph.num_edges)
        assert tuples.index.shape[1] == int(tuples.num_tuples.sum())
        assert torch.equal(tuples.ptr[1:] - tuples.ptr[:-1], tuples.num_tuples)
        # the edges of a tuple are in the structure of the tuple
        assert (graph.edge_batch[tuples.index] == tuples.batch).all()

        sender, receiver = graph.edge_index[:, tuples.index]
        if name == "triplets":
            assert (sender[0] == sender[1]).all()
            expected = torch.acos(triplet_cosines(vectors, tuples.index))
        elif name == "line_graph":
            assert (receiver[0] == sender[1]).all()
            expected = torch.acos(-triplet_cosines(vectors, tuples.index))
        else:
            assert (receiver[0] == sender[1]).all()
            assert (receiver[1] == sender[2]).all()
            expected = dihedral_angles(vectors, tuples.index)

        if tuples.angle is not None:
            assert torch.allclose(tuples.angle, expected, atol=1e-4)


class TestTransforms:
    def test_radius_graph_is_cutoff(self, structures):
        batch = Structures.collate(structures)
        result = AddRadiusGraph(3.0, self_images=True)(batch)
        expected = cutoff(batch, 3.0, self_images=True)

        assert result is not batch and batch.graph is None
        assert result.graph is not None
        assert torch.equal(result.graph.edge_index, expected.edge_index)
        assert torch.equal(result.graph.edge_shift, expected.edge_shift)
        assert result.graph.edge_vector is None
        assert result.graph.edge_distance is None

    def test_atomic_radius_graph(self):
        batch = Structures.collate([Structures.zincblende(), Structures.diamond()])
        result = AddAtomicRadiusGraph(1.1, ensure_gap=0.0, geometry=True)(batch)
        expected = atomic_radius(batch, 1.1, ensure_gap=0.0)

        assert result.graph is not None and result.graph.edge_distance is not None
        assert torch.equal(result.graph.edge_index, expected.edge_index)
        assert torch.allclose(result.graph.edge_distance, expected.edges_distances())

    def test_geometry_of_the_edges(self, structures):
        batch = AddRadiusGraph(3.0, geometry=True)(Structures.collate(structures))
        graph = batch.graph

        assert graph is not None and graph.edge_vector is not None
        assert graph.edge_vector.shape == (graph.edge_index.shape[1], 3)
        assert torch.allclose(graph.edge_vector, graph.edges_vectors())
        assert torch.allclose(graph.edge_distance, graph.edges_distances())
        assert float(graph.edge_distance.max()) <= 3.0

    def test_keep_nearest_neighbors(self, structures):
        batch = AddRadiusGraph(4.0, geometry=True)(Structures.collate(structures))
        result = KeepNearestNeighbors(3)(batch)
        expected = nearest_neighbors(cutoff(Structures.collate(structures), 4.0), 3)

        assert result.graph is not None and result.graph.edge_distance is not None
        assert torch.equal(result.graph.edge_index, expected.edge_index)
        assert torch.allclose(result.graph.edge_distance, expected.edges_distances())
        assert int(torch.bincount(result.graph.edge_index[0]).max()) == 3

    def test_keep_radius(self, structures):
        batch = Structures.collate(structures)
        result = KeepRadius(2.8)(AddRadiusGraph(4.0, geometry=True)(batch))

        assert_same_attached(result, AddRadiusGraph(2.8, geometry=True)(batch))
        assert_consistent(result)

    @pytest.mark.parametrize("keep", [KeepNearestNeighbors(2), KeepRadius(2.0)])
    def test_keep_removes_the_tuples(self, keep):
        chain = Compose(
            [AddRadiusGraph(3.0), AddLineGraph(), AddTriplets(), AddQuadruplets(2.0)]
        )
        result = keep(chain(Structures.diamond()))

        assert result.graph is not None
        assert all(getattr(result, name) is None for name in TUPLES_FIELDS)

    def test_new_graph_removes_the_tuples(self):
        batch = AddTriplets()(AddRadiusGraph(2.0)(Structures.diamond()))
        result = AddRadiusGraph(3.0)(batch)

        assert batch.triplets is not None
        assert result.triplets is None

    def test_line_graph(self, structures):
        batch = AddRadiusGraph(3.0)(Structures.collate(structures))
        graph = batch.graph
        assert graph is not None

        result = AddLineGraph()(batch).line_graph
        index, pair_batch, ptr = edge_pairs(graph)
        assert result is not None and result.angle is None
        assert torch.equal(result.index, index)
        assert torch.equal(result.batch, pair_batch)
        assert torch.equal(result.ptr, ptr)

        # with the pairs going back: every pair of an edge arriving and leaving
        backtrack = AddLineGraph(backtrack=True)(batch).line_graph
        assert backtrack is not None
        degree = torch.bincount(graph.edge_index[0], minlength=len(batch.z))
        assert backtrack.index.shape[1] == int((degree * degree).sum())
        assert backtrack.index.shape[1] - index.shape[1] == graph.edge_index.shape[1]

    def test_triplets_and_quadruplets(self, structures):
        batch = AddRadiusGraph(3.0)(Structures.collate(structures))
        graph = batch.graph
        assert graph is not None
        mask = graph.edges_distances() <= 2.6

        result = AddQuadruplets(2.6)(AddTriplets(2.6)(batch))
        assert result.triplets is not None and result.quadruplets is not None
        assert torch.equal(result.triplets.index, triplets(graph, mask)[0])
        assert torch.equal(result.quadruplets.index, quadruplets(graph, mask)[0])
        assert result.graph is graph

    def test_radius_masks_the_edges_of_the_tuples(self, structures):
        batch = AddRadiusGraph(3.5, geometry=True)(Structures.collate(structures))
        result = AddQuadruplets(2.6)(AddTriplets(3.0)(AddLineGraph(2.8)(batch)))
        distance = result.graph.edge_distance

        assert float(distance.max()) > 3.0
        assert float(distance[result.line_graph.index].max()) <= 2.8
        assert float(distance[result.triplets.index].max()) <= 3.0
        assert float(distance[result.quadruplets.index].max()) <= 2.6

    def test_angles_of_diamond(self):
        chain = Compose(
            [
                AddRadiusGraph(2.0),
                AddLineGraph(geometry=True),
                AddTriplets(geometry=True),
                AddQuadruplets(geometry=True),
            ]
        )
        result = chain(Structures.diamond())
        tetrahedral = math.acos(-1.0 / 3.0)

        assert torch.allclose(
            result.triplets.angle, torch.full((96,), tetrahedral), atol=1e-4
        )
        assert torch.allclose(
            result.line_graph.angle, torch.full((96,), tetrahedral), atol=1e-4
        )
        # staggered bonds: dihedral angles of 60°, -60° and 180°
        dihedral = result.quadruplets.angle.abs().rad2deg().round().unique()
        assert dihedral.tolist() == [60.0, 180.0]

    def test_same_as_line_graph_network(self):
        batch = Structures.collate([Structures.zincblende(), Structures.diamond()])
        model = LineGraphNetwork(order=3, radius=(4.0, 3.0, 2.5))
        index, geometry, _ = model.build_graph(batch)

        chain = Compose(
            [
                AddRadiusGraph(4.0, self_images=True, geometry=True),
                AddTriplets(3.0, geometry=True),
                AddQuadruplets(2.5, geometry=True),
            ]
        )
        result = chain(batch)

        assert torch.equal(result.graph.edge_index, index["bonds"])
        assert torch.allclose(result.graph.edge_distance, geometry["bonds"])
        assert torch.equal(result.triplets.index, index["triplets"])
        # the model keeps the cosines away from ±1, for its gradient
        assert torch.allclose(result.triplets.angle, geometry["triplets"], atol=5e-3)
        assert torch.equal(result.quadruplets.index, index["quadruplets"])
        assert torch.allclose(result.quadruplets.angle, geometry["quadruplets"])

    @pytest.mark.parametrize(
        "transform",
        [
            KeepNearestNeighbors(2),
            KeepRadius(2.0),
            AddLineGraph(),
            AddTriplets(),
            AddQuadruplets(),
        ],
    )
    def test_needs_a_graph(self, transform):
        with pytest.raises(ValueError, match="needs a graph"):
            transform(Structures.diamond())

    def test_after_augmentations(self):
        chain = Compose(
            [
                Supercell(max_repeat=2),
                LatticeBasisChange(),
                LatticeRotation(),
                AtomTranslation(),
                AtomPermutation(),
                AddRadiusGraph(2.0, geometry=True),
                AddTriplets(geometry=True),
            ]
        )
        result = chain(Structures.collate([Structures.diamond()] * 3))

        assert_consistent(result)
        # the bonds of diamond, whatever its description
        assert torch.allclose(
            result.graph.edge_distance,
            torch.full_like(result.graph.edge_distance, 1.5446),
            atol=1e-3,
        )
        assert torch.equal(result.graph.num_edges, 4 * result.num_atoms)

    @pytest.mark.parametrize(
        "augmentation",
        [
            Supercell(max_repeat=2),
            LatticeBasisChange(),
            LatticeRotation(),
            AtomTranslation(),
            AtomPermutation(),
        ],
    )
    def test_augmentation_removes_what_is_attached(self, augmentation):
        batch = AddTriplets()(AddRadiusGraph(2.0)(Structures.diamond()))
        result = augmentation(batch)

        assert batch.graph is not None and batch.triplets is not None
        assert result.graph is None
        assert all(getattr(result, name) is None for name in TUPLES_FIELDS)


class TestBatching:
    """The batch of the graphs is the graph of the batch."""

    def test_chain_of_batch_is_batch_of_chains(self, structures, chain):
        of_batch = chain(Structures.collate(structures))
        of_each = Structures.collate([chain(s) for s in structures])

        assert_consistent(of_batch)
        assert_consistent(of_each)
        assert_same_attached(of_each, of_batch)

    def test_batches_of_batches(self, structures, chain):
        of_batch = chain(Structures.collate(structures))
        of_parts = Structures.collate(
            [
                chain(Structures.collate(structures[:2])),
                chain(structures[2]),
                chain(Structures.collate(structures[3:])),
            ]
        )

        assert_consistent(of_parts)
        assert_same_attached(of_parts, of_batch)

    def test_separate_round_trip(self, structures, chain):
        batch = chain(Structures.collate(structures))
        separated = Structures.separate(batch)

        assert len(separated) == len(structures)
        for single, original in zip(separated, structures):
            assert_consistent(single)
            assert_same_attached(single, chain(original))

        again = Structures.collate(separated)
        assert_consistent(again)
        assert_same_attached(again, batch)
        assert torch.equal(again.graph.edge_index, batch.graph.edge_index)
        for name in TUPLES_FIELDS:
            if getattr(batch, name) is not None:
                assert torch.equal(
                    getattr(again, name).index, getattr(batch, name).index
                )

    def test_offsets(self):
        chain = Compose([AddRadiusGraph(2.0), AddTriplets()])
        diamond = chain(Structures.diamond())
        batch = Structures.collate([diamond, diamond, diamond])

        n_atoms, n_edges, n_triplets = 8, 32, 96
        for i in range(3):
            edges = slice(i * n_edges, (i + 1) * n_edges)
            assert torch.equal(
                batch.graph.edge_index[:, edges],
                diamond.graph.edge_index + i * n_atoms,
            )
            found = slice(i * n_triplets, (i + 1) * n_triplets)
            assert torch.equal(
                batch.triplets.index[:, found], diamond.triplets.index + i * n_edges
            )

        assert batch.graph.edge_ptr.tolist() == [0, 32, 64, 96]
        assert batch.triplets.ptr.tolist() == [0, 96, 192, 288]
        assert batch.triplets.num_edges.tolist() == [32, 32, 32]

    def test_structure_without_edge(self):
        chain = Compose([AddRadiusGraph(2.0), AddTriplets(geometry=True)])
        lonely = Structures(
            lattice=torch.eye(3).unsqueeze(0) * 9.0, x=torch.zeros(1, 3), z=[2]
        )
        batch = Structures.collate(
            [chain(Structures.diamond()), chain(lonely), chain(Structures.diamond())]
        )

        assert batch.graph.num_edges.tolist() == [32, 0, 32]
        assert batch.triplets.num_tuples.tolist() == [96, 0, 96]
        assert_consistent(batch)
        assert_same_attached(
            batch,
            chain(
                Structures.collate([Structures.diamond(), lonely, Structures.diamond()])
            ),
        )

    def test_default_collate(self, structures, chain):
        batch = default_collate([chain(s) for s in structures])

        assert isinstance(batch, Structures)
        assert_same_attached(batch, chain(Structures.collate(structures)))

    def test_default_collate_without_graph(self, structures):
        batch = default_collate(structures)
        expected = Structures.collate(structures)

        assert isinstance(batch, Structures)
        assert torch.equal(batch.x, expected.x)
        assert torch.equal(batch.ptr, expected.ptr)
        assert batch.graph is None

    def test_mixed_graph_raises(self):
        with_graph = AddRadiusGraph(2.0)(Structures.diamond())

        with pytest.raises(ValueError, match='"graph"'):
            Structures.collate([with_graph, Structures.diamond()])

    def test_mixed_tuples_raise(self):
        with_graph = AddRadiusGraph(2.0)(Structures.diamond())

        with pytest.raises(ValueError, match='"triplets"'):
            Structures.collate([AddTriplets()(with_graph), with_graph])

    def test_mixed_geometry_raises(self):
        diamond = Structures.diamond()

        with pytest.raises(ValueError, match="edge_vector"):
            Structures.collate(
                [
                    AddRadiusGraph(2.0, geometry=True)(diamond),
                    AddRadiusGraph(2.0)(diamond),
                ]
            )

        graph = AddRadiusGraph(2.0)(diamond)
        with pytest.raises(ValueError, match="angle"):
            Structures.collate(
                [AddTriplets(geometry=True)(graph), AddTriplets()(graph)]
            )

    def test_to(self, structures, chain):
        batch = chain(Structures.collate(structures))
        moved = batch.to("cpu")

        assert moved is not batch
        assert_consistent(moved)
        assert_same_attached(moved, batch)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_to_cuda(self, structures):
        chain = Compose(CHAINS["all"])
        batch = chain(Structures.collate(structures))
        moved = batch.to("cuda")

        assert moved.graph.device().type == "cuda"
        assert moved.graph.edge_vector.device.type == "cuda"
        assert moved.graph.structures.x.device.type == "cuda"
        assert moved.graph.structures.x is moved.x
        for name in TUPLES_FIELDS:
            assert getattr(moved, name).device().type == "cuda"
            assert getattr(moved, name).angle.device.type == "cuda"

        assert batch.graph.device().type == "cpu"
        assert_consistent(moved)
        assert_same_attached(moved.to("cpu"), batch)


class TestValidation:
    def test_graph_of_other_structures(self):
        graph = cutoff(Structures.diamond(), 2.0)
        zincblende = Structures.collate([Structures.zincblende()] * 2)

        with pytest.raises(InvalidStructure, match="atoms of the structures"):
            Structures(
                lattice=zincblende.lattice,
                x=zincblende.x,
                z=zincblende.z,
                num_atoms=zincblende.num_atoms,
                graph=graph,
            )

    def test_tuples_without_graph(self):
        diamond = Structures.diamond()
        tuples = AddTriplets()(AddRadiusGraph(2.0)(diamond)).triplets

        with pytest.raises(InvalidStructure, match="needs a graph"):
            Structures(
                lattice=diamond.lattice, x=diamond.x, z=diamond.z, triplets=tuples
            )

    def test_tuples_of_another_graph(self):
        diamond = Structures.diamond()
        tuples = AddTriplets()(AddRadiusGraph(2.0)(diamond)).triplets
        graph = cutoff(diamond, 3.0)

        with pytest.raises(InvalidStructure, match="edges of the graph"):
            Structures(
                lattice=diamond.lattice,
                x=diamond.x,
                z=diamond.z,
                graph=graph,
                triplets=tuples,
            )

    def test_tuples_out_of_their_structure(self):
        with pytest.raises(InvalidGraph, match="not in its structure"):
            EdgeTuples(index=[[0, 3], [1, 2]], num_edges=[3, 3], num_tuples=[2, 0])

    def test_tuples_shape(self):
        with pytest.raises(InvalidGraph, match="index"):
            EdgeTuples(index=[[0, 0], [1, 2]], num_edges=[3], num_tuples=[3])

    def test_geometry_shape(self):
        graph = cutoff(Structures.diamond(), 2.0)

        with pytest.raises(InvalidGraph, match="edge_distance"):
            Graph(
                edge_index=graph.edge_index,
                edge_shift=graph.edge_shift,
                num_atoms=graph.num_atoms,
                num_edges=graph.num_edges,
                edge_distance=torch.zeros(3),
            )


class TestGraphAlone:
    """Graph.collate and Graph.separate keep the structures and the geometry."""

    def test_collate_keeps_structures(self, structures):
        graphs = [cutoff(s, 3.0).with_geometry() for s in structures]
        batched = Graph.collate(graphs)
        expected = cutoff(Structures.collate(structures), 3.0)

        assert batched.structures is not None
        assert torch.equal(batched.structures.x, Structures.collate(structures).x)
        assert torch.allclose(
            batched.edges_distances().sort().values,
            expected.edges_distances().sort().values,
            atol=1e-5,
        )
        assert torch.allclose(
            batched.edge_distance, batched.edges_distances(), atol=1e-5
        )

    def test_collate_without_structures(self):
        graph = cutoff(Structures.diamond(), 2.0)
        bare = Graph(
            edge_index=graph.edge_index,
            edge_shift=graph.edge_shift,
            num_atoms=graph.num_atoms,
            num_edges=graph.num_edges,
        )

        assert Graph.collate([graph, bare]).structures is None

    def test_separate_keeps_structures(self, structures):
        batched = cutoff(Structures.collate(structures), 3.0).with_geometry()
        separated = Graph.separate(batched)

        for graph, original in zip(separated, structures):
            assert graph.structures is not None
            assert torch.equal(graph.structures.x, original.x)
            assert torch.allclose(
                graph.edge_distance, graph.edges_distances(), atol=1e-5
            )

    def test_to(self):
        graph = cutoff(Structures.diamond(), 2.0).with_geometry().to("cpu")

        assert graph.device().type == "cpu"
        assert graph.structures is not None
        assert graph.edge_vector is not None


class TestSplit:
    """A batch split into smaller batches, with what is attached."""

    @staticmethod
    def assert_equal(a: Structures, b: Structures):
        """Same tensors, for the structures and for what is attached."""
        for name in ("lattice", "x", "z", "num_atoms", "batch", "ptr"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        # unknown labels are NaN
        assert torch.allclose(a.total_energy, b.total_energy, equal_nan=True)

        assert (a.graph is None) == (b.graph is None)
        if a.graph is not None and b.graph is not None:
            for name in ("edge_index", "edge_shift", "num_atoms", "num_edges"):
                assert torch.equal(getattr(a.graph, name), getattr(b.graph, name)), name
            for name in ("edge_batch", "edge_ptr"):
                assert torch.equal(getattr(a.graph, name), getattr(b.graph, name)), name
            for name in ("edge_vector", "edge_distance"):
                first, second = getattr(a.graph, name), getattr(b.graph, name)
                assert (first is None) == (second is None), name
                if first is not None:
                    assert torch.allclose(first, second, atol=1e-5), name

        for field in TUPLES_FIELDS:
            first, second = getattr(a, field), getattr(b, field)
            assert (first is None) == (second is None), field
            if first is None:
                continue
            for name in ("index", "num_edges", "num_tuples", "batch", "ptr"):
                assert torch.equal(getattr(first, name), getattr(second, name)), name
            assert (first.angle is None) == (second.angle is None)
            if first.angle is not None:
                assert torch.allclose(first.angle, second.angle, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
    def test_parts_are_the_graphs_of_the_small_batches(
        self, structures, chain, batch_size
    ):
        parts = chain(Structures.collate(structures)).split(batch_size)

        sizes = [len(part.num_atoms) for part in parts]
        assert sum(sizes) == len(structures)
        assert all(size == batch_size for size in sizes[:-1])
        assert 0 < sizes[-1] <= batch_size

        for k, part in enumerate(parts):
            expected = chain(
                Structures.collate(structures[k * batch_size : (k + 1) * batch_size])
            )
            assert_consistent(part)
            self.assert_equal(part, expected)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_collate_of_the_parts_is_the_batch(self, structures, chain, batch_size):
        batch = chain(Structures.collate(structures))
        again = Structures.collate(batch.split(batch_size))

        assert_consistent(again)
        self.assert_equal(again, batch)

    def test_without_graph(self, structures):
        batch = Structures.collate(structures)
        parts = batch.split(2)

        assert [len(part.num_atoms) for part in parts] == [2, 2, 1]
        assert all(part.graph is None for part in parts)
        for k, part in enumerate(parts):
            self.assert_equal(part, Structures.collate(structures[2 * k : 2 * k + 2]))

    def test_split_in_one_is_separate(self, structures, chain):
        batch = chain(Structures.collate(structures))

        for part, single in zip(batch.split(1), Structures.separate(batch)):
            self.assert_equal(part, single)

    def test_size_larger_than_the_batch(self, structures, chain):
        batch = chain(Structures.collate(structures))
        (part,) = batch.split(100)

        self.assert_equal(part, batch)

    def test_structure_without_edge_between_batches(self):
        chain = Compose([AddRadiusGraph(2.0), AddTriplets(geometry=True)])
        lonely = Structures(
            lattice=torch.eye(3).unsqueeze(0) * 9.0, x=torch.zeros(1, 3), z=[2]
        )
        structures = [Structures.diamond(), lonely, lonely, Structures.diamond()]
        parts = chain(Structures.collate(structures)).split(2)

        assert [part.graph.num_edges.tolist() for part in parts] == [[32, 0], [0, 32]]
        assert [part.triplets.num_tuples.tolist() for part in parts] == [
            [96, 0],
            [0, 96],
        ]
        for part in parts:
            assert_consistent(part)

    def test_parts_refer_to_their_own_structures(self, structures):
        batch = AddRadiusGraph(3.0, geometry=True)(Structures.collate(structures))

        for part in batch.split(2):
            assert part.graph.structures is not None
            assert part.graph.structures.graph is None
            assert torch.equal(part.graph.structures.x, part.x)
            assert torch.allclose(
                part.graph.edges_distances(), part.graph.edge_distance, atol=1e-5
            )

    def test_invalid_size(self, structures):
        with pytest.raises(ValueError, match="positive"):
            Structures.collate(structures).split(0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_on_gpu(self, structures):
        chain = Compose(CHAINS["all"])
        batch = chain(Structures.collate(structures).to("cuda"))
        expected = chain(Structures.collate(structures)).split(2)

        for part, reference in zip(batch.split(2), expected):
            assert part.device().type == "cuda"
            assert part.graph.device().type == "cuda"
            assert part.triplets.device().type == "cuda"
            assert_consistent(part)
            assert_same_attached(part.to("cpu"), reference)
