"""Batches of a loader moved to a device and transformed on it."""

import pytest
import torch
from torch.utils.data import DataLoader

from torchms.data import HDF5Dataset, OnDevice, Structures
from torchms.data.augmentation import Compose, LatticeRotation
from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets


@pytest.fixture
def dataset(tmp_path) -> HDF5Dataset:
    structures = [
        Structures.diamond() if i % 2 == 0 else Structures.zincblende()
        for i in range(10)
    ]
    for i, struct in enumerate(structures):
        struct.total_energy = torch.tensor([float(i)])

    hdf5_path = tmp_path / "test.hdf5"
    HDF5Dataset.from_structures(hdf5_path, structures)

    return HDF5Dataset(hdf5_path)


def chain() -> Compose:
    return Compose(
        [
            AddRadiusGraph(3.0, geometry=True),
            AddTriplets(geometry=True),
            AddQuadruplets(2.0, geometry=True),
        ]
    )


def assert_same_graphs(result: Structures, expected: Structures):
    assert torch.equal(result.x.cpu(), expected.x)
    assert torch.equal(result.graph.edge_index.cpu(), expected.graph.edge_index)
    assert torch.equal(result.graph.edge_shift.cpu(), expected.graph.edge_shift)
    assert torch.allclose(
        result.graph.edge_distance.cpu(), expected.graph.edge_distance, atol=1e-5
    )
    assert torch.equal(result.triplets.index.cpu(), expected.triplets.index)
    assert torch.equal(result.quadruplets.index.cpu(), expected.quadruplets.index)
    assert torch.allclose(
        result.quadruplets.angle.cpu(), expected.quadruplets.angle, atol=1e-4
    )


def test_without_transform(dataset):
    loader = OnDevice(DataLoader(dataset, batch_size=4), "cpu")
    batches = list(loader)

    assert len(loader) == 3
    assert [len(batch.num_atoms) for batch in batches] == [4, 4, 2]
    assert all(batch.graph is None for batch in batches)
    assert torch.cat([b.total_energy for b in batches]).tolist() == list(range(10))


def test_same_graphs_as_the_transform_of_the_dataset(dataset):
    on_device = list(OnDevice(DataLoader(dataset, batch_size=4), "cpu", chain()))

    dataset.transform = chain()
    in_dataset = list(DataLoader(dataset, batch_size=4))

    assert len(on_device) == len(in_dataset) == 3
    for result, expected in zip(on_device, in_dataset):
        assert_same_graphs(result, expected)


def test_several_epochs(dataset):
    loader = OnDevice(DataLoader(dataset, batch_size=4), "cpu", AddRadiusGraph(3.0))

    first = [batch.graph.num_edges.tolist() for batch in loader]
    second = [batch.graph.num_edges.tolist() for batch in loader]

    assert len(first) == 3
    assert first == second


def test_transform_runs_after_the_move(dataset):
    seen = []

    def record(structures: Structures) -> Structures:
        seen.append(structures.device())
        return structures

    loader = OnDevice(DataLoader(dataset, batch_size=5), torch.device("cpu"), record)
    list(loader)

    assert seen == [torch.device("cpu")] * 2
    assert loader.device == torch.device("cpu")


def test_augmentations_in_the_dataset_then_graphs(dataset):
    """Augmentations in the workers, graphs on the device, on the same structures."""
    dataset.transform = LatticeRotation()
    loader = OnDevice(DataLoader(dataset, batch_size=4), "cpu", chain())

    for batch in loader:
        expected = chain()(batch.without_graphs())
        assert_same_graphs(batch, expected)
        # rotated cells, with the bonds of diamond and zincblende
        assert not torch.allclose(batch.lattice[0], Structures.diamond().lattice[0])
        distances = batch.graph.edge_distance.round(decimals=2).unique()
        assert torch.allclose(distances, torch.tensor([1.54, 2.34, 2.52, 2.96]))


def test_any_iterable_of_structures():
    batches = [Structures.diamond(), Structures.zincblende()]
    loader = OnDevice(batches, "cpu", AddRadiusGraph(3.0))

    assert len(loader) == 2
    assert [batch.graph.num_edges.tolist() for batch in loader] == [[224], [32]]


def test_length_of_a_loader_without_length():
    loader = OnDevice(iter([Structures.diamond()]), "cpu")

    with pytest.raises(TypeError, match="no length"):
        len(loader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.timeout(120)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_graphs_built_on_the_gpu(dataset, num_workers):
    loader = OnDevice(
        DataLoader(dataset, batch_size=4, num_workers=num_workers), "cuda", chain()
    )
    batches = list(loader)

    dataset.transform = chain()
    in_dataset = list(DataLoader(dataset, batch_size=4))

    assert len(batches) == 3
    for result, expected in zip(batches, in_dataset):
        assert result.device().type == "cuda"
        assert result.graph.device().type == "cuda"
        assert result.graph.edge_distance.device.type == "cuda"
        assert result.triplets.device().type == "cuda"
        assert result.quadruplets.angle.device.type == "cuda"
        assert result.graph.structures.x is result.x
        assert_same_graphs(result, expected)
