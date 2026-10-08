"""Dataset of structures held in memory."""

import pytest
import torch
from torch.utils.data import DataLoader

from torchms.data import HDF5Dataset, InMemoryDataset, OnDevice, Structures
from torchms.data.augmentation import Compose, LatticeRotation
from torchms.graph.transforms import AddRadiusGraph, AddTriplets


@pytest.fixture
def structures() -> list[Structures]:
    generator = torch.Generator().manual_seed(0)

    return [
        Structures(
            lattice=(4.0 * torch.eye(3) + torch.rand(3, 3, generator=generator))[None],
            x=torch.rand(n_atoms, 3, generator=generator),
            z=torch.randint(1, 90, (n_atoms,), generator=generator),
            total_energy=[float(i)],
            band_gap=[0.5 * i],
        )
        for i, n_atoms in enumerate([3, 1, 5, 2, 4, 2, 6, 1, 3, 2])
    ]


@pytest.fixture
def hdf5(tmp_path, structures) -> HDF5Dataset:
    hdf5_path = tmp_path / "test.hdf5"
    HDF5Dataset.from_structures(hdf5_path, structures)

    return HDF5Dataset(hdf5_path)


def assert_same(result: Structures, expected: Structures):
    for name in ("lattice", "x", "z", "num_atoms", "batch", "ptr"):
        assert torch.equal(getattr(result, name).cpu(), getattr(expected, name)), name
    for name in ("total_energy", "band_gap", "energy_above_hull"):
        assert torch.allclose(
            getattr(result, name).cpu(), getattr(expected, name), equal_nan=True
        ), name


class TestSelect:
    @pytest.mark.parametrize(
        "index", [[0], [4, 0, 6, 2], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [3, 3, 0, 3]]
    )
    def test_is_the_batch_of_the_structures(self, structures, index):
        batch = Structures.collate(structures)

        result = batch.select(torch.tensor(index))

        assert_same(result, Structures.collate([structures[i] for i in index]))
        assert result.graph is None

    def test_copies(self, structures):
        batch = Structures.collate(structures)
        result = batch.select(torch.tensor([2, 5]))
        result.x.zero_()

        assert_same(batch, Structures.collate(structures))

    def test_with_a_graph(self, structures):
        batch = AddRadiusGraph(3.0)(Structures.collate(structures))

        with pytest.raises(ValueError, match="graph"):
            batch.select(torch.tensor([0, 1]))


class TestDataset:
    def test_same_as_the_file(self, hdf5, structures):
        dataset = InMemoryDataset.from_hdf5(hdf5)

        assert len(dataset) == len(hdf5) == len(structures)
        for index in (0, -1, slice(2, 9, 3), slice(None, None, -2), [7, 1, 1, 4]):
            assert_same(dataset[index], hdf5[index])
        assert_same(dataset[torch.tensor([5, 0, 8])], hdf5[[5, 0, 8]])

    def test_from_a_path(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5.hdf5_file)

        assert_same(dataset[:], hdf5[:])

    def test_structures_with_a_graph(self, structures):
        batch = AddRadiusGraph(3.0)(Structures.collate(structures))

        with pytest.raises(ValueError, match="no graph"):
            InMemoryDataset(batch)

    def test_transform(self, hdf5):
        calls = []

        def record(batch: Structures) -> Structures:
            calls.append(len(batch.num_atoms))
            return batch

        dataset = InMemoryDataset.from_hdf5(hdf5, transform=record)
        dataset[[1, 2, 3]]
        (batch,) = dataset.__getitems__([4, 0])

        assert calls == [3, 2]
        assert_same(batch, hdf5[[4, 0]])

    def test_dataloader(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5)
        batches = list(DataLoader(dataset, batch_size=4))

        assert [len(batch.num_atoms) for batch in batches] == [4, 4, 2]
        for batch, expected in zip(batches, DataLoader(hdf5, batch_size=4)):
            assert_same(batch, expected)

    def test_shuffled_dataloader_is_the_one_of_the_file(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5)
        first = torch.Generator().manual_seed(0)
        second = torch.Generator().manual_seed(0)

        in_memory = DataLoader(dataset, batch_size=3, shuffle=True, generator=first)
        from_file = DataLoader(hdf5, batch_size=3, shuffle=True, generator=second)

        for batch, expected in zip(in_memory, from_file):
            assert_same(batch, expected)

    def test_batches(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5)

        batches = list(dataset.batches(4))
        assert [len(batch.num_atoms) for batch in batches] == [4, 4, 2]
        assert_same(Structures.collate(batches), hdf5[:])

        batches = list(dataset.batches(4, drop_last=True))
        assert [len(batch.num_atoms) for batch in batches] == [4, 4]

    def test_shuffled_batches(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5)
        generator = torch.Generator().manual_seed(0)

        batches = list(dataset.batches(3, shuffle=True, generator=generator))
        energy = torch.cat([batch.total_energy for batch in batches])

        assert [len(batch.num_atoms) for batch in batches] == [3, 3, 3, 1]
        assert energy.tolist() != list(range(10))
        assert sorted(energy.tolist()) == list(range(10))
        for batch in batches:
            # the atoms of a batch are the ones of its structures
            assert_same(batch, hdf5[batch.total_energy.long().tolist()])

        again = list(
            dataset.batches(3, shuffle=True, generator=torch.Generator().manual_seed(0))
        )
        assert torch.equal(torch.cat([b.total_energy for b in again]), energy)

    def test_graphs_in_small_batches(self, hdf5):
        """Large batches from memory, graphs built once, given in small batches."""
        chain = Compose([AddRadiusGraph(3.0, geometry=True), AddTriplets()])
        dataset = InMemoryDataset.from_hdf5(hdf5)

        result = list(OnDevice(dataset.batches(6), "cpu", chain, batch_size=2))
        expected = list(OnDevice(DataLoader(hdf5, batch_size=2), "cpu", chain))

        assert len(result) == len(expected) == 5
        for batch, reference in zip(result, expected):
            assert_same(batch, reference)
            assert torch.equal(batch.graph.edge_index, reference.graph.edge_index)
            assert torch.equal(batch.triplets.index, reference.triplets.index)

    def test_augmentation(self, hdf5):
        dataset = InMemoryDataset.from_hdf5(hdf5, transform=LatticeRotation())
        (batch,) = list(dataset.batches(10))

        assert not torch.allclose(batch.lattice, hdf5[:].lattice)
        assert torch.allclose(
            batch.lattice.norm(dim=-1), hdf5[:].lattice.norm(dim=-1), atol=1e-4
        )
        # the structures in memory are not modified
        assert_same(dataset.structures, hdf5[:])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_on_gpu(self, hdf5):
        chain = Compose([AddRadiusGraph(3.0, geometry=True), AddTriplets()])
        dataset = InMemoryDataset.from_hdf5(hdf5, device="cuda")

        assert dataset.device().type == "cuda"
        assert dataset.to("cpu").device().type == "cpu"
        assert_same(dataset[[7, 1, 1, 4]], hdf5[[7, 1, 1, 4]])

        loader = DataLoader(dataset, batch_size=6)
        result = list(OnDevice(loader, "cuda", chain, batch_size=2))
        expected = list(OnDevice(DataLoader(hdf5, batch_size=2), "cpu", chain))

        assert len(result) == len(expected) == 5
        for batch, reference in zip(result, expected):
            assert batch.device().type == "cuda"
            assert batch.graph.device().type == "cuda"
            assert_same(batch, reference)
            assert torch.equal(batch.graph.edge_index.cpu(), reference.graph.edge_index)

        generator = torch.Generator(device="cuda").manual_seed(0)
        shuffled = list(dataset.batches(4, shuffle=True, generator=generator))
        energy = torch.cat([batch.total_energy for batch in shuffled]).cpu()
        assert sorted(energy.tolist()) == list(range(10))
