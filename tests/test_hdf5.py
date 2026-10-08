import copy
import multiprocessing
import os
import pickle

import pytest
import torch
from torch.utils.data import DataLoader

from torchms.data import Structures, HDF5Dataset


@pytest.fixture
def diamond_struct():
    return Structures.diamond()


@pytest.fixture
def zincblende_struct():
    return Structures.zincblende()


@pytest.fixture
def diamond_batch():
    return Structures.collate([Structures.diamond() for _ in range(4)])


@pytest.fixture
def zincblende_batch():
    return Structures.collate([Structures.zincblende() for _ in range(4)])


@pytest.fixture
def mixed_batch():
    return Structures.collate(
        [
            Structures.diamond(),
            Structures.zincblende(),
            Structures.diamond(),
            Structures.zincblende(),
        ]
    )


class TestGetItem:
    def test_single_index(self, tmp_path, mixed_batch):
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])
        dataset = HDF5Dataset(hdf5_path)

        result = dataset[0]
        assert torch.equal(result.z, torch.tensor([6] * 8))
        assert result.num_atoms.shape == (1,)
        assert torch.equal(result.num_atoms, torch.tensor([8]))

        result = dataset[1]
        assert torch.equal(result.z, torch.tensor([16, 16, 16, 16, 30, 30, 30, 30]))
        assert result.num_atoms.shape == (1,)
        assert torch.equal(result.num_atoms, torch.tensor([8]))

    def test_slice(self, tmp_path, mixed_batch):
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])
        dataset = HDF5Dataset(hdf5_path)

        result = dataset[0:2]

        assert torch.equal(
            result.z,
            torch.tensor([6, 6, 6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 30, 30, 30, 30]),
        )
        assert torch.equal(result.num_atoms, torch.tensor([8, 8]))

    def test_list_indices(self, tmp_path, mixed_batch):
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])
        dataset = HDF5Dataset(hdf5_path)

        result = dataset[[1, 2]]

        assert torch.equal(
            result.z,
            torch.tensor([16, 16, 16, 16, 30, 30, 30, 30, 6, 6, 6, 6, 6, 6, 6, 6]),
        )
        assert torch.equal(result.num_atoms, torch.tensor([8, 8]))


class TestRoundTripMixed:
    def test_batch(self, tmp_path, mixed_batch):
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])
        dataset = HDF5Dataset(hdf5_path)

        loaded = dataset[:]
        originals = Structures.separate(mixed_batch)

        assert len(originals) == len(loaded.num_atoms)

        for i, original in enumerate(originals):
            assert torch.equal(loaded.num_atoms[i : i + 1], original.num_atoms)
            assert torch.allclose(loaded.lattice[i], original.lattice, atol=1e-6)
            start = loaded.ptr[i].item()
            end = loaded.ptr[i + 1].item()
            assert torch.allclose(loaded.x[start:end], original.x, atol=1e-6)
            assert torch.equal(loaded.z[start:end], original.z)


class TestTransform:
    def test_transform_applied(self, tmp_path, diamond_struct):
        hdf5_path = tmp_path / "test.hdf5"
        diamond_struct.total_energy = torch.tensor([1.0])
        diamond_struct.energy_above_hull = torch.tensor([0.5])
        diamond_struct.band_gap = torch.tensor([5.0])
        HDF5Dataset.from_structures(hdf5_path, [diamond_struct])

        def double_energy(s):
            s.total_energy = s.total_energy * 2
            return s

        dataset = HDF5Dataset(hdf5_path, transform=double_energy)
        result = dataset[0]

        assert result.total_energy is not None
        assert torch.equal(result.total_energy, torch.tensor([2.0]))


class TestVariousCompositions:
    def test_various_compositions_roundtrip(self, tmp_path):
        diamond = Structures.diamond()
        zn_s = Structures.zincblende(cation=30, anion=16, a=5.41)
        ga_as = Structures.zincblende(cation=31, anion=33, a=5.65)
        cd_te = Structures.zincblende(cation=48, anion=52, a=6.48)

        batch = Structures.collate([diamond, zn_s, ga_as, cd_te])

        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [batch])
        dataset = HDF5Dataset(hdf5_path)

        loaded = dataset[:]
        originals = Structures.separate(batch)

        assert len(originals) == 4

        original = originals[0]
        assert torch.allclose(original.lattice[0, 0, 0], torch.tensor(3.567), atol=1e-2)
        assert torch.equal(original.z, torch.tensor([6] * 8))

        original = originals[1]
        assert torch.allclose(original.lattice[0, 0, 0], torch.tensor(5.41), atol=1e-2)
        assert torch.equal(original.z, torch.tensor([16, 16, 16, 16, 30, 30, 30, 30]))

        original = originals[2]
        assert torch.allclose(original.lattice[0, 0, 0], torch.tensor(5.65), atol=1e-2)
        assert torch.equal(original.z, torch.tensor([33, 33, 33, 33, 31, 31, 31, 31]))

        original = originals[3]
        assert torch.allclose(original.lattice[0, 0, 0], torch.tensor(6.48), atol=1e-2)
        assert torch.equal(original.z, torch.tensor([52, 52, 52, 52, 48, 48, 48, 48]))


class TestMultipleBatches:
    """Anti-regression: from_structures must handle more than one batch."""

    @pytest.fixture
    def batches(self, mixed_batch):
        return [
            Structures.collate([Structures.zincblende(), Structures.diamond()]),
            Structures.diamond(),
            mixed_batch,
        ]

    @pytest.fixture
    def sources(self, batches):
        return [s for batch in batches for s in Structures.separate(batch)]

    @staticmethod
    def assert_same(loaded, expected):
        assert torch.equal(loaded.num_atoms, expected.num_atoms)
        assert torch.equal(loaded.z, expected.z)
        assert torch.allclose(loaded.lattice, expected.lattice)
        assert torch.allclose(loaded.x, expected.x)
        assert torch.equal(loaded.ptr, expected.ptr)
        assert torch.equal(loaded.batch, expected.batch)

    def test_length_and_ptr(self, tmp_path, batches, sources):
        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(hdf5_path, batches)

        dataset = HDF5Dataset(hdf5_path)
        assert len(dataset) == len(sources)

        num_atoms = torch.cat([s.num_atoms for s in sources])
        expected_ptr = num_atoms.cumsum(0) - num_atoms
        stored_ptr = torch.from_numpy(dataset.data_file["ptr"][:])
        assert torch.equal(stored_ptr, expected_ptr)

    def test_single_index(self, tmp_path, batches, sources):
        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(hdf5_path, batches)

        dataset = HDF5Dataset(hdf5_path)
        for i, expected in enumerate(sources):
            # separate() keeps the batch index of the source batch, collate resets it
            self.assert_same(dataset[i], Structures.collate([expected]))

    def test_across_batch_boundary(self, tmp_path, batches, sources):
        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(hdf5_path, batches)

        dataset = HDF5Dataset(hdf5_path)
        self.assert_same(dataset[1:5], Structures.collate(sources[1:5]))
        self.assert_same(
            dataset[[0, 2, 3, 6]],
            Structures.collate([sources[i] for i in (0, 2, 3, 6)]),
        )

    def test_same_as_single_batch(self, tmp_path, batches, sources):
        multi_path = tmp_path / "multi.h5"
        single_path = tmp_path / "single.h5"
        HDF5Dataset.from_structures(multi_path, batches)
        HDF5Dataset.from_structures(single_path, Structures.collate(sources))

        multi = HDF5Dataset(multi_path)
        single = HDF5Dataset(single_path)
        assert len(multi) == len(single)
        self.assert_same(multi[:], single[:])
        assert torch.equal(
            torch.from_numpy(multi.data_file["ptr"][:]),
            torch.from_numpy(single.data_file["ptr"][:]),
        )


class TestFromHub:
    def test_from_hub(self, tmp_path, mixed_batch, monkeypatch):
        import sys
        import types

        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])

        calls = []

        def hf_hub_download(repo_id, filename, repo_type=None, **kwargs):
            calls.append((repo_id, filename, repo_type, kwargs))
            return str(hdf5_path)

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        dataset = HDF5Dataset.from_hub("user/repo", "test.h5", revision="main")

        assert calls == [("user/repo", "test.h5", "dataset", {"revision": "main"})]
        assert len(dataset) == 4


class TestFormationEnergy:
    def test_roundtrip(self, tmp_path, mixed_batch):
        hdf5_path = tmp_path / "test.h5"
        mixed_batch.formation_energy_per_atom = torch.tensor(
            [-1.5, 0.25, float("nan"), -0.5]
        )
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch, mixed_batch])

        dataset = HDF5Dataset(hdf5_path)
        assert torch.equal(dataset[1].formation_energy_per_atom, torch.tensor([0.25]))
        assert torch.equal(
            dataset[[3, 4, 5]].formation_energy_per_atom,
            torch.tensor([-0.5, -1.5, 0.25]),
        )
        assert torch.isnan(dataset[2].formation_energy_per_atom).all()


class TestSingleAtom:
    """Anti-regression: structures with a single atom must be readable."""

    @pytest.fixture
    def single_atom(self):
        return Structures(
            lattice=torch.eye(3).unsqueeze(0) * 2.5,
            x=torch.tensor([[0.0, 0.0, 0.0]]),
            z=torch.tensor([26]),
        )

    def test_single_atom_alone(self, tmp_path, single_atom):
        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(hdf5_path, [single_atom])

        loaded = HDF5Dataset(hdf5_path)[0]
        assert torch.equal(loaded.z, torch.tensor([26]))
        assert loaded.x.shape == (1, 3)
        assert torch.equal(loaded.num_atoms, torch.tensor([1]))

    def test_single_atom_in_batch(self, tmp_path, single_atom, diamond_struct):
        hdf5_path = tmp_path / "test.h5"
        HDF5Dataset.from_structures(
            hdf5_path, [diamond_struct, single_atom, diamond_struct]
        )

        dataset = HDF5Dataset(hdf5_path)
        assert torch.equal(dataset[1].z, torch.tensor([26]))
        assert dataset[1].x.shape == (1, 3)
        assert torch.equal(dataset[:].num_atoms, torch.tensor([8, 1, 8]))


class TestProcesses:
    """The file of a dataset is opened in each process which reads it."""

    @staticmethod
    def make_dataset(tmp_path, mixed_batch, transform=None) -> HDF5Dataset:
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, [mixed_batch])
        return HDF5Dataset(hdf5_path, transform=transform)

    def test_missing_file_fails_at_creation(self, tmp_path):
        with pytest.raises(OSError):
            HDF5Dataset(tmp_path / "missing.hdf5")

    def test_pickle(self, tmp_path, mixed_batch):
        dataset = self.make_dataset(tmp_path, mixed_batch)
        assert len(dataset) == 4

        # an open HDF5 file cannot be pickled: it is not part of the state
        restored = pickle.loads(pickle.dumps(dataset))

        assert restored.hdf5_file == dataset.hdf5_file
        assert restored.data_file is not dataset.data_file
        assert len(restored) == 4
        assert torch.equal(restored[1:3].x, dataset[1:3].x)
        assert torch.equal(restored[1:3].z, dataset[1:3].z)

    def test_file_is_opened_again_in_another_process(self, tmp_path, mixed_batch):
        dataset = self.make_dataset(tmp_path, mixed_batch)
        opened = dataset.data_file
        assert dataset.data_file is opened

        # as after a fork: the file was opened by a process with another pid
        dataset._pid = os.getpid() + 1

        assert dataset.data_file is not opened
        assert dataset._pid == os.getpid()
        assert torch.equal(dataset[0].z, torch.tensor([6] * 8))

    @pytest.mark.timeout(60)
    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(), reason="no fork"
    )
    def test_workers_started_by_fork(self, tmp_path, mixed_batch):
        dataset = self.make_dataset(tmp_path, mixed_batch)

        def record_processes(structures: Structures) -> Structures:
            # the pid of the process which reads in a label, and the pid of
            # the one which opened the file in another one
            n = len(structures.num_atoms)
            structures.total_energy = torch.full((n,), float(os.getpid()))
            structures.band_gap = torch.full((n,), float(dataset._pid or -1))
            return structures

        dataset.transform = record_processes
        assert len(dataset) == 4  # the file is open before the fork

        loader = DataLoader(
            dataset,
            sampler=[[0, 1], [2, 3]],
            batch_size=None,
            num_workers=2,
            multiprocessing_context="fork",
        )
        batches = list(loader)

        assert torch.equal(torch.cat([batch.z for batch in batches]), mixed_batch.z)
        for batch in batches:
            reader, opener = batch.total_energy, batch.band_gap
            assert (reader != float(os.getpid())).all()
            assert (opener == reader).all()

    @pytest.mark.timeout(120)
    def test_workers_started_by_spawn(self, tmp_path, mixed_batch):
        dataset = self.make_dataset(tmp_path, mixed_batch)

        loader = DataLoader(
            dataset,
            sampler=[[0, 1], [2, 3]],
            batch_size=None,
            num_workers=1,
            multiprocessing_context="spawn",
        )
        batches = list(loader)

        assert len(batches) == 2
        assert torch.equal(torch.cat([batch.z for batch in batches]), mixed_batch.z)
        assert torch.equal(torch.cat([batch.x for batch in batches]), mixed_batch.x)


class TestAnyOrder:
    """Indices in any order, read in one call."""

    @pytest.fixture
    def dataset(self, tmp_path):
        generator = torch.Generator().manual_seed(0)
        structures = [
            Structures(
                lattice=(4.0 * torch.eye(3) + torch.rand(3, 3, generator=generator))[
                    None
                ],
                x=torch.rand(n_atoms, 3, generator=generator),
                z=torch.randint(1, 90, (n_atoms,), generator=generator),
                total_energy=[float(i)],
            )
            for i, n_atoms in enumerate([3, 1, 5, 2, 4, 2, 6])
        ]
        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, structures)

        return HDF5Dataset(hdf5_path), structures

    @staticmethod
    def assert_same(result: Structures, expected: Structures):
        assert torch.equal(result.lattice, expected.lattice)
        assert torch.equal(result.x, expected.x)
        assert torch.equal(result.z, expected.z)
        assert torch.equal(result.num_atoms, expected.num_atoms)
        assert torch.equal(result.batch, expected.batch)
        assert torch.equal(result.ptr, expected.ptr)
        assert torch.equal(result.total_energy, expected.total_energy)

    @pytest.mark.parametrize(
        "indices",
        [[4, 0, 6, 2], [6, 5, 4, 3, 2, 1, 0], [3, 3, 0, 3], [-1, 0, -7], [2], []],
    )
    def test_list(self, dataset, indices):
        dataset, structures = dataset

        if len(indices) == 0:
            pytest.skip("structures cannot be empty")

        expected = Structures.collate([structures[i] for i in indices])
        self.assert_same(dataset[indices], expected)

    def test_tensor(self, dataset):
        dataset, structures = dataset
        indices = torch.tensor([5, 1, 4, 1])

        expected = Structures.collate([structures[i] for i in indices.tolist()])
        self.assert_same(dataset[indices], expected)

    def test_negative_index_and_slices(self, dataset):
        dataset, structures = dataset

        self.assert_same(dataset[-1], structures[-1])
        self.assert_same(dataset[::-2], Structures.collate(structures[::-2]))
        self.assert_same(dataset[1:6:2], Structures.collate(structures[1:6:2]))

    def test_getitems_reads_one_batch(self, dataset):
        dataset, structures = dataset
        calls = []

        def record(batch: Structures) -> Structures:
            calls.append(len(batch.num_atoms))
            return batch

        dataset.transform = record
        (batch,) = dataset.__getitems__([4, 0, 6])

        assert calls == [3]
        self.assert_same(batch, Structures.collate([structures[i] for i in (4, 0, 6)]))


class TestDataLoader:
    """A DataLoader batches the dataset without sampler nor collate function."""

    @pytest.fixture
    def dataset(self, tmp_path):
        structures = [
            Structures.diamond() if i % 2 == 0 else Structures.zincblende()
            for i in range(10)
        ]
        for i, struct in enumerate(structures):
            struct.total_energy = torch.tensor([float(i)])

        hdf5_path = tmp_path / "test.hdf5"
        HDF5Dataset.from_structures(hdf5_path, structures)

        return HDF5Dataset(hdf5_path)

    def test_in_order(self, dataset):
        batches = list(DataLoader(dataset, batch_size=4))

        assert all(isinstance(batch, Structures) for batch in batches)
        assert [len(batch.num_atoms) for batch in batches] == [4, 4, 2]
        energy = torch.cat([batch.total_energy for batch in batches])
        assert energy.tolist() == list(range(10))
        assert torch.equal(torch.cat([batch.x for batch in batches]), dataset[:].x)

    def test_shuffle(self, dataset):
        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=generator)
        batches = list(loader)

        energy = torch.cat([batch.total_energy for batch in batches])
        assert energy.tolist() != list(range(10))
        assert sorted(energy.tolist()) == list(range(10))
        for batch in batches:
            # atoms follow their structure: diamond is carbon, zincblende is not
            is_diamond = batch.total_energy.long() % 2 == 0
            assert ((batch.z == 6) == is_diamond[batch.batch]).all()

    def test_drop_last(self, dataset):
        batches = list(DataLoader(dataset, batch_size=4, drop_last=True))

        assert [len(batch.num_atoms) for batch in batches] == [4, 4]

    def test_random_split(self, dataset):
        generator = torch.Generator().manual_seed(0)
        train, test = torch.utils.data.random_split(dataset, [7, 3], generator)

        seen = []
        for subset in (train, test):
            batches = list(DataLoader(subset, batch_size=2, shuffle=True))
            assert all(isinstance(batch, Structures) for batch in batches)
            seen.append(torch.cat([batch.total_energy for batch in batches]).tolist())

        assert len(seen[0]) == 7 and len(seen[1]) == 3
        assert sorted(seen[0] + seen[1]) == list(range(10))

    def test_collate_function(self, dataset):
        def count(batch: list[Structures]) -> int:
            assert isinstance(batch, list) and len(batch) == 1
            return len(batch[0].num_atoms)

        assert list(DataLoader(dataset, batch_size=4, collate_fn=count)) == [4, 4, 2]

    def test_graph_transforms(self, dataset):
        from torchms.data.augmentation import Compose
        from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets

        chain = Compose(
            [
                AddRadiusGraph(3.0, geometry=True),
                AddTriplets(geometry=True),
                AddQuadruplets(2.0, geometry=True),
            ]
        )
        plain = list(DataLoader(dataset, batch_size=4))
        dataset.transform = chain
        batches = list(DataLoader(dataset, batch_size=4))

        for batch, structures in zip(batches, plain):
            expected = chain(structures)
            assert torch.equal(batch.graph.edge_index, expected.graph.edge_index)
            assert torch.equal(batch.graph.edge_distance, expected.graph.edge_distance)
            assert torch.equal(batch.triplets.index, expected.triplets.index)
            assert torch.equal(batch.quadruplets.angle, expected.quadruplets.angle)
            assert batch.graph.structures.x is batch.x

    @pytest.mark.timeout(120)
    def test_workers_started_by_spawn(self, dataset):
        from torchms.graph.transforms import AddRadiusGraph, AddTriplets

        dataset.transform = AddRadiusGraph(3.0, geometry=True)
        loader = DataLoader(
            dataset, batch_size=4, num_workers=1, multiprocessing_context="spawn"
        )
        batches = list(loader)
        dataset.transform = None

        assert [len(batch.num_atoms) for batch in batches] == [4, 4, 2]
        for batch, structures in zip(batches, DataLoader(dataset, batch_size=4)):
            assert torch.equal(batch.x, structures.x)
            expected = AddRadiusGraph(3.0, geometry=True)(structures).graph
            assert torch.equal(batch.graph.edge_index, expected.edge_index)
            assert torch.equal(batch.graph.structures.x, batch.x)
            # tuples of edges can still be added to a graph coming from a worker
            assert AddTriplets()(batch).triplets.num_tuples.sum() > 0

    def test_graph_is_not_written(self, tmp_path):
        from torchms.graph.transforms import AddRadiusGraph, AddTriplets

        structures = AddTriplets()(AddRadiusGraph(2.0)(Structures.diamond()))
        hdf5_path = tmp_path / "graph.hdf5"
        HDF5Dataset.from_structures(hdf5_path, structures)

        dataset = HDF5Dataset(hdf5_path)
        assert "graph" not in dataset.data_file
        assert dataset[0].graph is None
        assert torch.equal(dataset[0].x, structures.x)
