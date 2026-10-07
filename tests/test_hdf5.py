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
