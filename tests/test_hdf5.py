import copy
import pytest
import torch

from pycrystalgen.data import Structures, HDF5Dataset


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

        result = dataset[[1, 0]]

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
