"""Subsets, filters and splits of datasets."""

import dataclasses

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchms.data import (
    Filter,
    HDF5Dataset,
    InMemoryDataset,
    Structures,
    auto_loader,
    load_split,
    random_split,
    save_split,
    split_indices,
)
from torchms.data.auto import dataset_size

N = 40


@pytest.fixture
def structures() -> Structures:
    """Crystals and molecules of different sizes, numbered by a property."""
    generator = torch.Generator().manual_seed(0)
    elements = []

    for i in range(N):
        n_atoms = 1 + i % 7
        x = torch.rand(n_atoms, 3, generator=generator)
        z = torch.randint(1, 20, (n_atoms,), generator=generator)

        if i % 3 == 0:
            elements.append(Structures.molecule(4.0 * x, z))
        else:
            lattice = 4.0 * torch.eye(3) + torch.rand(3, 3, generator=generator)
            elements.append(Structures(lattice=lattice[None], x=x, z=z))

    batch = Structures.collate(elements)
    number = torch.arange(N, dtype=torch.float32)
    energy = -number.clone()
    energy[5] = torch.nan

    return dataclasses.replace(
        batch, total_energy=energy, properties={"number": number, "odd": number % 2}
    )


@pytest.fixture
def hdf5(tmp_path, structures) -> HDF5Dataset:
    file = tmp_path / "structures.hdf5"
    HDF5Dataset.from_structures(file, structures)

    return HDF5Dataset(file)


@pytest.fixture(params=["file", "memory"])
def dataset(request, hdf5, structures):
    return hdf5 if request.param == "file" else InMemoryDataset(structures)


def numbers(structures: Structures) -> list[int]:
    return structures.properties["number"].long().tolist()


def assert_same(a: Structures, b: Structures):
    assert torch.equal(a.num_atoms, b.num_atoms)
    assert torch.equal(a.x, b.x) and torch.equal(a.z, b.z)
    assert torch.equal(a.lattice, b.lattice) and torch.equal(a.pbc, b.pbc)
    assert numbers(a) == numbers(b)


class TestSubset:
    def test_structures_of_a_subset(self, dataset):
        index = [7, 2, 39, 2, 11]
        subset = dataset.subset(index)

        assert type(subset) is type(dataset) and len(subset) == 5
        assert numbers(subset[:]) == index
        assert_same(subset[:], dataset[index])
        # indexed as a dataset: from the end, slices, any order
        assert numbers(subset[-1]) == [11]
        assert numbers(subset[1:3]) == [2, 39]
        assert numbers(subset[[4, 0]]) == [11, 7]
        assert numbers(subset[torch.tensor([3])]) == [2]

    def test_subset_of_a_subset(self, dataset):
        subset = dataset.subset(torch.arange(10, 30)).subset([19, 0, 5])

        assert numbers(subset[:]) == [29, 10, 15]

    def test_invalid_indices(self, hdf5):
        with pytest.raises(IndexError):
            hdf5.subset([N])
        with pytest.raises(IndexError):
            hdf5.subset([3]).subset([1])

    def test_the_transform_is_kept(self, dataset):
        def transform(structures):
            return dataclasses.replace(structures, band_gap=structures.total_energy)

        subset = dataset.with_transform(transform).subset([3, 4])

        assert subset[:].band_gap.tolist() == [-3.0, -4.0]

    def test_in_memory_from_a_subset_of_a_file(self, hdf5):
        subset = hdf5.subset([9, 4, 30])
        in_memory = InMemoryDataset.from_hdf5(subset)

        assert numbers(in_memory.structures) == [9, 4, 30]
        assert_same(in_memory.structures, hdf5[[9, 4, 30]])
        # the whole file is still read whole
        assert len(InMemoryDataset.from_hdf5(hdf5)) == N

    def test_size_of_a_subset(self, hdf5):
        subset = hdf5.subset(torch.arange(0, N, 2))

        assert dataset_size(subset) < dataset_size(hdf5)
        assert dataset_size(subset) == dataset_size(InMemoryDataset.from_hdf5(subset))

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataloader(self, hdf5, num_workers):
        index = list(range(3, 33, 3))
        loader = DataLoader(hdf5.subset(index), batch_size=4, num_workers=num_workers)

        assert [numbers(batch) for batch in loader] == [
            index[0:4],
            index[4:8],
            index[8:10],
        ]

    @pytest.mark.parametrize("storage", ["file", "memory"])
    def test_auto_loader(self, hdf5, storage):
        index = list(range(0, N, 4))
        subset = hdf5.subset(index)
        source = subset if storage == "file" else InMemoryDataset.from_hdf5(subset)

        loader = auto_loader(source, batch_size=4, device="cpu")
        seen = [n for batch in loader for n in numbers(batch)]

        assert len(loader) == 3 and sorted(seen) == index


class TestFilter:
    def test_filter_of_a_dataset(self, dataset, structures):
        condition = Filter(max_atoms=4, periodic=True)
        kept = dataset.filter(condition)

        expected = condition(structures).nonzero().squeeze(1).tolist()
        assert 0 < len(expected) < N
        assert numbers(kept[:]) == expected
        assert (kept[:].num_atoms <= 4).all() and kept[:].pbc.all()

    def test_filter_by_chunks(self, hdf5, structures):
        condition = Filter(equal={"odd": 1})

        assert numbers(hdf5.filter(condition, chunk_size=7)[:]) == list(range(1, N, 2))

    def test_any_function(self, dataset):
        kept = dataset.filter(lambda structures: structures.properties["number"] < 3)

        assert numbers(kept[:]) == [0, 1, 2]

    def test_filter_of_a_filter(self, dataset):
        kept = dataset.filter(Filter(equal={"odd": 0})).filter(
            Filter(minimum={"number": 30})
        )

        assert numbers(kept[:]) == [30, 32, 34, 36, 38]

    def test_the_filter_does_not_see_the_transform(self, dataset):
        def transform(structures):
            return dataclasses.replace(
                structures, properties={"number": -structures.properties["number"]}
            )

        kept = dataset.with_transform(transform).filter(Filter(minimum={"number": 38}))

        assert kept[:].properties["number"].tolist() == [-38.0, -39.0]

    def test_auto_loader(self, hdf5):
        loader = auto_loader(
            hdf5.hdf5_file,
            batch_size=4,
            filter=Filter(known=("total_energy",)),
            device="cpu",
        )

        assert sorted(n for batch in loader for n in numbers(batch)) == [
            n for n in range(N) if n != 5
        ]

    def test_a_filter_per_source(self, hdf5):
        loader = auto_loader(
            [hdf5, hdf5],
            batch_size=4,
            filter=[Filter(maximum={"number": 2}), None],
            device="cpu",
        )

        assert len(loader.dataset) == 3 + N

        with pytest.raises(ValueError, match="one filter per source"):
            auto_loader([hdf5, hdf5], batch_size=4, filter=[None], device="cpu")


class TestSplit:
    @pytest.mark.parametrize(
        "fractions", [[0.8, 0.1, 0.1], [0.5, 0.5], [1.0], [0.3] * 3 + [0.1]]
    )
    @pytest.mark.parametrize("n", [1, 10, 37, 1000])
    def test_every_structure_in_one_part(self, n, fractions):
        parts = split_indices(n, fractions, seed=3)

        assert len(parts) == len(fractions)
        assert torch.cat(parts).sort().values.tolist() == list(range(n))
        for part, fraction in zip(parts, fractions):
            assert abs(len(part) - fraction * n) <= 1
            assert (part[1:] > part[:-1]).all()

    def test_seed(self):
        first = split_indices(100, [0.5, 0.5], seed=1)
        again = split_indices(100, [0.5, 0.5], seed=1)
        other = split_indices(100, [0.5, 0.5], seed=2)

        assert torch.equal(first[0], again[0])
        assert not torch.equal(first[0], other[0])
        # drawn at random, not the first half
        assert first[0].tolist() != list(range(50))

    @pytest.mark.parametrize(
        "fractions", [[], [0.5, 0.4], [0.7, 0.7], [1.5, -0.5], [1.0, 0.0]]
    )
    def test_invalid_fractions(self, fractions):
        with pytest.raises(ValueError, match="fractions"):
            split_indices(10, fractions)

    def test_random_split(self, dataset):
        train, val, test = random_split(dataset, [0.7, 0.2, 0.1], seed=0)

        assert [len(train), len(val), len(test)] == [28, 8, 4]
        assert type(train) is type(dataset)
        seen = numbers(train[:]) + numbers(val[:]) + numbers(test[:])
        assert sorted(seen) == list(range(N))

        again = random_split(dataset, [0.7, 0.2, 0.1], seed=0)
        assert numbers(again[1][:]) == numbers(val[:])

    def test_split_of_what_a_filter_kept(self, dataset):
        molecules = dataset.filter(Filter(periodic=False))
        train, test = random_split(molecules, [0.5, 0.5])

        assert len(train) + len(test) == len(molecules) == 14
        assert not train[:].pbc.any() and not test[:].pbc.any()

    def test_save_and_load(self, tmp_path, dataset):
        train, test = split_indices(len(dataset), [0.75, 0.25], seed=4)
        file = tmp_path / "split.npz"
        save_split(file, {"train": train, "test": test}, dataset)

        parts = load_split(file, dataset)

        assert list(parts) == ["train", "test"]
        assert torch.equal(parts["train"], train) and torch.equal(parts["test"], test)
        assert parts["train"].dtype == torch.int64
        assert numbers(dataset.subset(parts["test"])[:]) == test.tolist()
        # readable without torchms
        assert np.load(file)["test"].tolist() == test.tolist()

    def test_split_of_another_dataset(self, tmp_path, dataset):
        file = tmp_path / "split.npz"
        save_split(file, {"train": [0, 1], "test": [2]}, 3)

        assert load_split(file)["test"].tolist() == [2]
        with pytest.raises(ValueError, match="3 structures, not 40"):
            load_split(file, dataset)

        # saved without the size of its dataset: the indices are checked
        save_split(file, {"train": [0, 50]})
        with pytest.raises(ValueError, match="indices"):
            load_split(file, dataset)
