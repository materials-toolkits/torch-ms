"""Datasets mixed with a fixed number of structures of each one per batch."""

import dataclasses

import pytest
import torch

from torchms.data import (
    HDF5Dataset,
    InMemoryDataset,
    LoaderConfig,
    MixedDataset,
    Structures,
    auto_loader,
    mix,
    random_split,
)
from torchms.data import auto
from torchms.graph.transforms import AddRadiusGraph


def numbered(structures: list[Structures], first: int) -> InMemoryDataset:
    batch = Structures.collate(structures)
    number = first + torch.arange(len(structures), dtype=torch.float32)

    return InMemoryDataset(dataclasses.replace(batch, properties={"number": number}))


@pytest.fixture
def crystals() -> InMemoryDataset:
    """10 crystals, numbered from 0."""
    return numbered([Structures.diamond(), Structures.zincblende()] * 5, 0)


@pytest.fixture
def molecules() -> InMemoryDataset:
    """47 molecules, numbered from 100."""
    return numbered([Structures.water()] * 47, 100)


def numbers(structures: Structures) -> list[int]:
    return structures.properties["number"].long().tolist()


def assert_shares(batch: Structures, crystals: int, molecules: int):
    """The crystals of a batch first, then its molecules."""
    periodic = batch.pbc.all(dim=1).tolist()

    assert periodic == [True] * crystals + [False] * molecules


class TestMix:
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_shortest(self, crystals, molecules, shuffle):
        mixed = mix([crystals, molecules], [2, 3])
        batches = list(mixed.batches(5, shuffle=shuffle))

        # until the crystals run out
        assert isinstance(mixed, MixedDataset)
        assert len(batches) == 5 and mixed.epoch_size() == 25 and len(mixed) == 57
        for batch in batches:
            assert_shares(batch, 2, 3)

        seen = [n for batch in batches for n in numbers(batch)]
        assert len(set(seen)) == 25  # nothing twice
        assert sorted(n for n in seen if n < 100) == list(range(10))

    def test_longest(self, crystals, molecules):
        mixed = mix([crystals, molecules], [2, 3], epoch="longest")
        batches = list(mixed.batches(5, shuffle=True))

        # until every molecule has been given: 16 batches of 3
        assert len(batches) == 16 and mixed.epoch_size() == 80
        for batch in batches:
            assert_shares(batch, 2, 3)

        seen = [n for batch in batches for n in numbers(batch)]
        assert set(seen) == set(range(10)) | set(range(100, 147))
        # the crystals are gone through again, each one as often as another
        counts = torch.bincount(torch.tensor([n for n in seen if n < 100]))
        assert counts.min() >= 3 and counts.max() <= 4

    def test_without_shuffle(self, crystals, molecules):
        mixed = mix([crystals, molecules], [1, 2])

        assert numbers(next(iter(mixed.batches(3)))) == [0, 100, 101]
        assert numbers(list(mixed.batches(3))[1]) == [1, 102, 103]

    def test_larger_batches_hold_the_shares(self, crystals, molecules):
        mixed = mix([crystals, molecules], [1, 4])

        for batch in mixed.batches(10, shuffle=True):
            for part in batch.split(5):
                assert_shares(part, 1, 4)

        with pytest.raises(ValueError, match="multiple"):
            next(mixed.batches(7))

    def test_order_of_a_generator(self, crystals, molecules):
        mixed = mix([crystals, molecules], [2, 3])

        first = mixed.order(True, torch.Generator().manual_seed(0))
        again = mixed.order(True, torch.Generator().manual_seed(0))
        other = mixed.order(True, torch.Generator().manual_seed(1))

        assert torch.equal(first, again) and not torch.equal(first, other)

    def test_three_datasets(self, crystals, molecules):
        mixed = mix([molecules, crystals, molecules], [2, 1, 1])

        for batch in mixed.batches(4, shuffle=True):
            assert batch.pbc.all(dim=1).tolist() == [False, False, True, False]

    def test_invalid(self, crystals, molecules):
        with pytest.raises(ValueError, match="one count per dataset"):
            mix([crystals, molecules], [2])
        with pytest.raises(ValueError, match="at least one structure"):
            mix([crystals, molecules], [2, 0])
        with pytest.raises(ValueError, match="too small"):
            mix([crystals, molecules], [11, 1])
        with pytest.raises(ValueError, match="epoch"):
            mix([crystals, molecules], [1, 1], epoch="first")

    def test_subset_is_not_mixed(self, crystals, molecules):
        subset = mix([crystals, molecules], [1, 1]).subset([0, 12])

        assert type(subset) is InMemoryDataset and numbers(subset[:]) == [0, 102]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_on_gpu(self, crystals, molecules):
        mixed = mix([crystals.to("cuda"), molecules.to("cuda")], [2, 3])
        generator = torch.Generator().manual_seed(0)

        for batch in mixed.batches(5, shuffle=True, generator=generator):
            assert batch.device().type == "cuda"
            assert_shares(batch, 2, 3)


class TestAutoLoader:
    @pytest.fixture
    def files(self, tmp_path, crystals, molecules) -> list[str]:
        paths = []
        for name, dataset in (("crystals", crystals), ("molecules", molecules)):
            paths.append(str(tmp_path / f"{name}.hdf5"))
            HDF5Dataset.from_structures(paths[-1], dataset.structures)

        return paths

    def check(self, loader, n_batches: int):
        batches = list(loader)

        assert len(batches) == len(loader) == n_batches
        for batch in batches:
            assert_shares(batch, 2, 6)
            assert len(batch.graph.num_edges) == 8
            molecule = ~batch.pbc.all(dim=1)[batch.graph.edge_batch]
            assert (batch.graph.edge_shift[molecule] == 0).all()

        return batches

    def test_in_memory(self, files):
        loader = auto_loader(
            files,
            batch_size=8,
            counts=[2, 6],
            transform=AddRadiusGraph(2.0),
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
            device="cpu",
            config=LoaderConfig(torch.device("cpu"), "ram", 8, 8, 0, "device", []),
        )

        assert isinstance(loader.dataset, MixedDataset)
        self.check(loader, 5)

    def test_large_batches_split(self, files):
        """Graphs built on batches of 24, given by batches of 8."""
        loader = auto_loader(
            files,
            batch_size=8,
            counts=[2, 6],
            transform=AddRadiusGraph(2.0),
            shuffle=True,
            device="cpu",
            config=LoaderConfig(torch.device("cpu"), "ram", 8, 24, 0, "device", []),
        )

        self.check(loader, 5)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataloader(self, files, num_workers):
        """Without GPU: a DataLoader, which must not draw its own order."""
        config = LoaderConfig(
            torch.device("cpu"), "ram", 8, 8, num_workers, "workers", []
        )
        loader = auto_loader(
            files,
            batch_size=8,
            counts=[2, 6],
            transform=AddRadiusGraph(2.0),
            shuffle=True,
            device="cpu",
            config=config,
        )

        first = self.check(loader, 5)
        second = self.check(loader, 5)
        seen = lambda batches: [n for batch in batches for n in numbers(batch)]
        # another order at the next epoch
        assert seen(first) != seen(second)

    def test_detected(self, files):
        loader = auto_loader(
            files,
            batch_size=8,
            counts=[2, 6],
            transform=AddRadiusGraph(2.0),
            shuffle=True,
            device="cpu",
        )

        self.check(loader, 5)

    def test_longest(self, files):
        loader = auto_loader(
            files,
            batch_size=8,
            counts=[2, 6],
            epoch="longest",
            transform=AddRadiusGraph(2.0),
            shuffle=True,
            device="cpu",
        )

        batches = self.check(loader, 8)
        seen = {n for batch in batches for n in numbers(batch)}
        assert seen == set(range(10)) | set(range(100, 147))

    def test_filter_split_then_mix(self, files, crystals):
        """Half of the crystals, half of the molecules, one crystal for three molecules."""
        molecules, _ = random_split(HDF5Dataset(files[1]), [0.5, 0.5], seed=0)

        loader = auto_loader(
            [files[0], molecules],
            batch_size=4,
            counts=[1, 3],
            filter=[lambda structures: structures.properties["number"] % 2 == 1, None],
            shuffle=True,
            device="cpu",
        )

        batches = list(loader)
        seen = [n for batch in batches for n in numbers(batch)]

        assert len(batches) == 5  # 5 crystals kept, 23 molecules
        assert sorted(n for n in seen if n < 100) == [1, 3, 5, 7, 9]
        assert all(
            batch.pbc.all(dim=1).tolist() == [True, False, False, False]
            for batch in batches
        )

    def test_invalid(self, files, monkeypatch):
        with pytest.raises(ValueError, match="sum to the batch size"):
            auto_loader(files, batch_size=8, counts=[2, 2], device="cpu")
        with pytest.raises(ValueError, match="one count per source"):
            auto_loader(files, batch_size=8, counts=[8], device="cpu")
        with pytest.raises(ValueError, match="list of sources"):
            auto_loader(files[0], batch_size=8, counts=[8], device="cpu")

        monkeypatch.setattr(auto, "RAM_SHARE", 0.0)
        with pytest.raises(ValueError, match="do not fit"):
            auto_loader(files, batch_size=8, counts=[2, 6], device="cpu")
