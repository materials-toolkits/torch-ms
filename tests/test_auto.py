"""Loader configured for the hardware at hand."""

import sys
import types
from dataclasses import replace

import pytest
import torch

from torchms.data import (
    HDF5Dataset,
    InMemoryDataset,
    LoaderConfig,
    Structures,
    auto_config,
    auto_loader,
    detect_hardware,
)
from torchms.data import auto
from torchms.data.auto import Hardware, batch_candidates, probe_batch_size
from torchms.data.augmentation import LatticeRotation
from torchms.graph.transforms import AddRadiusGraph, AddTriplets

GB = 2**30
MB = 2**20
N = 23


@pytest.fixture
def structures() -> list[Structures]:
    generator = torch.Generator().manual_seed(0)

    return [
        Structures(
            lattice=(4.0 * torch.eye(3) + torch.rand(3, 3, generator=generator))[None],
            x=torch.rand(1 + i % 5, 3, generator=generator),
            z=torch.randint(1, 90, (1 + i % 5,), generator=generator),
            total_energy=[float(i)],
        )
        for i in range(N)
    ]


@pytest.fixture
def hdf5(tmp_path, structures) -> HDF5Dataset:
    hdf5_path = tmp_path / "test.hdf5"
    HDF5Dataset.from_structures(hdf5_path, structures)

    return HDF5Dataset(hdf5_path)


def gpu(total: float, free: float | None = None, ram: float = 64, cpus: int = 32):
    """Made-up machine with a GPU, sizes in GB."""
    free = total if free is None else free

    return Hardware(
        cpus=cpus,
        ram_available=int(ram * GB),
        device=torch.device("cuda"),
        gpu_name="made up",
        gpu_free=int(free * GB),
        gpu_total=int(total * GB),
    )


def no_gpu(ram: float = 16, cpus: int = 8) -> Hardware:
    return Hardware(cpus=cpus, ram_available=int(ram * GB), device=torch.device("cpu"))


def chain():
    add_graph, add_triplets = AddRadiusGraph(3.0, geometry=True), AddTriplets()

    return lambda structures: add_triplets(add_graph(structures))


class TestRules:
    def test_small_dataset_on_a_large_gpu(self):
        config = auto_config(27136, 10 * MB, 128, gpu(80))

        assert config.storage == "gpu"
        assert config.num_workers == 0
        assert config.graphs_on == "device"
        assert config.loader_batch_size == 8192
        assert config.batch_size == 128

    def test_small_gpu_builds_on_smaller_batches(self):
        config = auto_config(27136, 10 * MB, 128, gpu(8))

        assert config.storage == "gpu"
        assert config.loader_batch_size == 2048

    def test_dataset_too_large_for_the_gpu_goes_in_ram(self):
        # 4 GB is more than 20 % of the 8 GB of the GPU, less than half the RAM
        config = auto_config(5_000_000, 4 * GB, 128, gpu(8, ram=64))

        assert config.storage == "ram"
        assert config.num_workers == 0
        assert config.graphs_on == "device"

    def test_free_memory_counts_not_the_total(self):
        config = auto_config(271325, 204 * MB, 128, gpu(80, free=0.5))

        assert config.storage == "ram"

    def test_dataset_too_large_for_memory_stays_in_its_file(self):
        config = auto_config(5_000_000, 4 * GB, 128, gpu(8, ram=6, cpus=12))

        assert config.storage == "file"
        assert config.num_workers == 11
        assert config.graphs_on == "device"

    def test_workers_are_capped(self):
        config = auto_config(5_000_000, 4 * GB, 128, gpu(8, ram=6, cpus=64))

        assert config.num_workers == 16

    def test_unknown_ram_stays_in_its_file(self):
        hardware = replace(gpu(8), ram_available=None)

        assert auto_config(5_000_000, 4 * GB, 128, hardware).storage == "file"

    def test_without_gpu(self):
        config = auto_config(27136, 10 * MB, 128, no_gpu(cpus=8))

        assert config.storage == "ram"
        assert config.graphs_on == "workers"
        assert config.num_workers == 7
        assert config.loader_batch_size == config.batch_size == 128

    def test_without_gpu_and_one_core(self):
        config = auto_config(27136, 10 * MB, 128, no_gpu(cpus=1))

        assert config.num_workers == 0

    def test_storage_given(self):
        config = auto_config(27136, 10 * MB, 128, gpu(80), storage="file")

        assert config.storage == "file"
        assert config.num_workers == 16

    @pytest.mark.parametrize("batch_size", [1, 32, 100, 128, 300])
    @pytest.mark.parametrize("n_structures", [50, 1000, 27136])
    def test_large_batch_is_a_multiple(self, batch_size, n_structures):
        config = auto_config(n_structures, MB, batch_size, gpu(80))

        assert config.loader_batch_size % batch_size == 0
        assert batch_size <= config.loader_batch_size <= max(8192, batch_size)
        # not more than twice the dataset
        assert config.loader_batch_size < max(2 * n_structures, 2 * batch_size)

    def test_every_choice_has_a_reason(self):
        for hardware in (gpu(80), gpu(8, ram=6), no_gpu()):
            config = auto_config(27136, 10 * MB, 128, hardware)

            assert len(config.reasons) >= 2
            text = str(config)
            assert f"batches of {config.batch_size}" in text
            assert all(reason in text for reason in config.reasons)

    def test_candidates(self):
        assert batch_candidates(128, 27136) == [128, 256, 512, 1024, 2048, 4096, 8192]
        assert batch_candidates(128, 300) == [128, 256, 512]
        assert batch_candidates(128, 100) == [128]
        assert batch_candidates(10000, 10**6) == [10000]


class TestHardware:
    def test_detected(self):
        hardware = detect_hardware("cpu")

        assert hardware.cpus >= 1
        assert hardware.device == torch.device("cpu")
        assert hardware.gpu_name is None and hardware.gpu_free is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_gpu_detected(self):
        hardware = detect_hardware()

        assert hardware.device.type == "cuda"
        assert hardware.gpu_name
        assert 0 < hardware.gpu_free <= hardware.gpu_total

    def test_cores_of_the_job(self, monkeypatch):
        monkeypatch.setattr(auto.os, "sched_getaffinity", lambda pid: {0, 1, 2})

        assert detect_hardware("cpu").cpus == 3

    def files(self, tmp_path, monkeypatch, meminfo=None, limit=None, usage=None):
        path = tmp_path / "meminfo"
        if meminfo is not None:
            path.write_text(f"MemTotal: 99 kB\nMemAvailable:   {meminfo} kB\n")
        monkeypatch.setattr(auto, "MEMINFO", path)

        limit_file, usage_file = tmp_path / "limit", tmp_path / "usage"
        if limit is not None:
            limit_file.write_text(f"{limit}\n")
        if usage is not None:
            usage_file.write_text(f"{usage}\n")
        monkeypatch.setattr(auto, "CGROUP_MEMORY", ((limit_file, usage_file),))

    def test_ram_of_the_machine(self, tmp_path, monkeypatch):
        self.files(tmp_path, monkeypatch, meminfo=1000)

        assert auto.available_ram() == 1000 * 1024

    def test_ram_of_the_job(self, tmp_path, monkeypatch):
        self.files(tmp_path, monkeypatch, meminfo=10**8, limit=64 * GB, usage=4 * GB)

        assert auto.available_ram() == 60 * GB

    def test_job_without_limit(self, tmp_path, monkeypatch):
        self.files(tmp_path, monkeypatch, meminfo=1000, limit="max", usage=5)

        assert auto.available_ram() == 1000 * 1024

    def test_unknown_ram(self, tmp_path, monkeypatch):
        self.files(tmp_path, monkeypatch)

        assert auto.available_ram() is None
        assert detect_hardware("cpu").ram_available is None


class TestSource:
    def test_path(self, hdf5):
        loader = auto_loader(hdf5.hdf5_file, batch_size=4, device="cpu")

        assert len(loader.dataset) == N

    def test_datasets(self, hdf5):
        for source in (hdf5, InMemoryDataset.from_hdf5(hdf5)):
            loader = auto_loader(source, batch_size=4, device="cpu")
            assert sum(len(batch.num_atoms) for batch in loader) == N

    def test_dataset_in_memory_stays_where_it_is(self, hdf5):
        loader = auto_loader(
            InMemoryDataset.from_hdf5(hdf5), batch_size=4, device="cpu"
        )

        assert loader.config.storage == "ram"
        assert "given" in loader.config.reasons[0]

    def test_published_dataset(self, hdf5, monkeypatch):
        calls = []

        def download(repo_id, filename, repo_type=None, **kwargs):
            calls.append((repo_id, filename, repo_type, kwargs))
            return hdf5.hdf5_file

        hub = types.ModuleType("huggingface_hub")
        setattr(hub, "hf_hub_download", download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        loader = auto_loader("mp-20", "test", batch_size=4, device="cpu", revision="v1")
        again = auto_loader("MP_20/test", batch_size=4, device="cpu")

        assert len(loader.dataset) == len(again.dataset) == N
        assert calls[0] == (
            "materials-toolkits/mp-20",
            "test.hdf5",
            "dataset",
            {"revision": "v1"},
        )
        assert calls[1][:2] == ("materials-toolkits/mp-20", "test.hdf5")

    def test_unknown_name(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            auto_loader("not-a-dataset", batch_size=4, device="cpu")

    def test_dataset_with_a_transform(self, hdf5):
        hdf5.transform = LatticeRotation()

        with pytest.raises(ValueError, match="augment"):
            auto_loader(hdf5, batch_size=4, device="cpu")

    def test_split_of_a_file(self, hdf5):
        with pytest.raises(ValueError, match="split"):
            auto_loader(hdf5.hdf5_file, "train", batch_size=4, device="cpu")

    def test_batch_size(self, hdf5):
        with pytest.raises(ValueError, match="positive"):
            auto_loader(hdf5, batch_size=0, device="cpu")


def configs(device: str) -> list[LoaderConfig]:
    """One configuration per way of loading, on a device."""
    target = torch.device(device)
    made = [
        LoaderConfig(target, "ram", 4, 4, 0, "workers"),
        LoaderConfig(target, "file", 4, 4, 0, "workers"),
        LoaderConfig(target, "file", 4, 4, 2, "workers"),
        LoaderConfig(target, "ram", 4, 8, 0, "device"),
        LoaderConfig(target, "file", 4, 8, 0, "device"),
        LoaderConfig(target, "file", 4, 8, 2, "device"),
        LoaderConfig(target, "ram", 4, 4, 0, "device"),
    ]
    if device == "cuda":
        made.append(LoaderConfig(target, "gpu", 4, 8, 0, "device"))

    return made


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
WAYS = [(device, i) for device in DEVICES for i in range(len(configs(device)))]


@pytest.fixture(params=WAYS, ids=lambda way: f"{way[0]}-{way[1]}")
def config(request) -> LoaderConfig:
    device, i = request.param

    return configs(device)[i]


@pytest.mark.timeout(120)
class TestLoader:
    def test_every_structure_once_with_its_graph(self, hdf5, structures, config):
        loader = auto_loader(hdf5, batch_size=4, transform=chain(), config=config)
        batches = list(loader)

        assert len(batches) == len(loader) == 6
        assert [len(batch.num_atoms) for batch in batches] == [4, 4, 4, 4, 4, 3]
        assert loader.config is config

        for k, batch in enumerate(batches):
            expected = chain()(Structures.collate(structures[4 * k : 4 * k + 4]))

            assert batch.device().type == config.device.type
            assert batch.graph.device().type == config.device.type
            assert torch.equal(batch.x.cpu(), expected.x)
            assert torch.equal(batch.graph.edge_index.cpu(), expected.graph.edge_index)
            assert torch.equal(batch.graph.edge_ptr.cpu(), expected.graph.edge_ptr)
            assert torch.equal(batch.triplets.index.cpu(), expected.triplets.index)
            assert torch.allclose(
                batch.graph.edge_distance.cpu(),
                expected.graph.edge_distance,
                atol=1e-5,
            )

    def test_without_transform(self, hdf5, config):
        loader = auto_loader(hdf5, batch_size=4, config=config)

        energy = torch.cat([batch.total_energy for batch in loader])
        assert energy.tolist() == list(range(N))
        assert all(batch.graph is None for batch in loader)

    def test_drop_last(self, hdf5, config):
        loader = auto_loader(hdf5, batch_size=4, config=config, drop_last=True)

        assert len(loader) == 5
        assert [len(batch.num_atoms) for batch in loader] == [4] * 5

    def test_shuffle(self, hdf5, config):
        generator = torch.Generator().manual_seed(0)
        loader = auto_loader(
            hdf5, batch_size=4, config=config, shuffle=True, generator=generator
        )

        first = torch.cat([batch.total_energy for batch in loader]).tolist()
        second = torch.cat([batch.total_energy for batch in loader]).tolist()

        assert sorted(first) == sorted(second) == list(range(N))
        assert first != list(range(N))
        assert first != second

    def test_same_seed_same_order(self, hdf5, config):
        orders = []
        for _ in range(2):
            loader = auto_loader(
                hdf5,
                batch_size=4,
                config=config,
                shuffle=True,
                generator=torch.Generator().manual_seed(1),
            )
            orders.append(torch.cat([batch.total_energy for batch in loader]).tolist())

        assert orders[0] == orders[1]

    def test_augmentations_before_the_graphs(self, hdf5, structures, config):
        loader = auto_loader(
            hdf5,
            batch_size=4,
            config=config,
            transform=AddRadiusGraph(3.0, geometry=True),
            augment=LatticeRotation(),
        )
        batches = list(loader)

        original = Structures.collate(structures[:4])
        first = batches[0]
        # rotated cells, whose graph is the one of the rotated cells
        assert not torch.allclose(first.lattice.cpu(), original.lattice, atol=1e-3)
        assert torch.allclose(
            first.lattice.norm(dim=-1).cpu(), original.lattice.norm(dim=-1), atol=1e-4
        )
        assert first.graph is not None
        assert torch.allclose(
            first.graph.edge_distance, first.graph.edges_distances(), atol=1e-5
        )

    def test_where_the_dataset_is(self, hdf5, config):
        loader = auto_loader(hdf5, batch_size=4, config=config)

        if config.storage == "file":
            assert isinstance(loader.dataset, HDF5Dataset)
        else:
            assert isinstance(loader.dataset, InMemoryDataset)
            expected = "cuda" if config.storage == "gpu" else "cpu"
            assert loader.dataset.device().type == expected


class TestConfig:
    def test_detected_without_gpu(self, hdf5):
        loader = auto_loader(hdf5, batch_size=4, transform=chain(), device="cpu")

        assert loader.config.device == torch.device("cpu")
        assert loader.config.graphs_on == "workers"
        assert loader.config.loader_batch_size == 4
        assert loader.hardware.gpu_name is None
        assert sum(len(batch.num_atoms) for batch in loader) == N

    def test_config_of_another_loader(self, hdf5):
        loader = auto_loader(hdf5, batch_size=4, transform=chain(), device="cpu")
        config = replace(loader.config, storage="file", num_workers=0)

        again = auto_loader(hdf5, batch_size=4, transform=chain(), config=config)

        assert again.config.storage == "file"
        assert isinstance(again.dataset, HDF5Dataset)
        assert sum(len(batch.num_atoms) for batch in again) == N

    def test_large_batch_must_be_a_multiple(self, hdf5):
        config = LoaderConfig(torch.device("cpu"), "ram", 4, 6, 0, "device")

        with pytest.raises(ValueError, match="multiple"):
            auto_loader(hdf5, batch_size=4, config=config)

    def test_dataset_in_memory_has_no_file(self, hdf5):
        config = LoaderConfig(torch.device("cpu"), "file", 4, 4, 0, "workers")

        with pytest.raises(ValueError, match="no file"):
            auto_loader(InMemoryDataset.from_hdf5(hdf5), batch_size=4, config=config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
class TestProbe:
    @staticmethod
    def sample(n: int) -> Structures:
        return Structures.collate([Structures.diamond()] * n).to("cuda")

    def test_chooses_a_candidate(self):
        candidates = [4, 8, 16, 32, 64]
        chosen, lines = probe_batch_size(
            self.sample(64), AddRadiusGraph(3.0), 4, candidates, budget=GB
        )

        assert chosen in candidates
        assert len(lines) == len(candidates)
        assert all("structures/s" in line for line in lines)

    def test_memory_budget(self):
        # the graphs of 64 diamonds take more than 100 kB
        chosen, lines = probe_batch_size(
            self.sample(64), AddRadiusGraph(3.0), 4, [4, 64], budget=100_000
        )

        assert chosen == 4
        assert "over the" in lines[-1]

    def test_sample_smaller_than_the_candidates(self):
        chosen, lines = probe_batch_size(
            self.sample(8), AddRadiusGraph(3.0), 4, [4, 8, 16, 32], budget=GB
        )

        assert chosen in (4, 8)
        assert len(lines) == 2

    def test_detected_with_a_gpu(self, hdf5):
        loader = auto_loader(hdf5, batch_size=4, transform=chain(), shuffle=True)
        config = loader.config

        assert config.device.type == "cuda"
        assert config.storage == "gpu"
        assert config.graphs_on == "device"
        assert config.loader_batch_size % 4 == 0
        assert config.loader_batch_size in batch_candidates(4, N)
        assert any("structures/s" in reason for reason in config.reasons)

        batches = list(loader)
        assert sum(len(batch.num_atoms) for batch in batches) == N
        assert all(batch.graph.device().type == "cuda" for batch in batches)

    def test_without_probe(self, hdf5):
        loader = auto_loader(hdf5, batch_size=4, transform=chain(), probe=False)

        assert not any("structures/s" in reason for reason in loader.config.reasons)
        assert sum(len(batch.num_atoms) for batch in loader) == N
