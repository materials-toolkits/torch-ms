"""Loader configured for the hardware at hand.

The fastest way to give batches of structures with their graphs depends on the
machine. auto_loader detects the CPU cores, the RAM and the GPU, then chooses:

* where the dataset lives: in the memory of the GPU, in RAM, or in its file
  read by workers;
* the size of the batches the graphs are built on, larger than the batches of
  the training and split after: a GPU builds the graphs of one large batch
  much faster than the ones of many small batches;
* the number of workers.

The first choices follow rules on what is detected. The size of the large
batches is measured, by building the graphs of a sample at a few sizes.
"""

import math
import os
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .batching import Structures
from .datasets import load_dataset
from .hdf5 import HDF5Dataset
from .loader import OnDevice
from .memory import InMemoryDataset

Transform = Callable[[Structures], Structures]
Storage = Literal["gpu", "ram", "file"]

# bytes of a dataset in memory, as Structures, without its named properties
BYTES_PER_STRUCTURE = 71
BYTES_PER_ATOM = 28

# share of the free memory a dataset can take, and the graphs of a large batch
GPU_SHARE = 0.20
RAM_SHARE = 0.50
BATCH_SHARE = 0.25

# more workers do not read faster, larger batches do not build graphs faster
MAX_WORKERS = 16
MAX_BATCH_SIZE = 8192

MEMINFO = Path("/proc/meminfo")
# the cgroup of this process, and where the limits of the cgroups are
PROC_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class Hardware:
    """What a loader can use on this machine.

    Attributes:
        cpus: CPU cores this process can use.
        ram_available: Bytes of RAM available, None if unknown.
        device: Device of the batches.
        gpu_name: Name of the GPU, None without GPU.
        gpu_free: Bytes of GPU memory free, None without GPU.
        gpu_total: Bytes of GPU memory, None without GPU.
    """

    cpus: int
    ram_available: int | None
    device: torch.device
    gpu_name: str | None = None
    gpu_free: int | None = None
    gpu_total: int | None = None


def read_bytes(path: Path) -> int | None:
    """Number written in a file of the system, None if it is not a number."""
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def cgroup_memory() -> list[tuple[Path, Path]]:
    """Files of the limit and of the usage of the memory of this process.

    The cgroup of the process and the ones above it, each of which can have a
    limit, for the two versions of cgroups.
    """
    files = []

    try:
        lines = PROC_CGROUP.read_text(encoding="ascii").splitlines()
    except OSError:
        lines = []

    for line in lines:
        _, controllers, path = line.split(":", maxsplit=2)

        if controllers == "":
            root, limit, usage = CGROUP_ROOT, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            root = CGROUP_ROOT / "memory"
            limit, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue

        directory = root / path.strip("/")
        while True:
            files.append((directory / limit, directory / usage))
            if directory == root or root not in directory.parents:
                break
            directory = directory.parent

    return files


def available_ram() -> int | None:
    """Bytes of RAM this process can still use, None if unknown.

    The memory available on the machine, and no more than what is left of the
    limit of the job: the one of its cgroup, or the memory asked to Slurm.
    """
    available = None

    try:
        for line in MEMINFO.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    limits = []
    for limit_file, usage_file in cgroup_memory():
        limit, usage = read_bytes(limit_file), read_bytes(usage_file)
        # a cgroup without limit has "max", or a number larger than any memory
        if limit is not None and usage is not None and limit < 2**60:
            limits.append(max(limit - usage, 0))

    # Slurm does not always put the memory of a job in a cgroup
    slurm = os.environ.get("SLURM_MEM_PER_NODE", "")
    if slurm.isdigit():
        limits.append(int(slurm) * 2**20)

    for left in limits:
        available = left if available is None else min(available, left)

    return available


def detect_hardware(device: torch.device | str | None = None) -> Hardware:
    """Detect the CPU cores, the RAM and the GPU a loader can use.

    Args:
        device: Device of the batches. By default the GPU if there is one.

    Returns:
        The hardware detected. The cores are the ones of the job under Slurm,
        not the ones of the node.

    Examples:
        >>> hardware = detect_hardware("cpu")
        >>> hardware.cpus >= 1, hardware.gpu_name
        (True, None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1

    if device.type != "cuda":
        return Hardware(cpus=cpus, ram_available=available_ram(), device=device)

    free, total = torch.cuda.mem_get_info(device)

    return Hardware(
        cpus=cpus,
        ram_available=available_ram(),
        device=device,
        gpu_name=torch.cuda.get_device_name(device),
        gpu_free=free,
        gpu_total=total,
    )


@dataclass
class LoaderConfig:
    """How a dataset is loaded, see auto_loader.

    Attributes:
        device: Device of the batches.
        storage: Where the dataset lives: "gpu" in the memory of the GPU,
            "ram" in RAM, "file" in its file read by workers.
        batch_size: Number of structures of the batches given.
        loader_batch_size: Number of structures of the batches read and
            transformed, a multiple of batch_size. They are split in batches
            of batch_size.
        num_workers: Number of workers of the DataLoader, 0 for none.
        graphs_on: Where the transform runs: "device" on the device of the
            batches, "workers" in the workers of the DataLoader.
        reasons: Why each choice was made, one line per choice.

    Examples:
        >>> config = LoaderConfig(torch.device("cpu"), "ram", 32, 32, 4, "workers")
        >>> print(config)
        device cpu, dataset in RAM, 4 workers
        batches of 32, built by 32 in the workers
    """

    device: torch.device
    storage: Storage
    batch_size: int
    loader_batch_size: int
    num_workers: int
    graphs_on: Literal["device", "workers"]
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        place = {"gpu": "on the GPU", "ram": "in RAM", "file": "in its file"}
        workers = f"{self.num_workers} worker" + ("s" if self.num_workers != 1 else "")
        where = "on the device" if self.graphs_on == "device" else "in the workers"
        lines = [
            f"device {self.device}, dataset {place[self.storage]}, {workers}",
            f"batches of {self.batch_size}, built by {self.loader_batch_size} {where}",
        ]

        return "\n".join(lines + [f"- {reason}" for reason in self.reasons])


def megabytes(size: float) -> str:
    """A size in bytes, written in MB or GB."""
    if size >= 2**30:
        return f"{size / 2**30:.1f} GB"

    return f"{size / 2**20:.0f} MB"


def batch_candidates(
    batch_size: int, n_structures: int, limit: int = MAX_BATCH_SIZE
) -> list[int]:
    """Sizes of large batches to choose from: batch_size times a power of two.

    Args:
        batch_size: Number of structures of the batches given.
        n_structures: Number of structures of the dataset.
        limit: Largest size.

    Returns:
        Sizes in ascending order, the first one being batch_size. A size is not
        larger than the limit, nor than twice the dataset.

    Examples:
        >>> batch_candidates(128, 27136)
        [128, 256, 512, 1024, 2048, 4096, 8192]
        >>> batch_candidates(100, 1000, limit=500)
        [100, 200, 400]
    """
    sizes = [batch_size]
    while 2 * sizes[-1] <= limit and sizes[-1] < n_structures:
        sizes.append(2 * sizes[-1])

    return sizes


def auto_config(
    n_structures: int,
    dataset_bytes: int,
    batch_size: int,
    hardware: Hardware,
    storage: Storage | None = None,
) -> LoaderConfig:
    """Choose how to load a dataset from its size and from the hardware.

    Rules from measures on a desktop with a small GPU and on a node with an
    H100. The size of the large batches can then be measured, see
    probe_batch_size.

    Args:
        n_structures: Number of structures of the dataset.
        dataset_bytes: Bytes of the dataset in memory.
        batch_size: Number of structures of the batches given.
        hardware: What the loader can use, see detect_hardware.
        storage: Where the dataset lives, if it is already decided.

    Returns:
        The configuration, with the reason of each choice.

    Examples:
        A dataset of 10 MB on a GPU of 80 GB: in the memory of the GPU.

        >>> hardware = Hardware(
        ...     cpus=32, ram_available=64 * 2**30, device=torch.device("cuda"),
        ...     gpu_name="H100", gpu_free=80 * 2**30, gpu_total=80 * 2**30,
        ... )
        >>> config = auto_config(27136, 10 * 2**20, 128, hardware)
        >>> config.storage, config.loader_batch_size, config.num_workers
        ('gpu', 8192, 0)

        Without GPU, the graphs are built in the workers, by small batches.

        >>> hardware = Hardware(8, 16 * 2**30, torch.device("cpu"))
        >>> config = auto_config(27136, 10 * 2**20, 128, hardware)
        >>> config.storage, config.loader_batch_size, config.num_workers
        ('ram', 128, 7)
    """
    reasons = []
    gpu = hardware.device.type == "cuda"
    size = megabytes(dataset_bytes)

    if storage is not None:
        reasons.append(f"dataset {storage}: given")
    elif (
        gpu
        and hardware.gpu_free is not None
        and dataset_bytes <= GPU_SHARE * hardware.gpu_free
    ):
        storage = "gpu"
        reasons.append(
            f"dataset on the GPU: {size} is at most {GPU_SHARE:.0%} of the "
            f"{megabytes(hardware.gpu_free)} free on it"
        )
    elif (
        hardware.ram_available is not None
        and dataset_bytes <= RAM_SHARE * hardware.ram_available
    ):
        storage = "ram"
        reasons.append(
            f"dataset in RAM: {size} is at most {RAM_SHARE:.0%} of the "
            f"{megabytes(hardware.ram_available)} available"
            + (", and too large for the GPU" if gpu else "")
        )
    else:
        storage = "file"
        reasons.append(f"dataset in its file: {size} does not fit in memory")

    workers = max(min(hardware.cpus - 1, MAX_WORKERS), 0)

    if not gpu:
        reasons.append(
            f"no GPU: graphs built in {workers} workers ({hardware.cpus} cores), "
            "by batches of the size given"
        )
        return LoaderConfig(
            hardware.device,
            storage,
            batch_size,
            batch_size,
            workers,
            "workers",
            reasons,
        )

    if storage == "file":
        reasons.append(f"{workers} workers read the file ({hardware.cpus} cores)")
    else:
        workers = 0
        reasons.append("no worker: the batches are taken from memory")

    # a small GPU is busy before the largest batches
    limit = MAX_BATCH_SIZE
    if hardware.gpu_total is not None and hardware.gpu_total < 16 * 2**30:
        limit = MAX_BATCH_SIZE // 4
    large = batch_candidates(batch_size, n_structures, limit)[-1]
    reasons.append(
        f"graphs built on the GPU by batches of {large}, split in batches of "
        f"{batch_size}: default for a GPU of "
        f"{megabytes(hardware.gpu_total or 0)}"
    )

    return LoaderConfig(
        hardware.device, storage, batch_size, large, workers, "device", reasons
    )


def probe_batch_size(
    sample: Structures,
    transform: Transform,
    batch_size: int,
    candidates: list[int],
    budget: float,
) -> tuple[int, list[str]]:
    """Measure the best size of the large batches for a transform on a GPU.

    The transform is run on batches of each size then split in batches of
    batch_size, as the loader does. The smallest size within 10 % of the best
    time per structure is kept, among the ones whose memory fits the budget:
    less memory and less delay for the same rate.

    Args:
        sample: Structures of the dataset on the GPU, at least as many as the
            largest candidate to measure.
        transform: Transform building the graphs.
        batch_size: Number of structures of the batches given.
        candidates: Sizes to measure, in ascending order.
        budget: Bytes of GPU memory the graphs of a large batch can take.

    Returns:
        The size chosen, and what was measured, one line per size.
    """
    device = sample.device()
    measured: list[tuple[int, float]] = []
    lines = []

    for size in candidates:
        if size > len(sample.num_atoms):
            break

        part = sample.split(size)[0]
        try:
            transform(part).split(batch_size)
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)

            times = []
            for _ in range(3):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                transform(part).split(batch_size)
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            lines.append(f"batches of {size}: out of memory")
            break

        memory = torch.cuda.max_memory_allocated(device) - before
        per_structure = statistics.median(times) / size
        line = (
            f"batches of {size}: {1 / per_structure:,.0f} structures/s, "
            f"{megabytes(memory)}"
        )

        if memory > budget:
            lines.append(line + f", over the {megabytes(budget)} allowed")
            break

        lines.append(line)
        measured.append((size, per_structure))

    if not measured:
        return batch_size, lines

    best = min(per_structure for _, per_structure in measured)
    chosen = next(size for size, value in measured if value <= 1.1 * best)

    return chosen, lines


# a published dataset by its name, a file or a dataset
Source = str | os.PathLike | HDF5Dataset | InMemoryDataset


def resolve_source(
    source: Source,
    split: str | None = None,
    **hub: Any,
) -> HDF5Dataset | InMemoryDataset:
    """Dataset of a source: a published dataset, a file or a dataset.

    Args:
        source: Name of a published dataset ("mp-20", or "mp-20/train"), path
            of an HDF5 file, or a dataset without transform.
        split: Split of a published dataset, its first one by default.
        **hub: Forwarded to huggingface_hub.hf_hub_download for a published
            dataset (revision, token, cache_dir, ...).

    Returns:
        The dataset of the source. A published dataset is downloaded on the
        first call.

    Raises:
        KeyError: If the source is neither a file nor a published dataset.
        ValueError: If the dataset has a transform, or a split is given for a
            source which is not a published dataset.
    """
    if isinstance(source, (HDF5Dataset, InMemoryDataset)):
        if split is not None:
            raise ValueError("a split is only given with the name of a dataset")

        if source.transform is not None:
            raise ValueError(
                "the dataset has a transform: give it to auto_loader instead, as "
                "augment or transform, which runs it where it is fastest"
            )

        return source

    if os.path.isfile(source):
        if split is not None:
            raise ValueError("a split is only given with the name of a dataset")

        return HDF5Dataset(source)

    return load_dataset(os.fspath(source), split, **hub)


def dataset_size(dataset: HDF5Dataset | InMemoryDataset) -> int:
    """Bytes of a dataset in memory, without reading it."""
    if isinstance(dataset, HDF5Dataset):
        n_atoms = dataset.dataset("x").shape[0]
        n_properties = len(dataset.property_names())
    else:
        n_atoms = len(dataset.structures.z)
        n_properties = len(dataset.structures.properties)

    per_structure = BYTES_PER_STRUCTURE + 4 * n_properties

    return per_structure * len(dataset) + BYTES_PER_ATOM * n_atoms


def resolve_sources(
    sources: Sequence[Source],
    batch_size: int,
    hardware: Hardware,
    storage: Storage | None = None,
    **hub: Any,
) -> InMemoryDataset:
    """One dataset in memory from several sources, see resolve_source.

    Args:
        sources: Published datasets ("mp-20/train"), files or datasets.
        batch_size: Number of structures of the batches given.
        hardware: What the loader can use: the dataset is put on the GPU if
            it fits there, in RAM otherwise.
        storage: Where the dataset lives, if it is already decided.
        **hub: Forwarded to huggingface_hub.hf_hub_download.

    Raises:
        ValueError: If there is no source, or if the datasets do not fit
            together in memory.
    """
    if len(sources) == 0:
        raise ValueError("there must be at least one source")

    datasets = [resolve_source(source, **hub) for source in sources]

    if storage is None:
        n_structures = sum(len(dataset) for dataset in datasets)
        size = sum(dataset_size(dataset) for dataset in datasets)
        storage = auto_config(n_structures, size, batch_size, hardware).storage

    if storage == "file":
        raise ValueError(
            "several datasets are loaded together in memory, where these ones "
            "do not fit"
        )

    device = hardware.device if storage == "gpu" else torch.device("cpu")

    return InMemoryDataset.concat(
        [
            (
                InMemoryDataset.from_hdf5(dataset, device)
                if isinstance(dataset, HDF5Dataset)
                else dataset.to(device)
            )
            for dataset in datasets
        ]
    )


def chain(*transforms: Transform | None) -> Transform | None:
    """One transform applying the ones given in order, None if there is none."""
    steps = [transform for transform in transforms if transform is not None]

    if len(steps) == 0:
        return None

    if len(steps) == 1:
        return steps[0]

    def apply(structures: Structures) -> Structures:
        for step in steps:
            structures = step(structures)

        return structures

    return apply


class AutoLoader:
    """Batches of a dataset, loaded as its configuration says.

    Made by auto_loader. It is iterated once per epoch.

    Attributes:
        config: How the dataset is loaded, and why.
        hardware: The hardware detected.
        dataset: The dataset the batches come from: an InMemoryDataset when it
            is in memory, an HDF5Dataset when it is read from its file.
    """

    def __init__(
        self,
        source: HDF5Dataset | InMemoryDataset,
        config: LoaderConfig,
        hardware: Hardware,
        *,
        transform: Transform | None = None,
        augment: Transform | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ):
        if config.loader_batch_size % config.batch_size != 0:
            raise ValueError("loader_batch_size must be a multiple of batch_size")

        self.config = config
        self.hardware = hardware
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.length = len(source)

        self.dataset: HDF5Dataset | InMemoryDataset
        self.loader: DataLoader | None = None
        self.transform = chain(augment, transform)

        if config.graphs_on == "workers":
            # everything runs in the workers of a DataLoader
            in_workers, self.transform = self.transform, None
        elif config.storage == "file":
            # augmentations are cheap in the workers, graphs are not
            in_workers, self.transform = augment, transform
        else:
            in_workers = None

        if config.storage == "file":
            if not isinstance(source, HDF5Dataset):
                raise ValueError("a dataset in memory has no file to read")
            self.dataset = HDF5Dataset(source.hdf5_file, transform=in_workers)
        else:
            device = config.device if config.storage == "gpu" else torch.device("cpu")
            if isinstance(source, HDF5Dataset):
                source = InMemoryDataset.from_hdf5(source, device)
            elif source.device() != device:
                source = source.to(device)
            self.dataset = InMemoryDataset(source.structures, transform=in_workers)

        if config.storage == "file" or config.graphs_on == "workers":
            self.loader = DataLoader(
                self.dataset,
                batch_size=config.loader_batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
                generator=generator,
            )

    def batches(self) -> Iterable[Structures]:
        """Batches of an epoch as they are read, before the device."""
        if self.loader is not None:
            return self.loader

        assert isinstance(self.dataset, InMemoryDataset)

        return self.dataset.batches(
            self.config.loader_batch_size, self.shuffle, generator=self.generator
        )

    def __iter__(self) -> Iterator[Structures]:
        config = self.config
        size = None
        if config.loader_batch_size != config.batch_size:
            size = config.batch_size

        for structures in OnDevice(self.batches(), config.device, self.transform, size):
            if self.drop_last and len(structures.num_atoms) < config.batch_size:
                continue

            yield structures

    def __len__(self) -> int:
        """Number of batches of an epoch."""
        if self.drop_last:
            return self.length // self.config.batch_size

        return math.ceil(self.length / self.config.batch_size)


def detect_config(
    dataset: HDF5Dataset | InMemoryDataset,
    batch_size: int,
    hardware: Hardware,
    transform: Transform | None = None,
) -> LoaderConfig:
    """Configuration of a dataset on some hardware, see auto_loader.

    Args:
        dataset: The dataset to load. One in memory stays where it is.
        batch_size: Number of structures of the batches given.
        hardware: What the loader can use.
        transform: Transform building the graphs. With a GPU, the size of the
            large batches is measured with it; without transform it comes
            from the size of the memory of the GPU.
    """
    given: Storage | None = None
    if isinstance(dataset, InMemoryDataset):
        given = "gpu" if dataset.device().type == "cuda" else "ram"

    size = dataset_size(dataset)
    config = auto_config(len(dataset), size, batch_size, hardware, given)

    if transform is None or config.graphs_on != "device":
        return config

    candidates = batch_candidates(batch_size, len(dataset))
    count = min(candidates[-1], len(dataset))
    index = np.sort(np.random.default_rng(0).choice(len(dataset), count, False))

    free = hardware.gpu_free or 0
    if config.storage == "gpu":
        free -= size
    chosen, lines = probe_batch_size(
        dataset[index].to(config.device),
        transform,
        batch_size,
        candidates,
        BATCH_SHARE * free,
    )

    reasons = config.reasons[:-1] + [
        f"graphs built on the GPU by batches of {chosen}, split in batches "
        f"of {batch_size}: the smallest within 10 % of the fastest measured"
    ]

    return replace(config, loader_batch_size=chosen, reasons=reasons + lines)


def auto_loader(  # pylint: disable=too-many-arguments
    source: Source | list[Source] | tuple[Source, ...],
    split: str | None = None,
    *,
    batch_size: int,
    transform: Transform | None = None,
    augment: Transform | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
    device: torch.device | str | None = None,
    probe: bool = True,
    config: LoaderConfig | None = None,
    generator: torch.Generator | None = None,
    **hub: Any,
) -> AutoLoader:
    """Loader of a dataset configured for the hardware at hand.

    Detects the CPU cores, the RAM and the GPU, chooses where the dataset
    lives, the size of the batches the graphs are built on and the number of
    workers, and gives the batches on the device with their graphs.

    Args:
        source: Name of a published dataset ("mp-20", or "mp-20/train"), path
            of an HDF5 file, or a dataset without transform. Or a list of
            them, ["mp-20/train", "qm9"] for instance: the datasets are loaded
            together in memory, and shuffled batches mix their structures,
            crystals and molecules.
        split: Split of a published dataset, its first one by default.
        batch_size: Number of structures of the batches given.
        transform: Transform building the graphs (torchms.graph.transforms),
            run where it is fastest: on the GPU if there is one, on large
            batches which are then split.
        augment: Augmentations (torchms.data.augmentation), run before the
            transform.
        shuffle: If True, the structures are drawn in a random order at each
            epoch.
        drop_last: If True, a last batch smaller than batch_size is not given.
        device: Device of the batches. By default the GPU if there is one.
        probe: If True and there is a GPU, the size of the large batches is
            measured on a sample, in about a second. If False it comes from
            the size of the memory of the GPU.
        config: Configuration to use, instead of detecting one. For instance
            the one of another loader, modified with dataclasses.replace.
        generator: Random number generator of the order of the structures.
        **hub: Forwarded to huggingface_hub.hf_hub_download for a published
            dataset (revision, token, cache_dir, ...).

    Returns:
        A loader to iterate once per epoch. Its config says what was chosen
        and why.

    Raises:
        KeyError: If the source is neither a file nor a published dataset.
        ValueError: If the dataset has a transform, or if the datasets of a
            list do not fit together in memory.

    Examples:
        >>> import tempfile
        >>> from torchms.graph.transforms import AddRadiusGraph
        >>> batch = Structures.collate([Structures.diamond(), Structures.zincblende()])
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     file = os.path.join(directory, "structures.hdf5")
        ...     HDF5Dataset.from_structures(file, [batch] * 5)
        ...     loader = auto_loader(
        ...         file, batch_size=4, transform=AddRadiusGraph(3.0), device="cpu"
        ...     )
        ...     len(loader), [len(s.graph.num_edges) for s in loader]
        (3, [4, 4, 2])
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    hardware = detect_hardware(device if config is None else config.device)

    dataset: HDF5Dataset | InMemoryDataset
    if isinstance(source, (list, tuple)):
        if split is not None:
            raise ValueError("the split of each dataset of a list is given in its name")

        dataset = resolve_sources(
            source,
            batch_size,
            hardware,
            None if config is None else config.storage,
            **hub,
        )
    else:
        dataset = resolve_source(source, split, **hub)

    if config is None:
        config = detect_config(
            dataset, batch_size, hardware, transform if probe else None
        )

    return AutoLoader(
        dataset,
        config,
        hardware,
        transform=transform,
        augment=augment,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )
