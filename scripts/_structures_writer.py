"""Write structures read from a source, crystals or molecules, to an HDF5 file.

Shared by the scripts building the datasets of molecules and of MOFs. Their
structures do not go through CIF files: a molecule has no cell, and the labels
of these datasets are not the four ones of the CIF tags.
"""

import hashlib
import os
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tqdm

from _cif_writer import cell_parameters
from torchms.data.batching import Structures
from torchms.data.cif import CifParser
from torchms.data.hdf5 import HDF5Dataset

LABELS = ("total_energy", "formation_energy_per_atom", "energy_above_hull", "band_gap")


def check(condition: bool, message: str) -> None:
    """Stop the build when a verification fails."""
    if not condition:
        raise RuntimeError(f"verification failed: {message}")


def download(url: str, destination: Path, md5: str | None = None) -> None:
    """Download a file, skipped if it already exists, and check its MD5."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")

        with tqdm.tqdm(desc=f"downloading {url}", unit="B", unit_scale=True) as bar:

            def update(blocks: int, block_size: int, total_size: int) -> None:
                bar.total = total_size
                bar.update(min(blocks * block_size, total_size) - bar.n)

            urllib.request.urlretrieve(url, partial, reporthook=update)

        os.replace(partial, destination)

    if md5 is None:
        return

    digest = hashlib.md5()
    with open(destination, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    check(digest.hexdigest() == md5, f"{destination} is not the file of the source")


@dataclass
class Entry:
    """A structure of a source.

    Attributes:
        numbers: Atomic numbers, shape (n_atoms,).
        coords: Fractional coordinates of a crystal, positions in Å of a
            molecule, shape (n_atoms, 3), float64.
        lattice: Lattice vectors of a crystal as rows, shape (3, 3), in Å.
            None for a molecule.
        labels: Values of the labels of Structures (LABELS), the others are NaN.
        properties: Values of the named properties, by name.
    """

    numbers: np.ndarray
    coords: np.ndarray
    lattice: np.ndarray | None = None
    labels: dict[str, float] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)


def standard_lattice(lattice: np.ndarray) -> torch.Tensor:
    """Lattice in the standard orientation of the datasets, first vector along x.

    It is rebuilt from the lengths and the angles of the cell, as the lattice
    of a structure read from a CIF file. Fractional coordinates are unchanged.
    """
    lengths, angles = cell_parameters(np.asarray(lattice, dtype=np.float64))
    names = ("a", "b", "c", "alpha", "beta", "gamma")
    cell = dict(zip(names, [*lengths.tolist(), *angles.tolist()]))

    # pylint: disable-next=protected-access
    return CifParser()._build_lattice_vectors(cell)


def as_structures(entries: list[Entry], names: tuple[str, ...]) -> Structures:
    """Batch of the structures of some entries.

    The fractional coordinates of a crystal are wrapped into its cell. A
    molecule has the identity as lattice and its positions as coordinates.

    Args:
        entries: Entries of a source.
        names: Names of the properties of the dataset. An entry without one of
            them has NaN.
    """
    lattices, coords, pbc = [], [], []

    for entry in entries:
        if entry.lattice is None:
            lattices.append(torch.eye(3, dtype=torch.float32))
            coords.append(np.asarray(entry.coords, dtype=np.float64))
        else:
            lattices.append(standard_lattice(entry.lattice))
            coords.append(np.asarray(entry.coords, dtype=np.float64) % 1.0)
        pbc.append([entry.lattice is not None] * 3)

    def column(values: list[float | None]) -> torch.Tensor:
        cleaned = [float("nan") if value is None else value for value in values]
        return torch.tensor(cleaned, dtype=torch.float64).float()

    return Structures(
        lattice=torch.stack(lattices),
        x=torch.from_numpy(np.concatenate(coords)).float(),
        z=torch.from_numpy(np.concatenate([e.numbers for e in entries])).long(),
        num_atoms=torch.tensor([len(e.numbers) for e in entries], dtype=torch.int64),
        pbc=torch.tensor(pbc, dtype=torch.bool),
        properties={
            name: column([e.properties.get(name) for e in entries]) for name in names
        },
        **{label: column([e.labels.get(label) for e in entries]) for label in LABELS},
    )


def batches(
    entries: Iterable[Entry], names: tuple[str, ...], batch_size: int
) -> Iterator[Structures]:
    """Batches of the structures of the entries, in their order."""
    pending: list[Entry] = []

    for entry in entries:
        pending.append(entry)
        if len(pending) == batch_size:
            yield as_structures(pending, names)
            pending = []

    if len(pending) > 0:
        yield as_structures(pending, names)


def save(
    hdf5_file: Path,
    entries: Iterable[Entry],
    names: tuple[str, ...] = (),
    batch_size: int = 1024,
) -> None:
    """Save the structures of the entries to an HDF5 file, in their order.

    Args:
        hdf5_file: File written.
        entries: Entries of a source.
        names: Names of the properties of the dataset.
        batch_size: Number of structures written at once.
    """
    hdf5_file.parent.mkdir(parents=True, exist_ok=True)
    HDF5Dataset.from_structures(hdf5_file, batches(entries, names, batch_size))


def summary(hdf5_file: Path) -> str:
    """Number of structures and of atoms of a file, and its size."""
    dataset = HDF5Dataset(hdf5_file)
    num_atoms = dataset.dataset("num_atoms")[...]

    return (
        f"{len(num_atoms)} structures ({int(num_atoms.sum())} atoms, "
        f"{int(num_atoms.min())} to {int(num_atoms.max())} per structure), "
        f"{hdf5_file.stat().st_size / 1e6:.1f} MB  {hdf5_file}"
    )
