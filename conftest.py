"""Configuration of the tests, and of the examples of the documentation.

`--full` also runs the tests marked full: the long tests of the diffusion
models, which compare every recorded step of a sampling with the original.

`uv run poe docs-test` runs the Python blocks of the pages of docs/. A block is
a test of its own: it runs alone, in a directory with the placeholder files
below, and with the names of DOCS_GLOBALS already imported.

In a page, the first line of a block says how it is run:

* "```python": the block runs alone;
* "```{.python continuation}": it runs after the blocks before it, with what
  they define;
* "```{.python notest}": it is not run. For a block which needs the network or a
  GPU, a block which depends on one of them, or a fragment which is not a
  complete example.
"""

import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchms.data import (
    Filter,
    HDF5Dataset,
    InMemoryDataset,
    Structures,
    auto_loader,
    load_dataset,
    parse_cif,
    random_split,
)
from torchms.data.augmentation import Compose
from torchms.graph.cutoff import cutoff
from torchms.graph.sparse import scatter
from torchms.graph.transforms import (
    AddAtomicRadiusGraph,
    AddLineGraph,
    AddQuadruplets,
    AddRadiusGraph,
    AddTriplets,
    KeepNearestNeighbors,
)

ROOT = Path(__file__).parent

# names a block of the documentation can use without importing them
DOCS_GLOBALS = {
    "torch": torch,
    "F": F,
    "DataLoader": DataLoader,
    "Structures": Structures,
    "HDF5Dataset": HDF5Dataset,
    "InMemoryDataset": InMemoryDataset,
    "Filter": Filter,
    "load_dataset": load_dataset,
    "auto_loader": auto_loader,
    "random_split": random_split,
    "parse_cif": parse_cif,
    "Compose": Compose,
    "AddRadiusGraph": AddRadiusGraph,
    "AddAtomicRadiusGraph": AddAtomicRadiusGraph,
    "AddLineGraph": AddLineGraph,
    "AddTriplets": AddTriplets,
    "AddQuadruplets": AddQuadruplets,
    "KeepNearestNeighbors": KeepNearestNeighbors,
    "cutoff": cutoff,
    "scatter": scatter,
}

# files the examples read, as they name them: a few structures in each
PLACEHOLDER_HDF5 = (
    "my_structures.hdf5",
    "structures.hdf5",
    "datasets/zinc20_3d/small.hdf5",
)
PLACEHOLDER_CIF = ("file.cif", "structures.cif")

# the scripts which run the original codes, in their own environments: they
# import packages torchms does not depend on (alignn, matgl, mace...)
collect_ignore = ["tests/references"]


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        help="also run the long tests of the diffusion models (marked full): "
        "every recorded step of their samplings, their chained dynamics",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked full, unless pytest is given --full."""
    if config.getoption("--full"):
        return

    skip = pytest.mark.skip(reason="long test of a diffusion model: run with --full")
    for item in items:
        if item.get_closest_marker("full") is not None:
            item.add_marker(skip)


def pytest_markdown_docs_globals():
    """Names defined in every block of the documentation."""
    return dict(DOCS_GLOBALS)


@pytest.fixture(scope="session")
def docs_files(tmp_path_factory) -> Path:
    """Directory with the files read by the examples of the documentation."""
    directory = tmp_path_factory.mktemp("docs")
    structures = [Structures.diamond(), Structures.zincblende(), Structures.water()]

    for name in PLACEHOLDER_HDF5:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        HDF5Dataset.from_structures(directory / name, structures)

    for name in PLACEHOLDER_CIF:
        shutil.copy(ROOT / "tests" / "cif" / "triclinic1.cif", directory / name)

    return directory


@pytest.fixture(autouse=True)
def docs_directory(request, monkeypatch):
    """Run a block of the documentation in a copy of the placeholder files."""
    if request.node.get_closest_marker("markdown-docs") is None:
        return

    directory = request.getfixturevalue("docs_files")
    work = request.getfixturevalue("tmp_path")
    shutil.copytree(directory, work, dirs_exist_ok=True)
    monkeypatch.chdir(work)
