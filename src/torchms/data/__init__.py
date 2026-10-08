"""Data structures for crystal generation."""

from .auto import (
    AutoLoader,
    Hardware,
    LoaderConfig,
    auto_config,
    auto_loader,
    detect_hardware,
)
from .batching import Structures
from .cif import CifDataset, parse_cif
from .datasets import available_datasets, dataset_url, load_dataset
from .hdf5 import HDF5Dataset
from .loader import OnDevice
from .memory import InMemoryDataset

__all__ = [
    "AutoLoader",
    "Hardware",
    "LoaderConfig",
    "auto_config",
    "auto_loader",
    "detect_hardware",
    "Structures",
    "CifDataset",
    "HDF5Dataset",
    "InMemoryDataset",
    "OnDevice",
    "available_datasets",
    "dataset_url",
    "load_dataset",
    "parse_cif",
]
