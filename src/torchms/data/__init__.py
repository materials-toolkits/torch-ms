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
from .filters import Filter
from .hdf5 import HDF5Dataset
from .loader import OnDevice
from .memory import InMemoryDataset, MixedDataset, mix
from .split import load_split, random_split, save_split, split_indices
from .vasp import VaspDataset, parse_poscar

__all__ = [
    "AutoLoader",
    "Hardware",
    "LoaderConfig",
    "auto_config",
    "auto_loader",
    "detect_hardware",
    "Structures",
    "CifDataset",
    "Filter",
    "HDF5Dataset",
    "InMemoryDataset",
    "MixedDataset",
    "OnDevice",
    "VaspDataset",
    "available_datasets",
    "dataset_url",
    "load_dataset",
    "load_split",
    "mix",
    "parse_cif",
    "parse_poscar",
    "random_split",
    "save_split",
    "split_indices",
]
