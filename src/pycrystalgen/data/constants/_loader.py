"""Loader for periodic table and element data."""

from __future__ import annotations

import h5py
import json
from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch


@dataclass(frozen=True)
class ElementData:
    """Immutable data class for element properties.

    Attributes:
        atomic_number: Atomic number (Z).
        symbol: Element symbol (e.g., "H", "He").
        name: Element name (e.g., "Hydrogen").
        mass: Atomic mass in atomic mass units (u).
        electronegativity: Pauling electronegativity (None for noble gases).
        atomic_radius: Atomic radius in picometers (pm).
        ionization_energy: First ionization energy in electronvolts (eV).
    """

    atomic_number: int
    symbol: str
    name: str
    mass: float
    electronegativity: float | None
    atomic_radius: int
    ionization_energy: float


def _load_elements() -> tuple[dict[int, ElementData], dict[str, int]]:
    """Load element data from JSON file.

    Returns:
        Tuple of (atomic_number -> ElementData, symbol -> atomic_number).
    """
    with resources.files(__package__).joinpath("periodic_table.json").open() as f:
        data = json.load(f)

    elements: dict[int, ElementData] = {}
    symbol_map: dict[str, int] = {}
    for element in data["elements"]:
        atomic_number = element["atomic_number"]
        symbol = element["symbol"]
        elements[atomic_number] = ElementData(
            atomic_number=atomic_number,
            symbol=symbol,
            name=element["name"],
            mass=element["mass"],
            electronegativity=element["electronegativity"],
            atomic_radius=element["atomic_radius"],
            ionization_energy=element["ionization_energy"],
        )
        symbol_map[symbol] = atomic_number

    return elements, symbol_map


_ELEMENTS, SYMBOL_TO_NUMBER = _load_elements()


def symbol_to_number(symbol: str) -> int:
    """Convert element symbol to atomic number.

    Args:
        symbol: Element symbol (e.g., "H", "He", "Li").

    Returns:
        Atomic number.

    Raises:
        KeyError: If symbol is not found.
    """
    return SYMBOL_TO_NUMBER[symbol]


def number_to_symbol(atomic_number: int) -> str:
    """Convert atomic number to element symbol.

    Args:
        atomic_number: Atomic number (Z).

    Returns:
        Element symbol (e.g., "H", "He").

    Raises:
        KeyError: If atomic number is not found.
    """
    return _ELEMENTS[atomic_number].symbol


def get_element(identifier: int | str) -> ElementData:
    """Get element data by atomic number or symbol.

    Args:
        identifier: Atomic number (int) or element symbol (str).

    Returns:
        ElementData for the specified element.

    Raises:
        KeyError: If identifier is not found.
        TypeError: If identifier is not int or str.
    """
    if isinstance(identifier, int):
        return _ELEMENTS[identifier]
    if isinstance(identifier, str):
        return _ELEMENTS[symbol_to_number(identifier)]
    raise TypeError(f"identifier must be int or str, got {type(identifier).__name__}")


@dataclass(frozen=True)
class SpaceGroupData:
    """Immutable data class for space group properties.

    Attributes:
        number: Space group number (1-230).
        hall_number: Hall number for the space group.
        hall_symbol: Hall symbol.
        symbol: Full Hermann-Mauguin symbol.
        crystal_system: Crystal system name (triclinic, monoclinic, etc.).
    """

    number: int
    hall_number: int
    hall_symbol: str
    symbol: str
    crystal_system: str


def _load_spacegroups() -> dict[int, SpaceGroupData]:
    """Load space group data from JSON file.

    Returns:
        Dictionary mapping space group number to SpaceGroupData.
    """
    with resources.files(__package__).joinpath("spacegroups.json").open() as f:
        data = json.load(f)

    spacegroups: dict[int, SpaceGroupData] = {}
    for sg in data["spacegroups"]:
        spacegroups[sg["number"]] = SpaceGroupData(
            number=sg["number"],
            hall_number=sg["hall_number"],
            hall_symbol=sg["hall_symbol"],
            symbol=sg["symbol"],
            crystal_system=sg["crystal_system"],
        )

    return spacegroups


_SPACEGROUPS = _load_spacegroups()


def _open_spacegroup_h5() -> h5py.File:
    return h5py.File(resources.files(__package__).joinpath("spacegroups.h5"), "r")


_SYMMETRY_OPS: dict[int, np.ndarray] | None = None


def _get_symmetry_ops_cached() -> dict[int, np.ndarray]:
    global _SYMMETRY_OPS
    if _SYMMETRY_OPS is None:
        _SYMMETRY_OPS = {}
        with _open_spacegroup_h5() as h5:
            for key in h5.keys():
                _SYMMETRY_OPS[int(key)] = np.array(h5[key], dtype=np.float64)
    return _SYMMETRY_OPS


def spacegroup_number_to_data(number: int) -> SpaceGroupData:
    """Get space group data by number.

    Args:
        number: Space group number (1-230).

    Returns:
        SpaceGroupData for the specified space group.

    Raises:
        KeyError: If space group number is not found.
    """
    return _SPACEGROUPS[number]


def spacegroup_symbol_to_number(symbol: str) -> int:
    """Convert space group symbol to number.

    Args:
        symbol: Hermann-Mauguin symbol (e.g., "Pm-3m", "Fd-3m").

    Returns:
        Space group number.

    Raises:
        KeyError: If symbol is not found.
    """
    for number, data in _SPACEGROUPS.items():
        if data.symbol == symbol:
            return number
    raise KeyError(f"Space group symbol '{symbol}' not found")


def get_spacegroup(identifier: int | str) -> SpaceGroupData:
    """Get space group data by number or symbol.

    Args:
        identifier: Space group number (int) or symbol (str).

    Returns:
        SpaceGroupData for the specified space group.

    Raises:
        KeyError: If identifier is not found.
        TypeError: If identifier is not int or str.
    """
    if isinstance(identifier, int):
        return _SPACEGROUPS[identifier]
    if isinstance(identifier, str):
        return _SPACEGROUPS[spacegroup_symbol_to_number(identifier)]
    raise TypeError(f"identifier must be int or str, got {type(identifier).__name__}")


def get_symmetry_operations(identifier: int | str) -> torch.Tensor:
    """Get symmetry operations as affine transformation matrices.

    Args:
        identifier: Space group number (int) or symbol (str).

    Returns:
        Tensor of shape (num_ops, 4, 4) containing affine transformation matrices.
        Each matrix is a 4x4 homogeneous transformation matrix [R|t; 0 0 0 1].
    """
    sg = get_spacegroup(identifier)
    return torch.from_numpy(_get_symmetry_ops_cached()[sg.number].copy())
