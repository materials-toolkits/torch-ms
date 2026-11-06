"""Physical and chemical constants."""

from ._loader import (
    ElementData,
    SpaceGroupData,
    WyckoffPosition,
    get_element,
    get_spacegroup,
    get_symmetry_operations,
    get_wyckoff_positions,
    number_to_symbol,
    spacegroup_number_to_data,
    spacegroup_symbol_to_number,
    symbol_to_number,
)

# TODO: check values from periodic_table.json

__all__ = [
    "ElementData",
    "SpaceGroupData",
    "WyckoffPosition",
    "get_element",
    "get_spacegroup",
    "get_symmetry_operations",
    "get_wyckoff_positions",
    "number_to_symbol",
    "spacegroup_number_to_data",
    "spacegroup_symbol_to_number",
    "symbol_to_number",
]
