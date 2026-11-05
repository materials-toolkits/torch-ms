"""Write spacegroups.json: the 230 space groups by their international number.

Each group is described by the Hall setting of spglib whose operations are the
ones spacegroups.h5 holds for its number (written by update_spacegroups_h5.py
from pymatgen), checked: the first setting of spglib for every group, the
first origin of the groups which have two, hexagonal axes for the rhombohedral
ones. The symbol is the short Hermann-Mauguin symbol of spglib ("Fm-3m",
"P2_1/c"), unique.

    uv run python scripts/update_spacegroups_json.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
import spglib

CONSTANTS = Path(__file__).resolve().parent.parent / "src/torchms/data/constants"

# last international number of each crystal system
CRYSTAL_SYSTEMS = [
    (2, "triclinic"),
    (15, "monoclinic"),
    (74, "orthorhombic"),
    (142, "tetragonal"),
    (167, "trigonal"),
    (194, "hexagonal"),
    (230, "cubic"),
]
NUM_HALL_SETTINGS = 530


def crystal_system(number: int) -> str:
    return next(name for last, name in CRYSTAL_SYSTEMS if number <= last)


def group_type(hall: int):
    """Space-group type of spglib of a Hall setting."""
    kind = spglib.get_spacegroup_type(hall)
    assert kind is not None, hall
    return kind


def setting_operations(hall: int) -> set[tuple]:
    """Operations of a Hall setting in the database of spglib."""
    database = spglib.get_symmetry_from_database(hall)
    assert database is not None, hall
    return operations(database["rotations"], database["translations"])


def operations(rotations: np.ndarray, translations: np.ndarray) -> set[tuple]:
    """Affine operations as a set, translations modulo 1, rounded."""
    return {
        (tuple(r.astype(int).ravel()), tuple(np.round(t % 1.0, 4) % 1.0))
        for r, t in zip(rotations, translations)
    }


def main() -> None:
    settings: dict[int, list[int]] = {}
    for hall in range(1, NUM_HALL_SETTINGS + 1):
        settings.setdefault(group_type(hall).number, []).append(hall)

    groups = []
    with h5py.File(CONSTANTS / "spacegroups.h5", "r") as h5:
        for number in range(1, 231):
            affine = np.asarray(h5[str(number)], dtype=np.float64)
            stored = operations(np.rint(affine[:, :3, :3]), affine[:, :3, 3])
            matching = [
                hall for hall in settings[number] if setting_operations(hall) == stored
            ]
            assert matching, f"no Hall setting of {number} has the stored operations"
            hall = matching[0]
            kind = group_type(hall)
            groups.append(
                {
                    "number": number,
                    "hall_number": hall,
                    "hall_symbol": kind.hall_symbol,
                    "symbol": kind.international_short,
                    "crystal_system": crystal_system(number),
                }
            )

    symbols = [group["symbol"] for group in groups]
    assert len(set(symbols)) == len(symbols), "symbols are not unique"
    with open(CONSTANTS / "spacegroups.json", "w", encoding="utf-8") as fp:
        json.dump({"spacegroups": groups}, fp, indent=2)
        fp.write("\n")


if __name__ == "__main__":
    main()
