"""Write P1 CIF blocks readable by pycrystalgen.data.cif.CifParser.

Shared by the dataset conversion scripts. Numbers are written in fixed-point
notation because the parser does not read scientific notation.
"""

import math
from typing import Sequence, TextIO

import numpy as np

TOTAL_ENERGY_TAG = "_energy_total"
ENERGY_ABOVE_HULL_TAG = "_energy_above_hull"
BAND_GAP_TAG = "_band_gap"
FORMATION_ENERGY_PER_ATOM_TAG = "_formation_energy_per_atom"


def property_tags(
    total_energy: float | None = None,
    energy_above_hull: float | None = None,
    band_gap: float | None = None,
    formation_energy_per_atom: float | None = None,
) -> str:
    """Format the property tags, missing (None or NaN) properties are omitted."""
    lines = ""
    for tag, value in (
        (TOTAL_ENERGY_TAG, total_energy),
        (ENERGY_ABOVE_HULL_TAG, energy_above_hull),
        (BAND_GAP_TAG, band_gap),
        (FORMATION_ENERGY_PER_ATOM_TAG, formation_energy_per_atom),
    ):
        if value is None or not math.isfinite(value):
            continue
        lines += f"{tag} {value:.8f}\n"
    return lines


def cell_parameters(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute lengths (a, b, c) and angles (alpha, beta, gamma) in degrees."""
    lengths = np.linalg.norm(lattice, axis=1)
    unit = lattice / lengths[:, None]
    cosines = np.array([unit[1] @ unit[2], unit[0] @ unit[2], unit[0] @ unit[1]])
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return lengths, angles


def write_block(
    fp: TextIO,
    name: str,
    lattice: np.ndarray,
    frac: np.ndarray,
    symbols: Sequence[str],
    total_energy: float | None = None,
    energy_above_hull: float | None = None,
    band_gap: float | None = None,
    formation_energy_per_atom: float | None = None,
) -> None:
    """Write a single structure as a P1 CIF data block.

    Args:
        fp: Output file.
        name: Name of the data block (no whitespace).
        lattice: Lattice vectors as rows, shape (3, 3), in angstrom.
        frac: Fractional coordinates, shape (n_atoms, 3).
        symbols: Element symbol of each atom.
        total_energy: Total energy in eV.
        energy_above_hull: Energy above the hull in eV/atom.
        band_gap: Band gap in eV.
        formation_energy_per_atom: Formation energy in eV/atom.
    """
    (a, b, c), (alpha, beta, gamma) = cell_parameters(lattice)

    atoms = "".join(
        f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n"
        for symbol, (x, y, z) in zip(symbols, (frac % 1.0).tolist())
    )

    fp.write(
        f"data_{name}\n"
        "_symmetry_Int_Tables_number 1\n"
        f"_cell_length_a {a:.8f}\n"
        f"_cell_length_b {b:.8f}\n"
        f"_cell_length_c {c:.8f}\n"
        f"_cell_angle_alpha {alpha:.8f}\n"
        f"_cell_angle_beta {beta:.8f}\n"
        f"_cell_angle_gamma {gamma:.8f}\n"
        f"{property_tags(total_energy, energy_above_hull, band_gap, formation_energy_per_atom)}"
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        f"{atoms}"
        "\n"
    )
