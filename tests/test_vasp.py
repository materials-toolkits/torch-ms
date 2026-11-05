"""Tests for the POSCAR and CONTCAR files of VASP."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read, write

from torchms.data import VaspDataset, parse_poscar

CIF = Path(__file__).parent / "cif"

# structures of the tests of CIF, read by ASE and written as POSCAR by it
STRUCTURES = sorted(
    path for path in CIF.glob("**/*.cif") if path.stem not in ("FePO4",)
)  # ASE cannot read FePO4.cif


def ase_structure(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read(path)


def as_poscar(atoms, **options) -> str:
    stream = io.StringIO()
    write(stream, atoms, format="vasp", **options)
    return stream.getvalue()


def check_against_ase(text: str, expected, **options) -> None:
    """The crystal of a POSCAR is the one ASE reads: cell, elements, positions in Å."""
    structures = parse_poscar(text, **options)

    assert np.allclose(structures.lattice[0].numpy(), expected.cell[:], atol=1e-4)
    assert structures.z.tolist() == expected.numbers.tolist()
    wrapped = expected.get_scaled_positions(wrap=True)
    found = structures.x.double().numpy()
    difference = found - wrapped
    assert np.abs(difference - np.round(difference)).max() < 1e-5


@pytest.mark.parametrize("path", STRUCTURES, ids=lambda path: path.stem)
@pytest.mark.parametrize("direct", [True, False], ids=["direct", "cartesian"])
def test_poscar_written_by_ase(path, direct):
    poscar = as_poscar(ase_structure(path), direct=direct)
    check_against_ase(poscar, read(io.StringIO(poscar), format="vasp"))


@pytest.mark.parametrize("path", STRUCTURES[:10], ids=lambda path: path.stem)
def test_poscar_of_vasp4(path):
    """No line of elements: they are given, as ASE needs them from the comment."""
    atoms = ase_structure(path)
    poscar = as_poscar(atoms, vasp5=False, direct=True)
    expected = read(io.StringIO(poscar), format="vasp")
    groups = poscar.splitlines()[0].split()

    check_against_ase(poscar, expected, species=groups)
    check_against_ase(poscar, expected)  # the comment of ASE lists them


NACL = """NaCl, rocksalt
{scaling}
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
Na_pv Cl/a1b2c3
1 1
{mode}
{positions}"""


def test_negative_scaling_is_the_volume():
    poscar = NACL.format(scaling="-44.0", mode="Direct", positions="0 0 0\n0.5 0.5 0.5")
    lattice = parse_poscar(poscar).lattice[0].double()

    volume = torch.linalg.det(lattice).abs()  # pylint: disable=not-callable
    assert torch.isclose(volume, torch.tensor(44.0).double())


def test_three_scaling_factors_scale_the_components():
    poscar = NACL.format(
        scaling="2.0 3.0 4.0", mode="Cartesian", positions="0 0 0\n0.5 0.5 0.5"
    )
    structures = parse_poscar(poscar)

    expected = torch.tensor([[0.0, 1.5, 2.0], [1.0, 0.0, 2.0], [1.0, 1.5, 0.0]])
    assert torch.allclose(structures.lattice[0], expected)
    # the Cartesian position (1.0, 1.5, 2.0) after scaling: half of each vector
    assert torch.allclose(structures.x[1], torch.tensor([0.5, 0.5, 0.5]))


def test_selective_dynamics_and_velocities_are_skipped():
    poscar = NACL.format(
        scaling="5.64",
        mode="Selective dynamics\nDirect",
        positions="0 0 0 T T F\n0.5 0.5 0.5 F F F ! Cl\n\n0.1 0.2 0.3\n0.0 0.0 0.0\n",
    )
    structures = parse_poscar(poscar)

    assert structures.z.tolist() == [11, 17]
    assert torch.allclose(structures.x[1], torch.tensor([0.5, 0.5, 0.5]))


def test_positions_are_wrapped():
    poscar = NACL.format(
        scaling="5.64", mode="Direct", positions="-0.25 1.0 0\n0.5 0.5 0.5"
    )
    assert torch.allclose(parse_poscar(poscar).x[0], torch.tensor([0.75, 0.0, 0.0]))


@pytest.mark.parametrize(
    "text, message",
    [
        ("NaCl\n5.64\n", "at least 8 lines"),
        (
            NACL.format(scaling="5.64", mode="Direct", positions="0 0 0"),
            "fewer positions",
        ),
        (
            NACL.replace("Na_pv Cl/a1b2c3\n", "")
            .format(scaling="5.64", mode="Direct", positions="0 0 0\n0.5 0.5 0.5")
            .replace("NaCl, rocksalt", "rocksalt"),
            "VASP 4",
        ),
        (
            NACL.replace("1 1", "1 1 1").format(
                scaling="5.64", mode="Direct", positions="0 0 0\n0.5 0.5 0.5\n0 0 0"
            ),
            "2 elements for 3 groups",
        ),
    ],
    ids=["short", "missing positions", "no elements", "counts"],
)
def test_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_poscar(text)


def test_dataset_reads_the_directories(tmp_path):
    poscar = NACL.format(scaling="5.64", mode="Direct", positions="0 0 0\n0.5 0.5 0.5")
    (tmp_path / "relax").mkdir()
    (tmp_path / "relax" / "POSCAR").write_text(poscar)
    (tmp_path / "relax" / "CONTCAR").write_text(poscar)
    (tmp_path / "nacl.vasp").write_text(poscar)
    (tmp_path / "notes.txt").write_text("not a structure")

    dataset = VaspDataset(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in dataset.files] == [
        "nacl.vasp",
        "relax/CONTCAR",
        "relax/POSCAR",
    ]
    assert dataset[0].z.tolist() == [11, 17]
    with pytest.raises(ValueError, match="No POSCAR"):
        VaspDataset(tmp_path / "relax", patterns=("*.vasp",))
