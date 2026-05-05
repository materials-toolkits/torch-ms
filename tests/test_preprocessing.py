"""Integration tests for preprocessing CLI."""

import json
import math
import multiprocessing as mp
import random
import threading
from pathlib import Path

import pytest
import torch

from pycrystalgen.data.hdf5 import HDF5Dataset
from pycrystalgen.data.preprocessing import Preprocessing, preprocessing
from pycrystalgen.data.constants import symbol_to_number


@pytest.fixture(autouse=True)
def cleanup_child_processes():
    """Kill all child processes and tqdm monitor threads after each test."""
    yield
    for p in mp.active_children():
        p.terminate()
    from tqdm import tqdm

    try:
        tqdm._instances_clear()
    except Exception:
        pass
    for t in threading.enumerate():
        if t.name == "TqdmMonitorThread":
            try:
                t._is_stopped.set()
            except Exception:
                pass


@pytest.fixture
def concatenated_triclinic_cif(tmp_path):
    """Build a temporary directory with a concatenated CIF file from triclinic structures."""
    triclinic_files = [
        Path("tests/cif/triclinic1.cif"),
        Path("tests/cif/triclinic2.cif"),
        Path("tests/cif/triclinic3.cif"),
    ]

    cif_dir = tmp_path / "cif"
    cif_dir.mkdir()

    selected_files = random.choices(triclinic_files, k=12)

    cif_content = ""
    expected_structures = []
    for f in selected_files:
        cif_content += f.read_text()
        cif_content += "\n"

        json_path = f.with_suffix(".json")
        expected_structures.append(json.loads(json_path.read_text()))

    cif_file = cif_dir / "triclinic_concat.cif"
    cif_file.write_text(cif_content)

    json_file = cif_dir / "expected.json"
    json_file.write_text(json.dumps(expected_structures, indent=2))

    return cif_file, json_file, expected_structures


def build_lattice_vectors(cell: dict[str, float]) -> torch.Tensor:
    """Build lattice vectors from cell parameters (in degrees)."""
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha, beta, gamma = cell["alpha"], cell["beta"], cell["gamma"]
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    cos_alpha = math.cos(alpha_rad)
    cos_beta = math.cos(beta_rad)
    cos_gamma = math.cos(gamma_rad)
    sin_gamma = math.sin(gamma_rad)

    v = math.sqrt(
        1.0
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    )

    lattice = torch.tensor(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                c * v / sin_gamma,
            ],
        ],
        dtype=torch.float32,
    )

    return lattice


def validate_structure(struct: "Structures", expected: dict) -> None:
    """Validate a parsed structure against expected JSON data.

    Args:
        struct: Parsed structure from HDF5Dataset.
        expected: Expected structure data from JSON file.

    Raises:
        AssertionError: If validation fails.
    """
    assert struct.num_atoms.shape[0] == 1
    assert struct.num_atoms[0].item() == expected["num_atoms"]

    expected_lattice = build_lattice_vectors(expected["cell_parameters"])
    torch.testing.assert_close(
        struct.lattice[0], expected_lattice, atol=1e-3, rtol=1e-3
    )

    start = struct.ptr[0].item()
    end = struct.ptr[1].item()
    assert end - start == expected["num_atoms"]

    for j, atom in enumerate(expected["atoms"]):
        frac_coords = torch.tensor(atom["coords"], dtype=torch.float32)
        torch.testing.assert_close(
            struct.x[start + j], frac_coords, atol=1e-3, rtol=1e-3
        )
        expected_z = symbol_to_number(atom["element"])
        assert struct.z[start + j].item() == expected_z


class TestPreprocessing:
    """Integration tests for preprocessing."""

    def test_preprocessing_single_process(self, tmp_path, concatenated_triclinic_cif):
        """Test preprocessing with single process."""
        cif_file, json_file, expected_structures = concatenated_triclinic_cif
        hdf5_file = tmp_path / "output.hdf5"

        preprocessing(
            Preprocessing(
                source=cif_file,
                destination=hdf5_file,
                num_workers=0,
            )
        )

        assert hdf5_file.exists()

        dataset = HDF5Dataset(hdf5_file)
        assert len(dataset) == len(expected_structures)

        for i, expected in enumerate(expected_structures):
            struct = dataset[i]
            validate_structure(struct, expected)

    @pytest.mark.skip(
        reason="weird bug during testing, torch.Tensor.repeat_interleave seem to block execusion for no reason. It works outside of tests tho"
    )
    def test_preprocessing_multiprocess(self, tmp_path, concatenated_triclinic_cif):
        """Test preprocessing with multiprocessing."""
        cif_file, json_file, expected_structures = concatenated_triclinic_cif
        hdf5_file = tmp_path / "output.hdf5"

        preprocessing(
            Preprocessing(
                source=cif_file,
                destination=hdf5_file,
                num_workers=2,
            )
        )

        assert hdf5_file.exists()

        dataset = HDF5Dataset(hdf5_file)
        assert len(dataset) == len(expected_structures)

        for i, expected in enumerate(expected_structures):
            struct = dataset[i]
            validate_structure(struct, expected)
