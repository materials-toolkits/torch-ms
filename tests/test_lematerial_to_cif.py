"""Tests for the LeMat-Bulk conversion script."""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lematerial_to_cif import (  # noqa: E402
    HullLookup,
    convert,
    iter_rows,
    unique_mask,
)

from pycrystalgen.data import parse_cif  # noqa: E402


def test_hull_lookup():
    hull = HullLookup(np.array(["b-2", "a-1", "c-3"]), np.array([0.2, 0.1, 0.3]))

    values = hull(["c-3", "missing", "a-1", "a-10", ""])

    assert values[[0, 2]].tolist() == [0.3, 0.1]
    assert np.isnan(values[[1, 3, 4]]).all()
    assert hull.num_repeated == 0
    assert len(hull([])) == 0


def test_hull_lookup_repeated_id_uses_first_value():
    hull = HullLookup(np.array(["a", "b", "a", "a"]), np.array([0.5, 0.2, 0.1, 0.9]))

    assert hull(["a", "b"]).tolist() == [0.5, 0.2]
    assert hull.num_repeated == 2
    assert hull.max_disagreement == pytest.approx(0.8)


def test_unique_mask_keeps_lowest_energy():
    fingerprints = np.array(["f1", "f2", "f1", "f3", "f2", "f1"], dtype=object)
    energy = np.array([-1.0, -5.0, -3.0, 0.0, -4.0, -2.0])

    assert unique_mask(fingerprints, energy).tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_unique_mask_ties_keep_first_and_missing_are_kept():
    fingerprints = np.array(["f1", None, "f1", None, "f1"], dtype=object)
    energy = np.array([-2.0, -9.0, -2.0, -9.0, -1.0])

    assert unique_mask(fingerprints, energy).tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]


@pytest.fixture
def parquet_files(tmp_path):
    def row(name, lattice, cart, species, energy):
        return {
            "immutable_id": name,
            "lattice_vectors": lattice,
            "cartesian_site_positions": cart,
            "species_at_sites": species,
            "energy": energy,
        }

    cubic = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    tilted = [[3.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 2.0, 6.0]]
    rows = [
        row("a-1", cubic, [[0.0, 0.0, 0.0], [2.0, 1.0, 3.0]], ["Na", "Cl"], -7.5),
        row("a-2", tilted, [[0.5, 3.5, 3.0]], ["Fe"], -8.25),
        row("a-3", cubic, [[1.0, 1.0, 1.0]], ["Cu"], -3.0),
    ]

    files = []
    for i, chunk in enumerate((rows[:2], rows[2:])):
        path = tmp_path / f"part-{i}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), path)
        files.append(str(path))
    return files


def test_convert(tmp_path, parquet_files):
    hull = HullLookup(np.array(["a-1", "a-3"]), np.array([0.0, 0.25]))
    cif_file = tmp_path / "out.cif"

    counts = convert(parquet_files, cif_file, hull)
    structures = parse_cif(cif_file.read_text())

    assert counts == {"written": 3, "without_energy_above_hull": 1}
    assert structures.num_atoms.tolist() == [2, 1, 1]
    assert structures.z.tolist() == [11, 17, 26, 29]
    assert torch.allclose(structures.total_energy, torch.tensor([-7.5, -8.25, -3.0]))
    assert torch.allclose(structures.x[1], torch.tensor([0.5, 0.25, 0.75]))
    # tilted cell: cart = frac @ lattice with frac = (0, 0.5, 0.5)
    assert torch.allclose(structures.x[2], torch.tensor([0.0, 0.5, 0.5]), atol=1e-6)
    assert torch.isclose(structures.energy_above_hull[2], torch.tensor(0.25))
    assert torch.isnan(structures.energy_above_hull[1])
    assert torch.isnan(structures.band_gap).all()


def test_convert_with_mask_and_limit(tmp_path, parquet_files):
    keep = np.array([True, False, True])

    assert [r["immutable_id"] for r in iter_rows(parquet_files, keep)] == ["a-1", "a-3"]
    assert [r["immutable_id"] for r in iter_rows(parquet_files, batch_size=1)] == [
        "a-1",
        "a-2",
        "a-3",
    ]

    counts = convert(parquet_files, tmp_path / "kept.cif", keep=keep)
    assert counts["written"] == 2
    assert parse_cif((tmp_path / "kept.cif").read_text()).z.tolist() == [11, 17, 29]

    counts = convert(parquet_files, tmp_path / "limit.cif", limit=1)
    assert counts["written"] == 1
    assert parse_cif((tmp_path / "limit.cif").read_text()).z.tolist() == [11, 17]
