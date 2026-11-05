"""Tests of the weights, the calibration, EnergyEstimator and the energy above the hull."""

import h5py
import numpy as np
import pytest
import torch

from helpers.structures import tilted
from torchms.data import HDF5Dataset, Structures
from torchms.models import OrbV3
from torchms.models.estimator import EnergyEstimator
from torchms.models.estimator import calibration
from torchms.models.estimator import weights
from torchms.models.estimator.energy import atom_batches, to_device

# weights


def test_checkpoint_path(tmp_path, monkeypatch):
    path = weights.checkpoint_paths("orb", "orb-v3-conservative-inf-omat", tmp_path)[0]
    assert path == tmp_path / "orb" / "orb-v3-conservative-inf-omat-20250404.ckpt"

    monkeypatch.setenv("TORCHMS_CACHE", str(tmp_path / "cache"))
    assert weights.cache_directory() == tmp_path / "cache"
    assert weights.checkpoint_paths("orb", "orb-v3-conservative-inf-mpa")[0].parent == (
        tmp_path / "cache" / "orb"
    )

    monkeypatch.delenv("TORCHMS_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert weights.cache_directory() == tmp_path / "xdg" / "torchms"

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("orb", "orb-v9")[0]


def test_download_checks_the_content(tmp_path, monkeypatch):
    source = tmp_path / "weights.ckpt"
    source.write_bytes(b"pretrained weights")
    good = weights.Checkpoint(source.as_uri(), weights.sha256(source))
    bad = weights.Checkpoint(source.as_uri(), "0" * 64)
    monkeypatch.setitem(weights.CHECKPOINTS["orb"], "good", (good,))
    monkeypatch.setitem(weights.CHECKPOINTS["orb"], "bad", (bad,))

    path = weights.download("orb", "good", tmp_path / "cache")[0]
    assert path.read_bytes() == b"pretrained weights"
    assert not path.with_name(path.name + ".part").exists()

    # not downloaded again
    source.unlink()
    assert weights.download("orb", "good", tmp_path / "cache")[0] == path

    with pytest.raises(weights.InvalidCheckpoint, match="SHA-256"):
        weights.download("orb", "bad", tmp_path / "cache")[0]


# calibration


def dataset_arrays(compositions, energies):
    z, num_atoms = [], []
    for composition in compositions:
        atoms = [el for el, count in composition.items() for _ in range(count)]
        z += atoms
        num_atoms.append(len(atoms))
    return np.array(z), np.array(num_atoms), np.array(energies, dtype=float)


COMPOSITIONS = [
    {1: 2, 8: 1},
    {1: 1, 8: 2},
    {1: 2},
    {8: 1, 26: 1},
    {26: 3},
    {1: 4, 8: 2},
]


def test_fit_recovers_the_shifts():
    shift = {1: 1.0, 8: -0.5, 26: 0.25}
    z, num_atoms, target = dataset_arrays(
        COMPOSITIONS, [-10.0, -12.0, -4.0, -7.0, -9.0, -20.0]
    )
    offset = np.array([sum(shift[el] * n for el, n in c.items()) for c in COMPOSITIONS])

    correction, counts = calibration.fit_element_correction(
        z, num_atoms, target - offset, target
    )

    assert correction.shape == (128,) and counts.shape == (128,)
    np.testing.assert_allclose(correction[[1, 8, 26]], [1.0, -0.5, 0.25], atol=1e-9)
    assert counts[[1, 8, 26]].tolist() == [4, 4, 2]
    assert np.count_nonzero(correction) == 3 and counts.sum() == 10

    corrected = calibration.apply_element_correction(
        correction, z, num_atoms, target - offset
    )
    np.testing.assert_allclose(corrected, target, atol=1e-8)


def test_fit_mask_and_missing_energies():
    z, num_atoms, target = dataset_arrays(
        COMPOSITIONS, [-10.0, -12.0, -4.0, -7.0, -9.0, -20.0]
    )
    predicted = target - num_atoms * 0.5
    predicted[3] = 100.0  # wrong prediction, masked
    target[4] = np.nan  # the only other structure with iron

    mask = np.array([True, True, True, False, True, True])
    correction, counts = calibration.fit_element_correction(
        z, num_atoms, predicted, target, mask
    )

    np.testing.assert_allclose(correction[[1, 8]], [0.5, 0.5], atol=1e-9)
    # iron is in no fitted structure
    assert correction[26] == 0.0 and counts[26] == 0
    assert counts[[1, 8]].tolist() == [4, 3]


def test_fit_is_the_least_squares_solution():
    rng = np.random.default_rng(0)
    compositions = [
        {
            int(el): int(rng.integers(1, 5))
            for el in rng.choice([1, 6, 8, 26], size=rng.integers(1, 4), replace=False)
        }
        for _ in range(200)
    ]
    z, num_atoms, target = dataset_arrays(compositions, rng.normal(size=200))
    predicted = rng.normal(size=200)

    correction, _ = calibration.fit_element_correction(z, num_atoms, predicted, target)

    def loss(values):
        corrected = calibration.apply_element_correction(
            values, z, num_atoms, predicted
        )
        return np.square((corrected - target) / num_atoms).sum()

    best = loss(correction)
    for element in (1, 6, 8, 26):
        for step in (-1e-3, 1e-3):
            moved = correction.copy()
            moved[element] += step
            assert loss(moved) > best


def test_correction_file(tmp_path):
    correction, counts = np.zeros(128), np.zeros(128, dtype=np.int64)
    correction[[1, 8]], counts[[1, 8]] = [1.0, -0.5], [3, 2]

    path = tmp_path / "correction.hdf5"
    calibration.save_correction(
        path, correction, counts, model="orb", mae=0.014, structures=5
    )
    loaded, loaded_counts, attributes = calibration.load_correction(path)

    np.testing.assert_array_equal(loaded, correction)
    np.testing.assert_array_equal(loaded_counts, counts)
    assert attributes == {"model": "orb", "mae": 0.014, "structures": 5}

    with h5py.File(tmp_path / "other.hdf5", "w") as data:
        data.create_group("correction")
        data["counts"] = counts
    with pytest.raises(KeyError, match="not a correction"):
        calibration.load_correction(tmp_path / "other.hdf5")


def test_error_report():
    num_atoms = np.array([2, 4, 1, 5])
    target = np.array([-2.0, -8.0, -3.0, np.nan])
    predicted = target + num_atoms * np.array([0.1, -0.3, 0.2, 0.0])

    report = calibration.error_report(predicted, target, num_atoms)

    assert report["count"] == 3
    assert report["mae"] == pytest.approx(0.2)
    assert report["median"] == pytest.approx(0.2)
    assert report["max"] == pytest.approx(0.3)
    assert report["bias"] == pytest.approx(0.0)
    assert "stable_found" not in report


def test_error_report_stability():
    num_atoms = np.ones(5, dtype=np.int64)
    above_hull = np.array([0.0, 0.0, 0.05, 0.3, 0.08])
    error = np.array([0.0, 0.02, -0.06, 0.0, 0.05])

    report = calibration.error_report(error, np.zeros(5), num_atoms, above_hull)

    # stable: structures 0 and 1; predicted stable: 0 and 2
    assert report["stable_count"] == 2
    assert report["stable_found"] == pytest.approx(0.5)
    assert report["stable_false"] == pytest.approx(0.5)
    # only the last structure changes side of the 0.1 eV/atom threshold
    assert report["metastable_agreement"] == pytest.approx(0.8)


# estimator


def test_atom_batches():
    assert atom_batches(np.array([4, 4, 4, 10, 1, 1]), 8) == [(0, 2), (2, 4), (4, 6)]
    assert atom_batches(np.array([1, 1, 1]), 100) == [(0, 3)]
    assert atom_batches(np.array([50, 1]), 10) == [(0, 1), (1, 2)]
    assert atom_batches(np.zeros(0, dtype=int), 10) == []


def test_to_device():
    structures = Structures.diamond()
    moved = to_device(structures, "cpu")

    assert moved is not structures
    assert torch.equal(moved.x, structures.x) and torch.equal(moved.ptr, structures.ptr)
    assert torch.isnan(moved.formation_energy_per_atom).all()


@pytest.fixture(scope="module")
def float_model():
    torch.manual_seed(0)
    return OrbV3().eval().requires_grad_(False)


def test_estimator_adds_the_correction(float_model):
    structures = Structures.collate([Structures.zincblende(), Structures.diamond()])
    correction = np.zeros(128)
    correction[[6, 16, 30]] = [0.5, -1.0, 2.0]

    raw = EnergyEstimator(float_model)(structures)
    corrected = EnergyEstimator(float_model, correction)(structures)

    # zincblende: 4 S and 4 Zn, diamond: 8 C
    assert torch.allclose(
        corrected - raw, torch.tensor([4.0, 4.0], dtype=torch.float64)
    )
    assert corrected.dtype == torch.float64


@pytest.mark.timeout(60)
def test_estimator_predict(tmp_path, float_model):
    parts = [
        Structures.diamond(),
        Structures.zincblende(),
        tilted(),
        Structures.diamond(a=3.7),
    ]
    HDF5Dataset.from_structures(tmp_path / "data.hdf5", [Structures.collate(parts)])
    dataset = HDF5Dataset(tmp_path / "data.hdf5")
    estimator = EnergyEstimator(float_model)

    expected = torch.cat([estimator(part) for part in parts]).numpy()

    np.testing.assert_allclose(
        estimator.predict(dataset, max_atoms=10), expected, rtol=1e-5
    )
    np.testing.assert_allclose(
        estimator.predict(dataset, max_atoms=1000), expected, rtol=1e-5
    )
    np.testing.assert_allclose(
        estimator.predict(dataset, np.array([1, 3])), expected[[1, 3]], rtol=1e-5
    )
    assert estimator.predict(dataset, np.zeros(0, dtype=int)).shape == (0,)

    with pytest.raises(ValueError, match="ascending"):
        estimator.predict(dataset, np.array([2, 1]))

    # a subset of the file: its structures, not the first rows of the file.
    # They are in other batches than above, hence the wider tolerance.
    subset = dataset.subset([3, 1, 2])
    np.testing.assert_allclose(
        estimator.predict(subset, max_atoms=10), expected[[3, 1, 2]], rtol=1e-4
    )
    np.testing.assert_allclose(
        estimator.predict(subset, np.array([0, 2])), expected[[3, 2]], rtol=1e-4
    )


def test_estimator_correction_of_another_model(tmp_path):
    path = tmp_path / "correction.hdf5"
    calibration.save_correction(
        path,
        np.zeros(128),
        np.zeros(128, dtype=np.int64),
        model="orb-v3-conservative-inf-mpa",
    )

    with pytest.raises(ValueError, match="fitted for"):
        EnergyEstimator.from_pretrained("orb-v3-conservative-inf-omat", path)


def test_fit_absolute_loss_ignores_outliers():
    rng = np.random.default_rng(1)
    compositions = [
        {
            int(el): int(rng.integers(1, 4))
            for el in rng.choice([1, 8, 26], size=2, replace=False)
        }
        for _ in range(300)
    ]
    shift = {1: 0.2, 8: -0.1, 26: 0.05}
    z, num_atoms, target = dataset_arrays(compositions, rng.normal(size=300))
    predicted = target - np.array(
        [sum(shift[el] * n for el, n in c.items()) for c in compositions]
    )
    # a few failed calculations with iron
    outliers = [i for i, c in enumerate(compositions) if 26 in c][:8]
    predicted[outliers] += 50.0

    squared, _ = calibration.fit_element_correction(z, num_atoms, predicted, target)
    absolute, counts = calibration.fit_element_correction(
        z, num_atoms, predicted, target, loss="absolute"
    )

    np.testing.assert_allclose(absolute[[1, 8, 26]], [0.2, -0.1, 0.05], atol=2e-3)
    assert abs(squared[26] - 0.05) > 0.1
    assert counts[[1, 8, 26]].min() > 0

    with pytest.raises(ValueError, match="loss"):
        calibration.fit_element_correction(
            z, num_atoms, predicted, target, loss="huber"
        )


# energy above the hull


def hull_file(tmp_path, model: OrbV3, shift: float = 0.0):
    """Hull of zincblende ZnS, of its elements and of sodium, from the model."""
    from torchms.data.hull import save_hull

    zinc = Structures.diamond(a=5.6, element=30)
    sulfur = Structures.diamond(a=5.6, element=16)
    sodium = Structures.diamond(a=7.0, element=11)
    parts = [zinc, sulfur, sodium, Structures.zincblende()]
    batch = Structures.collate(parts)
    energy = EnergyEstimator(model)(batch).numpy()
    # zincblende is put below the line between its elements
    energy[3] = (energy[0] + energy[1]) / 2 - 8 * (0.5 + shift)

    path = tmp_path / "hull.hdf5"
    save_hull(path, batch.z.numpy(), batch.num_atoms.numpy(), energy)
    return path, energy


def test_energy_above_hull(tmp_path, float_model):
    from torchms.data.hull import energy_above_reference, load_hull

    path, reference = hull_file(tmp_path, float_model)
    estimator = EnergyEstimator(float_model, hull=load_hull(path))

    structures = Structures.collate(
        [
            Structures.diamond(a=5.6, element=30),  # on the hull
            Structures.zincblende(),  # above: its energy was lowered in the hull
            Structures.diamond(a=5.4, element=30),  # another energy for zinc
            Structures.diamond(),  # carbon is not in the hull
        ]
    )
    above = estimator.energy_above_hull(structures)

    assert above.shape == (4,) and above.dtype == torch.float64
    assert above[0].abs() < 1e-6
    # zincblende: its energy in the hull is 0.5 eV/atom below the line between
    # its elements, the estimated one is at some distance of this line
    elements = estimator(
        Structures.collate(
            [
                Structures.diamond(a=5.6, element=30),
                Structures.diamond(a=5.6, element=16),
            ]
        )
    )
    from_line = estimator(Structures.zincblende())[0] / 8 - elements.mean() / 8
    assert torch.isclose(above[1], 0.5 + from_line, atol=1e-5)
    assert above[2].abs() > 1e-6
    assert torch.isnan(above[3])

    expected = energy_above_reference(
        load_hull(path),
        structures.z.numpy(),
        structures.num_atoms.numpy(),
        estimator(structures).numpy(),
    )
    np.testing.assert_allclose(above.numpy(), expected, equal_nan=True)


def test_energy_above_hull_can_be_negative(tmp_path, float_model):
    from torchms.data.hull import load_hull

    # in this hull, zincblende is 0.2 eV/atom above the line between its elements
    path, _ = hull_file(tmp_path, float_model, shift=-0.7)
    estimator = EnergyEstimator(float_model, hull=load_hull(path))

    above = estimator.energy_above_hull(Structures.zincblende())

    # the estimated energy of zincblende is on this line: below it or above it
    # by the same amount as the reference
    line = (
        EnergyEstimator(float_model)(
            Structures.collate(
                [
                    Structures.diamond(a=5.6, element=30),
                    Structures.diamond(a=5.6, element=16),
                ]
            )
        ).mean()
        / 8
    )
    energy = EnergyEstimator(float_model)(Structures.zincblende()) / 8
    assert torch.isclose(above[0], (energy - line)[0], atol=1e-5)


def test_energy_above_hull_needs_a_hull(float_model):
    estimator = EnergyEstimator(float_model)

    with pytest.raises(ValueError, match="hull"):
        estimator.energy_above_hull(Structures.diamond())


@pytest.mark.timeout(60)
def test_predict_energy_above_hull(tmp_path, float_model):
    from torchms.data.hull import load_hull

    path, _ = hull_file(tmp_path, float_model)
    estimator = EnergyEstimator(float_model, hull=load_hull(path))

    parts = [
        Structures.zincblende(),
        Structures.diamond(a=5.4, element=30),
        Structures.diamond(),
        Structures.diamond(a=5.6, element=16),
        Structures.zincblende(a=5.6),
    ]
    HDF5Dataset.from_structures(tmp_path / "data.hdf5", [Structures.collate(parts)])
    dataset = HDF5Dataset(tmp_path / "data.hdf5")

    energy, above = estimator.predict_energy_above_hull(dataset, max_atoms=10)
    expected = torch.cat([estimator.energy_above_hull(part) for part in parts]).numpy()

    np.testing.assert_allclose(energy, estimator.predict(dataset), rtol=1e-4)
    np.testing.assert_allclose(above, expected, atol=1e-5, equal_nan=True)
    assert np.isnan(above[2]) and abs(above[3]) < 1e-6

    energy, above = estimator.predict_energy_above_hull(dataset, np.array([1, 4]))
    np.testing.assert_allclose(above, expected[[1, 4]], atol=1e-5)
    assert energy.shape == (2,)

    # a batch of Structures gives what the dataset gives
    batch = Structures.collate(parts)
    energy, above = estimator.predict_energy_above_hull(batch, max_atoms=10)
    np.testing.assert_allclose(energy, estimator.predict(dataset), rtol=1e-4)
    np.testing.assert_allclose(above, expected, atol=1e-5, equal_nan=True)

    energy, above = estimator.predict_energy_above_hull(batch, np.array([1, 4]))
    np.testing.assert_allclose(above, expected[[1, 4]], atol=1e-5)
    assert energy.shape == (2,)


def test_from_pretrained_loads_correction_and_hull(tmp_path, float_model, monkeypatch):
    path, _ = hull_file(tmp_path, float_model)
    correction = np.zeros(128)
    correction[30] = 0.25
    calibration.save_correction(
        tmp_path / "correction.hdf5",
        correction,
        np.zeros(128, dtype=np.int64),
        model="orb-v3-conservative-inf-mpa",
    )
    loaded = []

    def from_pretrained(name, device=None):
        loaded.append((name, device))
        return float_model

    monkeypatch.setattr(OrbV3, "from_pretrained", from_pretrained)

    estimator = EnergyEstimator.from_pretrained(
        correction_file=tmp_path / "correction.hdf5", hull_file=path
    )

    assert loaded == [("orb-v3-conservative-inf-mpa", None)]
    assert estimator.hull is not None and (30, 16)[::-1] in estimator.hull.stable
    # 8 zinc atoms: the correction moves zinc 0.25 eV/atom above its reference
    above = estimator.energy_above_hull(Structures.diamond(a=5.6, element=30))
    assert torch.isclose(above[0], torch.tensor(0.25, dtype=torch.float64), atol=1e-5)


def test_from_lematerial_downloads_correction_and_hull(
    tmp_path, float_model, monkeypatch
):
    import sys
    import types

    path, _ = hull_file(tmp_path, float_model)
    calibration.save_correction(
        tmp_path / "correction.hdf5",
        np.zeros(128),
        np.zeros(128, dtype=np.int64),
        model="orb-v3-conservative-inf-mpa",
    )
    files = {
        "energy_correction_orb.hdf5": tmp_path / "correction.hdf5",
        "hull.hdf5": path,
    }
    calls = []

    def hf_hub_download(repo_id, filename, repo_type=None):
        calls.append((repo_id, filename, repo_type))
        return str(files[filename])

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(OrbV3, "from_pretrained", lambda name, device=None: float_model)

    estimator = EnergyEstimator.from_lematerial()

    assert calls == [
        ("materials-toolkits/lematerial", "energy_correction_orb.hdf5", "dataset"),
        ("materials-toolkits/lematerial", "hull.hdf5", "dataset"),
    ]
    assert estimator.hull is not None and estimator.correction is not None
    assert estimator.energy_above_hull(Structures.zincblende()).shape == (1,)

    # where the hull comes from
    for expected in (
        "LeMat-Bulk compatible_pbe",
        "Materials Project",
        "Alexandria",
        "OQMD left out",
        "hull.hdf5",
        "materials-toolkits/lematerial",
    ):
        assert expected in estimator.hull_source
        assert expected in repr(estimator)

    calls.clear()
    without = EnergyEstimator.from_lematerial(hull=False)
    assert calls == [
        ("materials-toolkits/lematerial", "energy_correction_orb.hdf5", "dataset")
    ]
    assert without.hull is None and without.hull_source is None
    assert "hull=none" in repr(without)
    with pytest.raises(ValueError, match="from_lematerial"):
        without.energy_above_hull(Structures.zincblende())


def test_hull_source_of_a_file(tmp_path, float_model, monkeypatch):
    path, _ = hull_file(tmp_path, float_model)
    monkeypatch.setattr(OrbV3, "from_pretrained", lambda name, device=None: float_model)

    estimator = EnergyEstimator.from_pretrained(hull_file=path)
    assert estimator.hull_source == f"file {path}"

    from torchms.data.hull import load_hull

    assert EnergyEstimator(float_model, hull=load_hull(path)).hull_source is None
    assert "unknown source" in repr(EnergyEstimator(float_model, hull=load_hull(path)))
    assert (
        EnergyEstimator(float_model, hull_source="ignored without hull").hull_source
        is None
    )
