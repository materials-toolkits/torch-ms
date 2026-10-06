"""Tests for the estimation of the total energy with a pretrained model."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pycrystalgen.data import HDF5Dataset, Structures
from pycrystalgen.estimator import EnergyEstimator, OrbEnergyModel
from pycrystalgen.estimator import calibration, weights
from pycrystalgen.estimator.energy import atom_batches, to_device
from pycrystalgen.estimator.orb import (
    COVALENT_RADII,
    MAX_NEIGHBORS,
    RADIUS,
    UNUSED_PARAMETERS,
    BesselBasis,
    InvalidWeights,
    ZBLBasis,
    build_mlp,
    polynomial_cutoff,
    segment_sum,
    spherical_harmonics,
)

REFERENCE = Path(__file__).parent / "data" / "orb_reference.npz"


# weights


def test_checkpoint_path(tmp_path, monkeypatch):
    path = weights.checkpoint_path("orb-v3-conservative-inf-omat", tmp_path)
    assert path == tmp_path / "orb" / "orb-v3-conservative-inf-omat-20250404.ckpt"

    monkeypatch.setenv("PYCRYSTALGEN_CACHE", str(tmp_path / "cache"))
    assert weights.cache_directory() == tmp_path / "cache"
    assert weights.checkpoint_path("orb-v3-conservative-inf-mpa").parent == (
        tmp_path / "cache" / "orb"
    )

    monkeypatch.delenv("PYCRYSTALGEN_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert weights.cache_directory() == tmp_path / "xdg" / "pycrystalgen"

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_path("orb-v9")


def test_download_checks_the_content(tmp_path, monkeypatch):
    source = tmp_path / "weights.ckpt"
    source.write_bytes(b"pretrained weights")
    good = weights.Checkpoint(source.as_uri(), weights.sha256(source))
    bad = weights.Checkpoint(source.as_uri(), "0" * 64)
    monkeypatch.setitem(weights.CHECKPOINTS, "good", good)
    monkeypatch.setitem(weights.CHECKPOINTS, "bad", bad)

    path = weights.download("good", tmp_path / "cache")
    assert path.read_bytes() == b"pretrained weights"
    assert not path.with_name(path.name + ".part").exists()

    # not downloaded again
    source.unlink()
    assert weights.download("good", tmp_path / "cache") == path

    with pytest.raises(weights.InvalidCheckpoint, match="SHA-256"):
        weights.download("bad", tmp_path / "cache")


def test_pinned_checkpoints():
    assert set(weights.CHECKPOINTS) == {
        "orb-v3-conservative-inf-omat",
        "orb-v3-conservative-inf-mpa",
    }
    for checkpoint in weights.CHECKPOINTS.values():
        assert checkpoint.url.startswith("https://")
        assert len(checkpoint.sha256) == 64


# building blocks of the network


def test_polynomial_cutoff():
    distance = torch.linspace(0.0, 8.0, 801, dtype=torch.float64)
    envelope = polynomial_cutoff(distance, 6.0, p=4)

    assert envelope[0] == 1.0
    assert (envelope[distance >= 6.0] == 0.0).all()
    assert ((envelope >= 0.0) & (envelope <= 1.0)).all()
    assert (envelope[1:] <= envelope[:-1] + 1e-12).all()
    # smooth at the radius: value and slope go to zero
    assert envelope[599] < 1e-5

    radius = torch.tensor([2.0, 4.0])
    per_edge = polynomial_cutoff(torch.tensor([3.0, 3.0]), radius, p=6)
    assert per_edge[0] == 0.0 and 0.0 < per_edge[1] < 1.0


def test_spherical_harmonics_norms_and_parity():
    generator = torch.Generator().manual_seed(0)
    vectors = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    harmonics = spherical_harmonics(vectors)

    assert harmonics.shape == (50, 16)
    start = 0
    for degree in range(4):
        block = harmonics[:, start : start + 2 * degree + 1]
        expected = torch.full((50,), 2.0 * degree + 1, dtype=torch.float64)
        assert torch.allclose(block.square().sum(dim=1), expected)
        # parity of the degree
        opposite = spherical_harmonics(-vectors)[:, start : start + 2 * degree + 1]
        assert torch.allclose(opposite, (-1) ** degree * block)
        start += 2 * degree + 1

    # only the direction matters
    assert torch.allclose(spherical_harmonics(3.0 * vectors), harmonics)
    # degree 1 is the direction, in the order x, y, z
    unit = torch.nn.functional.normalize(vectors, dim=1)
    assert torch.allclose(harmonics[:, 1:4], math.sqrt(3.0) * unit)


def test_spherical_harmonics_rotation():
    generator = torch.Generator().manual_seed(1)
    vectors = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )[0]

    harmonics = spherical_harmonics(vectors)
    rotated = spherical_harmonics(vectors @ rotation)

    # a rotation mixes the harmonics of a degree: scalar products are kept
    start = 0
    for degree in range(4):
        block = slice(start, start + 2 * degree + 1)
        assert torch.allclose(
            harmonics[:, block] @ harmonics[:, block].T,
            rotated[:, block] @ rotated[:, block].T,
            atol=1e-10,
        )
        start += 2 * degree + 1


def test_bessel_basis():
    basis = BesselBasis()
    distance = torch.tensor([1.0, 3.0, 6.0])
    values = basis(distance)

    assert values.shape == (3, 8)
    expected = math.sqrt(2.0 / 6.0) * math.sin(2 * math.pi * 3.0 / 6.0) / 3.0
    assert torch.isclose(values[1, 1], torch.tensor(expected), atol=1e-6)
    # every function is zero at the radius
    assert values[2].abs().max() < 1e-6
    assert set(basis.state_dict()) == {"bessel_weights", "r_max", "prefactor"}


def test_segment_sum():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([2, 0, 2])

    assert segment_sum(values, index, 4).tolist() == [
        [3.0, 4.0],
        [0.0, 0.0],
        [6.0, 8.0],
        [0.0, 0.0],
    ]
    assert segment_sum(values[:, 0], index, 3).tolist() == [3.0, 0.0, 6.0]


def test_build_mlp_names():
    mlp = build_mlp(4, [8, 8], 2)

    assert [name for name, _ in mlp.named_children()] == [
        "NN-0",
        "Act-0",
        "NN-1",
        "Act-1",
        "NN-2",
        "Act-2",
    ]
    assert isinstance(mlp[-1], torch.nn.Identity)
    assert mlp(torch.zeros(3, 4)).shape == (3, 2)


def test_pair_repulsion_only_at_short_distance():
    repulsion = ZBLBasis()
    z = torch.tensor([6, 6])
    batch, num_atoms = torch.tensor([0, 0]), torch.tensor([2])
    senders, receivers = torch.tensor([0, 1]), torch.tensor([1, 0])

    def energy(distance: float) -> float:
        vectors = torch.tensor([[distance, 0.0, 0.0], [-distance, 0.0, 0.0]])
        return float(repulsion(z, vectors, senders, receivers, batch, num_atoms))

    # cut at the covalent radii of the next element (nitrogen), see ZBLBasis
    limit = 2 * COVALENT_RADII[7]
    assert energy(limit + 0.01) == 0.0
    assert energy(1.0) > energy(1.2) > 0.0
    assert len(COVALENT_RADII) == 119


# the model, with random weights


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return OrbEnergyModel().double().eval().requires_grad_(False)


def double(structures: Structures) -> Structures:
    return Structures(
        lattice=structures.lattice.double(),
        x=structures.x.double(),
        z=structures.z,
        num_atoms=structures.num_atoms,
        _validate=True,
        batch=structures.batch,
        ptr=structures.ptr,
    )


def tilted(scale: float = 1.0) -> Structures:
    """Cell with three atoms. With a scale of 1 sodium and chlorine are close
    enough for the pair repulsion, not with a larger one."""
    return Structures(
        lattice=scale
        * torch.tensor([[[4.1, 0.0, 0.0], [0.6, 4.4, 0.0], [0.2, 0.5, 5.0]]]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.25, 0.7, 0.1]]),
        z=torch.tensor([11, 17, 8]),
    )


def supercell(structures: Structures) -> Structures:
    """Two cells along the first lattice vector."""
    half = structures.x * torch.tensor([0.5, 1.0, 1.0])
    return Structures(
        lattice=structures.lattice * torch.tensor([2.0, 1.0, 1.0]).view(1, 3, 1),
        x=torch.cat([half, half + torch.tensor([0.5, 0.0, 0.0])]),
        z=structures.z.repeat(2),
    )


def repulsion(model: OrbEnergyModel, structures: Structures) -> torch.Tensor:
    graph, vectors = model.edges(structures)
    senders, receivers = graph.edge_index
    return model.pair_repulsion_fn(
        structures.z,
        vectors,
        senders,
        receivers,
        structures.batch,
        structures.num_atoms,
    )


def test_model_output(model):
    structures = double(Structures.collate([Structures.diamond(), tilted()]))

    energy = model(structures)

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert torch.isfinite(energy).all()


def test_model_graph(model):
    graph, vectors = model.edges(double(Structures.diamond()))
    degree = torch.bincount(graph.edge_index[0], minlength=8)

    # diamond has more than 120 atoms closer than 6 Å
    assert degree.tolist() == [MAX_NEIGHBORS] * 8
    assert vectors.norm(dim=1).max() <= RADIUS
    assert torch.allclose(vectors.norm(dim=1), graph.edges_distances())

    graph, vectors = model.edges(double(tilted()))
    assert torch.bincount(graph.edge_index[0], minlength=3).max() < MAX_NEIGHBORS
    assert torch.allclose(vectors.norm(dim=1), graph.edges_distances())


def test_model_batch_equals_single(model):
    parts = [double(Structures.zincblende()), double(tilted())]

    batched = model(Structures.collate(parts))
    single = torch.cat([model(part) for part in parts])

    assert torch.allclose(batched, single, rtol=1e-10)


def test_model_invariances(model):
    structures = double(tilted())
    reference = model(structures)

    def variant(lattice=None, x=None, z=None) -> torch.Tensor:
        return model(
            Structures(
                lattice=structures.lattice if lattice is None else lattice,
                x=structures.x if x is None else x,
                z=structures.z if z is None else z,
            )
        )

    # order of the atoms
    order = torch.tensor([2, 0, 1])
    assert torch.allclose(
        variant(x=structures.x[order], z=structures.z[order]), reference
    )

    # translation of the atoms, also out of the cell
    assert torch.allclose(
        variant(x=structures.x + torch.tensor([0.3, 0.9, -0.4])), reference
    )

    # the energy depends on the elements
    assert not torch.allclose(variant(z=torch.tensor([11, 17, 16])), reference)


def test_model_energy_of_a_supercell(model):
    structures = double(tilted(scale=1.3))
    assert repulsion(model, structures) == 0.0

    assert torch.allclose(
        model(supercell(structures)), 2 * model(structures), rtol=1e-4
    )


def test_model_pair_repulsion_is_a_mean_over_atoms(model):
    """As in orb-models for these weights, the repulsion does not grow with the cell."""
    structures = double(tilted())
    doubled = supercell(structures)

    assert repulsion(model, structures) > 0.0
    assert torch.allclose(repulsion(model, doubled), repulsion(model, structures))

    rest = model(structures) - repulsion(model, structures)
    assert torch.allclose(
        model(doubled) - repulsion(model, doubled), 2 * rest, rtol=1e-4
    )


def test_load_pretrained_checks_the_parameters(model):
    state_dict = dict(model.state_dict())
    state_dict["heads.energy.reference.linear.weight"] = state_dict[
        "heads.energy.reference.linear.weight"
    ].flatten()
    state_dict.update(
        {f"{prefix}weight": torch.zeros(1) for prefix in UNUSED_PARAMETERS}
    )

    other = OrbEnergyModel().double()
    other.load_pretrained(state_dict)
    assert torch.allclose(other(double(tilted())), model(double(tilted())))

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "model.unknown.weight": torch.zeros(1)})

    del state_dict["model.atom_emb.embeddings.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


# the pretrained model against the real orb-models


def reference_structures() -> tuple[Structures, dict[str, np.ndarray]]:
    data = dict(np.load(REFERENCE))
    structures = Structures(
        lattice=torch.from_numpy(data["lattice"]),
        x=torch.from_numpy(data["x"]),
        z=torch.from_numpy(data["z"]),
        num_atoms=torch.from_numpy(data["num_atoms"]),
    )
    return structures, data


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", ["omat", "mpa"])
def test_pretrained_matches_orb_models(name):
    """Energies of orb-models 0.7.0, computed once outside of the project."""
    full_name = f"orb-v3-conservative-inf-{name}"
    if not weights.checkpoint_path(full_name).exists():
        pytest.skip(f"weights of {full_name} are not in the cache")

    structures, data = reference_structures()
    pretrained = OrbEnergyModel.from_pretrained(full_name, device="cpu")

    graph, _ = pretrained.edges(structures)
    with torch.no_grad():
        energy = pretrained(structures).numpy()

    assert graph.num_edges.tolist() == data["num_edges"].tolist()

    error = np.abs(energy - data[f"energy_{name}"]) / data["num_atoms"]
    # atoms with more than 120 neighbours: equidistant ones are not kept in
    # the same order as orb-models
    capped = data["max_degree"] >= MAX_NEIGHBORS
    assert capped.sum() >= 3
    assert error[~capped].max() < 2e-5
    assert error[capped].max() < 5e-4


@pytest.mark.timeout(300)
def test_pretrained_is_almost_invariant_to_rotations():
    """The network is not equivariant: it learned the invariance of the energy."""
    name = "orb-v3-conservative-inf-omat"
    if not weights.checkpoint_path(name).exists():
        pytest.skip(f"weights of {name} are not in the cache")

    structures, data = reference_structures()
    pretrained = OrbEnergyModel.from_pretrained(name, device="cpu")
    generator = torch.Generator().manual_seed(0)
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    rotated = Structures(
        lattice=structures.lattice @ rotation,
        x=structures.x,
        z=structures.z,
        num_atoms=structures.num_atoms,
    )

    with torch.no_grad():
        change = (
            pretrained(rotated) - pretrained(structures)
        ).abs() / structures.num_atoms

    assert change.max() > 1e-7
    assert change.mean() < 1e-3
    assert change.max() < 2e-2


def test_pretrained_state_dict_is_fully_used():
    path = weights.checkpoint_path("orb-v3-conservative-inf-omat")
    if not path.exists():
        pytest.skip("weights are not in the cache")

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    used = {k for k in state_dict if not k.startswith(UNUSED_PARAMETERS)}

    assert used == set(OrbEnergyModel().state_dict())
    assert len(state_dict) - len(used) == 17


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
    return OrbEnergyModel().eval().requires_grad_(False)


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
