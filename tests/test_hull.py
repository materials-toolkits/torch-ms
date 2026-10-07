"""Tests for the computation of the energy above the convex hull."""

import h5py
import numpy as np
import pytest
import torch

from torchms.data import HDF5Dataset, Structures
from torchms.data.hull import (
    STABLE_TOLERANCE,
    Hull,
    compositions,
    energy_above_hull,
    energy_above_reference,
    fill_missing,
    fractions,
    hull_energy,
    hull_energy_lp,
    load_hull,
    save_hull,
)


def build(structures):
    """structures: list of ({z: count}, energy per atom)"""
    z, num_atoms, total_energy = [], [], []
    for composition, energy in structures:
        atoms = [el for el, count in composition.items() for _ in range(count)]
        z += atoms[::-1]  # atoms are not sorted in a dataset
        num_atoms.append(len(atoms))
        total_energy.append(energy * len(atoms))
    return np.array(z), np.array(num_atoms), np.array(total_energy)


def above(structures, **kwargs):
    return energy_above_hull(*build(structures), **kwargs)


def test_compositions():
    z, num_atoms, _ = build([({11: 1, 17: 3}, 0.0), ({26: 2}, 0.0)])
    elements, counts, ptr = compositions(z, num_atoms)

    assert elements.tolist() == [11, 17, 26]
    assert counts.tolist() == [1, 3, 2]
    assert ptr.tolist() == [0, 2, 3]
    assert fractions(counts, ptr).tolist() == [0.25, 0.75, 1.0]


def test_elements():
    result = above([({1: 2}, -3.0), ({1: 4}, -3.5), ({2: 1}, -1.0)])

    assert result.tolist() == [0.5, 0.0, 0.0]


def test_binary():
    structures = [
        ({1: 1}, -1.0),
        ({2: 1}, -2.0),
        ({1: 1, 2: 1}, -2.5),  # formation -1.0: stable
        ({1: 2, 2: 2}, -2.2),  # same composition, 0.3 above
        ({1: 3, 2: 1}, -1.5),  # tie line A-AB at x=0.25: -1.75
        ({1: 1, 2: 3}, -1.0),  # formation +0.75, tie line AB-B: -2.25
    ]

    np.testing.assert_allclose(
        above(structures), [0.0, 0.0, 0.0, 0.3, 0.25, 1.25], atol=1e-12
    )


def test_no_stable_compound():
    structures = [({1: 1}, -1.0), ({2: 1}, -2.0), ({1: 1, 2: 1}, -1.2)]

    np.testing.assert_allclose(above(structures), [0.0, 0.0, 0.3], atol=1e-12)


def test_ternary_uses_stable_binaries():
    structures = [
        ({1: 1}, 0.0),
        ({2: 1}, 0.0),
        ({3: 1}, 0.0),
        ({1: 1, 2: 1}, -1.0),  # stable binary
        ({1: 1, 3: 1}, 0.5),  # unstable binary
        ({1: 1, 2: 1, 3: 2}, -0.2),  # on the line AB - C the hull is -0.5
        ({1: 1, 2: 1, 3: 1}, -1.0),  # below the line AB - C (-2/3): stable
    ]

    result = above(structures)

    np.testing.assert_allclose(result[:5], [0.0, 0.0, 0.0, 0.0, 0.5], atol=1e-12)
    # with ABC stable, the hull at ABC2 is between ABC (-1) and C (0): -0.75
    np.testing.assert_allclose(result[5:], [0.55, 0.0], atol=1e-12)


def test_missing_energy_and_missing_element():
    structures = [
        ({1: 1}, -1.0),
        ({2: 1}, -2.0),
        ({1: 1, 2: 1}, float("nan")),
        ({1: 1, 2: 1}, -2.5),
        ({1: 1, 3: 1}, -5.0),  # no structure of element 3 alone
    ]

    result = above(structures)

    assert np.isnan(result[[2, 4]]).all()
    np.testing.assert_allclose(result[[0, 1, 3]], 0.0, atol=1e-12)


def random_dataset(seed, num_elements=4, num_structures=300):
    rng = np.random.default_rng(seed)
    structures = [({el: 1}, rng.uniform(-3, -1)) for el in range(1, num_elements + 1)]
    structures += [({el: 2}, rng.uniform(-1, 0)) for el in range(1, num_elements + 1)]

    for _ in range(num_structures):
        size = rng.integers(2, num_elements + 1)
        chosen = rng.choice(np.arange(1, num_elements + 1), size=size, replace=False)
        composition = {int(el): int(rng.integers(1, 5)) for el in chosen}
        structures.append((composition, rng.uniform(-4.0, 0.0)))

    return structures


def reference_lp(structures):
    """Energy above the hull with a linear program over the whole dataset."""
    elements = sorted({el for composition, _ in structures for el in composition})
    fractions = np.array(
        [
            [composition.get(el, 0) / sum(composition.values()) for el in elements]
            for composition, _ in structures
        ]
    )
    energies = np.array([energy for _, energy in structures])

    hull = hull_energy_lp(fractions[:, :-1], energies, fractions[:, :-1])
    return energies - hull


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.timeout(60)
def test_matches_linear_program(seed):
    structures = random_dataset(seed)

    np.testing.assert_allclose(above(structures), reference_lp(structures), atol=1e-9)


@pytest.mark.timeout(60)
def test_parallel_matches_sequential():
    structures = random_dataset(3, num_elements=5, num_structures=400)

    sequential = above(structures)
    parallel = above(structures, num_workers=3)

    np.testing.assert_array_equal(sequential, parallel)
    assert (sequential == 0.0).sum() >= 5


def test_hull_energy_without_negative_points():
    points = np.eye(3)[:, :2]

    assert hull_energy(points, np.zeros(3), np.array([[0.2, 0.3]])).tolist() == [0.0]


def test_fill_missing(tmp_path):
    def structure(z, energy_per_atom, hull):
        return Structures(
            lattice=torch.eye(3).unsqueeze(0) * 4.0,
            x=torch.rand(len(z), 3),
            z=torch.tensor(z),
            total_energy=torch.tensor([energy_per_atom * len(z)]),
            energy_above_hull=torch.tensor([hull]),
        )

    nan = float("nan")
    structures = [
        structure([1], -1.0, 0.0),
        structure([2], -2.0, nan),
        structure([1, 2], -2.5, 0.0),
        structure([1, 1, 2, 2], -2.2, nan),  # 0.3 above the hull
        structure([1, 2, 2, 2], -1.0, 1.25),
        structure([1, 1, 1, 2], -1.5, 0.75),  # stored value is wrong: kept
    ]
    hdf5_file = tmp_path / "data.hdf5"
    HDF5Dataset.from_structures(hdf5_file, [Structures.collate(structures)])

    computed, stats = fill_missing(hdf5_file)

    np.testing.assert_allclose(computed, [0.0, 0.0, 0.0, 0.3, 1.25, 0.25], atol=1e-6)
    assert stats["filled"] == 2
    assert stats["still_missing"] == 0
    assert stats["compared"] == 4
    assert stats["max_difference"] == pytest.approx(0.5, abs=1e-6)
    assert stats["differences_above_1e-4"] == 1

    with h5py.File(hdf5_file, "r") as data:
        stored = data["energy_above_hull"][:]
    np.testing.assert_allclose(stored, [0.0, 0.0, 0.0, 0.3, 1.25, 0.75], atol=1e-6)

    # nothing left to fill the second time
    assert fill_missing(hdf5_file)[1]["filled"] == 0


BINARY = [
    ({1: 1}, -1.0),
    ({1: 2}, -0.5),
    ({2: 1}, -2.0),
    ({1: 1, 2: 1}, -2.5),  # stable
    ({1: 2, 2: 2}, -2.2),
    ({1: 3, 2: 1}, -1.5),
]


def test_save_hull_format(tmp_path):
    result = save_hull(tmp_path / "hull.hdf5", *build(BINARY))

    np.testing.assert_allclose(result, [0.0, 0.5, 0.0, 0.0, 0.3, 0.25], atol=1e-12)

    with h5py.File(tmp_path / "hull.hdf5", "r") as data:
        assert sorted(data.keys()) == ["counts", "elements", "energy_per_atom", "ptr"]
        assert data["elements"][:].tolist() == [1, 2, 1, 2]
        assert data["counts"][:].tolist() == [1, 1, 1, 1]
        assert data["ptr"][:].tolist() == [0, 1, 2, 4]
        assert data["energy_per_atom"][:].tolist() == [-1.0, -2.0, -2.5]
        assert data["energy_per_atom"].dtype == np.float64


def test_reference_reproduces_the_dataset(tmp_path):
    structures = random_dataset(4, num_elements=4, num_structures=300)
    expected = save_hull(tmp_path / "hull.hdf5", *build(structures))

    hull = load_hull(tmp_path / "hull.hdf5")
    result = energy_above_reference(hull, *build(structures))

    np.testing.assert_allclose(result, expected, atol=1e-9)
    assert ((result == 0.0) == (expected == 0.0)).all()


def test_reference_queries(tmp_path):
    save_hull(tmp_path / "hull.hdf5", *build(BINARY))
    hull = load_hull(tmp_path / "hull.hdf5")

    queries = [
        ({1: 1, 2: 1}, -2.5),  # the stable phase itself
        ({1: 1, 2: 1}, -3.0),  # more stable than the hull
        ({1: 3, 2: 1}, -1.5),  # tie line A-AB at x=0.25: -1.75
        ({1: 3}, -1.2),  # element below its reference
        ({2: 2}, -1.5),
        ({1: 1, 3: 1}, -1.0),  # element 3 is not in the hull
        ({1: 1, 2: 1}, float("nan")),
    ]
    result = energy_above_reference(hull, *build(queries))

    np.testing.assert_allclose(result[:5], [0.0, -0.5, 0.25, -0.2, 0.5], atol=1e-12)
    assert np.isnan(result[5:]).all()

    # the reference is not modified by the queries
    again = energy_above_reference(hull, *build(queries[:3]))
    np.testing.assert_allclose(again, [0.0, -0.5, 0.25], atol=1e-12)


def test_reference_system_absent_from_the_hull(tmp_path):
    structures = [
        ({1: 1}, 0.0),
        ({2: 1}, 0.0),
        ({3: 1}, 0.0),
        ({1: 1, 2: 1}, -1.0),
    ]
    save_hull(tmp_path / "hull.hdf5", *build(structures))
    hull = load_hull(tmp_path / "hull.hdf5")

    # no ternary structure in the hull: resolved from AB and C
    queries = [
        ({1: 1, 2: 1, 3: 2}, -0.2),
        ({1: 1, 2: 1, 3: 2}, -0.9),
        ({1: 1, 3: 1}, 0.1),
    ]
    result = energy_above_reference(hull, *build(queries))

    np.testing.assert_allclose(result, [0.3, -0.4, 0.1], atol=1e-12)


@pytest.mark.timeout(60)
def test_reference_parallel(tmp_path):
    structures = random_dataset(5, num_elements=5, num_structures=400)
    save_hull(tmp_path / "hull.hdf5", *build(structures), num_workers=3)
    hull = load_hull(tmp_path / "hull.hdf5")

    queries = random_dataset(6, num_elements=5, num_structures=200)
    sequential = energy_above_reference(hull, *build(queries))
    parallel = energy_above_reference(hull, *build(queries), num_workers=3)

    np.testing.assert_array_equal(sequential, parallel)
    assert (sequential < 0).any() and (sequential > 0).any()


# unit tests of the building blocks


def test_compositions_single_structure_and_empty():
    elements, counts, ptr = compositions(np.array([8, 8, 1, 8, 1]), np.array([5]))
    assert (elements.tolist(), counts.tolist(), ptr.tolist()) == (
        [1, 8],
        [2, 3],
        [0, 2],
    )

    elements, counts, ptr = compositions(np.zeros(0, dtype=np.int64), np.zeros(0, int))
    assert (len(elements), len(counts), ptr.tolist()) == (0, 0, [0])


def test_compositions_same_elements_in_several_structures():
    z = np.array([1, 8, 8, 8, 1, 1, 6])
    elements, counts, ptr = compositions(z, np.array([2, 4, 1]))

    assert elements.tolist() == [1, 8, 1, 8, 6]
    assert counts.tolist() == [1, 1, 2, 2, 1]
    assert ptr.tolist() == [0, 2, 4, 5]


def test_fractions_sum_to_one_per_structure():
    counts = np.array([1, 3, 2, 5, 5, 10])
    ptr = np.array([0, 2, 3, 6])

    result = fractions(counts, ptr)

    np.testing.assert_allclose(result, [0.25, 0.75, 1.0, 0.25, 0.25, 0.5])
    np.testing.assert_allclose(np.add.reduceat(result, ptr[:-1]), 1.0)


def test_hull_defaults():
    hull = Hull()

    assert hull.reference.shape == (128,)
    assert np.isnan(hull.reference).all()
    assert hull.stable == {}
    assert Hull().reference is not hull.reference


def test_hull_energy_binary():
    # elements at x=1 and x=0, compounds at x=0.5 (-1.0) and x=0.25 (-0.2)
    points = np.array([[1.0], [0.0], [0.5], [0.25]])
    energies = np.array([0.0, 0.0, -1.0, -0.2])
    queries = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])

    result = hull_energy(points, energies, queries)

    # the compound at x=0.25 is above the line between x=0 and x=0.5
    np.testing.assert_allclose(result, [0.0, -0.5, -1.0, -0.5, 0.0], atol=1e-12)


def test_hull_energy_ternary():
    # elements A, B, C and a compound ABC with a formation energy of -3
    points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1 / 3, 1 / 3]])
    energies = np.array([0.0, 0.0, 0.0, -3.0])
    queries = np.array([[1 / 3, 1 / 3], [0.5, 0.5], [2 / 3, 1 / 6], [0.0, 0.0]])

    result = hull_energy(points, energies, queries)

    np.testing.assert_allclose(result, [-3.0, 0.0, -1.5, 0.0], atol=1e-12)


def test_hull_energy_empty_queries():
    points = np.array([[1.0], [0.0], [0.5]])
    energies = np.array([0.0, 0.0, -1.0])

    assert hull_energy(points, energies, np.zeros((0, 1))).shape == (0,)


def test_hull_energy_is_never_positive():
    rng = np.random.default_rng(0)
    points = np.vstack([np.eye(3)[:, :2], rng.dirichlet(np.ones(3), size=30)[:, :2]])
    energies = np.concatenate([np.zeros(3), rng.uniform(-2, 0, size=30)])
    queries = rng.dirichlet(np.ones(3), size=200)[:, :2]

    result = hull_energy(points, energies, queries)

    assert (result <= 0).all()
    # the hull is below every point defining it
    assert (hull_energy(points, energies, points) <= energies + 1e-12).all()


@pytest.mark.parametrize("num_elements", [2, 3, 4, 5])
def test_hull_energy_matches_linear_program(num_elements):
    rng = np.random.default_rng(num_elements)
    compounds = rng.dirichlet(np.ones(num_elements), size=40)
    points = np.vstack([np.eye(num_elements), compounds])[:, :-1]
    energies = np.concatenate([np.zeros(num_elements), rng.uniform(-2, 0, size=40)])
    queries = rng.dirichlet(np.ones(num_elements), size=25)[:, :-1]

    np.testing.assert_allclose(
        hull_energy(points, energies, queries),
        hull_energy_lp(points, energies, queries),
        atol=1e-9,
    )


def test_hull_energy_degenerate_points_fall_back():
    # repeated points and compounds all on one line of a ternary system
    points = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.25, 0.75]]
    )
    energies = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -0.5])
    queries = np.array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25]])

    result = hull_energy(points, energies, queries)

    np.testing.assert_allclose(result, [-1.0, -0.5, -0.5], atol=1e-9)


def test_hull_energy_lp_accepts_positive_energies():
    points = np.array([[1.0], [0.0], [0.5]])
    energies = np.array([-1.0, -2.0, 3.0])

    result = hull_energy_lp(points, energies, np.array([[0.5], [1.0]]))

    np.testing.assert_allclose(result, [-1.5, -1.0], atol=1e-12)


def test_hull_energy_lp_query_outside_the_points():
    points = np.array([[1.0], [0.5]])

    with pytest.raises(RuntimeError, match="hull energy not found"):
        hull_energy_lp(points, np.array([0.0, -1.0]), np.array([[0.0]]))


def test_energy_above_hull_is_invariant():
    structures = random_dataset(7, num_elements=3, num_structures=60)
    expected = above(structures)

    # order of the structures
    permutation = np.random.default_rng(0).permutation(len(structures))
    shuffled = above([structures[i] for i in permutation])
    np.testing.assert_allclose(shuffled, expected[permutation], atol=1e-12)

    # size of the cells: same composition and energy per atom
    doubled = [({el: 2 * n for el, n in c.items()}, e) for c, e in structures]
    np.testing.assert_allclose(above(doubled), expected, atol=1e-12)

    # shift of the energy of an element
    shifted = [(c, e + 0.7 * c.get(1, 0) / sum(c.values())) for c, e in structures]
    np.testing.assert_allclose(above(shifted), expected, atol=1e-9)


def test_energy_above_hull_stable_structures_are_exactly_zero():
    result = above(random_dataset(8))

    assert (result >= 0).all()
    assert (result == 0.0).sum() >= 4
    assert not ((result > 0) & (result <= STABLE_TOLERANCE)).any()


def test_energy_above_hull_accepts_float32_and_lists():
    z, num_atoms, total_energy = build(BINARY)

    result = energy_above_hull(
        z.astype(np.int32), num_atoms.tolist(), total_energy.astype(np.float32)
    )

    np.testing.assert_allclose(result, [0.0, 0.5, 0.0, 0.0, 0.3, 0.25], atol=1e-6)


def test_load_hull(tmp_path):
    save_hull(tmp_path / "hull.hdf5", *build(BINARY))

    hull = load_hull(tmp_path / "hull.hdf5")

    assert hull.reference[1] == -1.0
    assert hull.reference[2] == -2.0
    assert np.isnan(np.delete(hull.reference, [1, 2])).all()
    assert list(hull.stable) == [(1, 2)]
    system_fractions, formation = hull.stable[(1, 2)]
    np.testing.assert_allclose(system_fractions, [[0.5, 0.5]])
    np.testing.assert_allclose(formation, [-1.0])


def test_load_hull_invalid_file(tmp_path):
    with h5py.File(tmp_path / "hull.hdf5", "w") as data:
        data.create_group("elements")

    with pytest.raises(KeyError, match="elements"):
        load_hull(tmp_path / "hull.hdf5")


def test_fill_missing_keeps_a_complete_dataset(tmp_path):
    structure = Structures(
        lattice=torch.eye(3).unsqueeze(0),
        x=torch.zeros(1, 3),
        z=torch.tensor([1]),
        total_energy=torch.tensor([-1.0]),
        energy_above_hull=torch.tensor([0.4]),
    )
    hdf5_file = tmp_path / "data.hdf5"
    HDF5Dataset.from_structures(hdf5_file, [structure])
    before = hdf5_file.read_bytes()

    computed, stats = fill_missing(hdf5_file)

    assert computed.tolist() == [0.0]
    assert stats["filled"] == 0
    assert stats["max_difference"] == pytest.approx(0.4)
    assert hdf5_file.read_bytes() == before


def test_command_line(tmp_path, capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from convex_hull import main

    def write(path, structures):
        z, num_atoms, total_energy = build(structures)
        with h5py.File(path, "w") as data:
            data["z"], data["num_atoms"], data["total_energy"] = (
                z,
                num_atoms,
                total_energy,
            )

    write(tmp_path / "data.hdf5", BINARY)
    write(tmp_path / "new.hdf5", [({1: 1, 2: 1}, -3.0), ({1: 3, 2: 1}, -1.5)])

    main(
        [
            str(tmp_path / "data.hdf5"),
            str(tmp_path / "above.npy"),
            "--save-hull",
            str(tmp_path / "hull.hdf5"),
            "--num-workers",
            "1",
        ]
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "above.npy"), [0.0, 0.5, 0.0, 0.0, 0.3, 0.25], atol=1e-12
    )

    main(
        [
            str(tmp_path / "new.hdf5"),
            str(tmp_path / "new.npy"),
            "--reference",
            str(tmp_path / "hull.hdf5"),
            "--num-workers",
            "1",
        ]
    )
    np.testing.assert_allclose(np.load(tmp_path / "new.npy"), [-0.5, 0.25], atol=1e-12)
    assert "1 below" in capsys.readouterr().out
