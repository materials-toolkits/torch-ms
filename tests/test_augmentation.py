import pytest
import torch

from pycrystalgen.data import Structures, parse_cif
from pycrystalgen.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
    Supercell,
    change_lattice_basis,
    make_supercell,
    permute_atoms,
    random_orthogonal,
    random_permutation,
    random_repeat,
    random_unimodular,
    rotate_lattice,
    translate_atoms,
)
from pycrystalgen.graph.atomic_radius import atomic_radius
from pycrystalgen.graph.coloring import color_refinement, refined_coloring
from pycrystalgen.graph.cutoff import cutoff

RADIUS = 3.0

# transforms which keep the atoms of the cell
TRANSFORMS = [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange]


def make_batch() -> Structures:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    batch = Structures.collate(
        [Structures.diamond(), triclinic, Structures.zincblende()]
    )
    batch.total_energy = torch.tensor([-8.0, float("nan"), -4.0])
    batch.band_gap = torch.tensor([5.5, 1.0, 3.6])
    return batch


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def environments(structures: Structures) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sorted species and distances of the pairs of neighbours of each structure."""
    graph = cutoff(structures, RADIUS, self_images=True)
    source, target = graph.edge_index
    species = structures.z[source] * 1000 + structures.z[target]
    dists = graph.edges_distances()

    result = []
    for i in range(len(structures.num_atoms)):
        mask = graph.edge_batch == i
        d, order = dists[mask].sort()
        s, order = species[mask][order].sort(stable=True)
        result.append((s, d[order]))

    return result


def assert_same_crystals(a: Structures, b: Structures, num_cells=None):
    """The pairs of neighbours of b are the ones of a, repeated num_cells times."""
    if num_cells is None:
        num_cells = torch.ones_like(a.num_atoms)

    assert (b.num_atoms == a.num_atoms * num_cells).all()

    for (s_a, d_a), (s_b, d_b), n in zip(
        environments(a), environments(b), num_cells.tolist()
    ):
        assert len(s_a) > 0
        assert len(s_b) == n * len(s_a)
        assert (s_b.view(-1, n) == s_a.unsqueeze(1)).all()
        assert torch.allclose(d_b.view(-1, n), d_a.unsqueeze(1), atol=1e-3)


def hashes(structures: Structures) -> torch.Tensor:
    """Hash of the graph of bonds of each structure."""
    return color_refinement(structures, atomic_radius(structures))


def assert_same_hashes(a: Structures, b: Structures):
    """The graphs of bonds of a and b have the same hash, structure by structure."""
    assert (hashes(a) == hashes(b)).all()


def colors(structures: Structures) -> torch.Tensor:
    """Color of each atom in the graph of bonds, after refinement."""
    return refined_coloring(structures, atomic_radius(structures))


def assert_same_colors(a: Structures, b: Structures, num_cells: torch.Tensor):
    """The colors of the atoms of b are the ones of a, repeated num_cells times."""
    colors_a, colors_b = colors(a), colors(b)

    for i, n in enumerate(num_cells.tolist()):
        unique_a, count_a = colors_a[a.batch == i].unique(dim=0, return_counts=True)
        unique_b, count_b = colors_b[b.batch == i].unique(dim=0, return_counts=True)

        assert unique_a.shape == unique_b.shape
        assert (unique_a == unique_b).all()
        assert (count_b == n * count_a).all()


def assert_valid(structures: Structures):
    structures.check_tensors()
    assert ((structures.x >= 0) & (structures.x < 1)).all()
    assert (structures.ptr[1:] - structures.ptr[:-1] == structures.num_atoms).all()
    assert (torch.bincount(structures.batch) == structures.num_atoms).all()


def test_permute_atoms():
    batch = make_batch()
    permutation = random_permutation(batch.batch, generator())
    permuted = permute_atoms(batch, permutation)

    assert (permutation != torch.arange(len(permutation))).any()
    assert (permutation.sort().values == torch.arange(len(permutation))).all()
    assert (permuted.z == batch.z[permutation]).all()
    assert (permuted.x == batch.x[permutation]).all()
    assert_valid(permuted)
    assert_same_crystals(batch, permuted)
    assert_same_hashes(batch, permuted)


def test_permute_atoms_across_structures():
    batch = make_batch()
    permutation = torch.arange(len(batch.z)).flip(0)

    with pytest.raises(ValueError):
        permute_atoms(batch, permutation)


def test_random_orthogonal():
    rotation = random_orthogonal(256, torch.device("cpu"), generator())
    identity = torch.eye(3).expand(256, 3, 3)
    det = torch.linalg.det(rotation)

    assert torch.allclose(
        torch.bmm(rotation, rotation.transpose(1, 2)), identity, atol=1e-5
    )
    assert torch.allclose(det.abs(), torch.ones(256), atol=1e-5)
    # rotations and rotoreflections
    assert (det > 0).any() and (det < 0).any()


def test_random_orthogonal_proper():
    rotation = random_orthogonal(256, torch.device("cpu"), generator(), proper=True)
    identity = torch.eye(3).expand(256, 3, 3)

    assert torch.allclose(
        torch.bmm(rotation, rotation.transpose(1, 2)), identity, atol=1e-5
    )
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(256), atol=1e-5)

    # same draws as O(3), the rotoreflections are negated
    full = random_orthogonal(256, torch.device("cpu"), generator())
    sign = torch.linalg.det(full).sign().view(-1, 1, 1)
    assert torch.allclose(rotation, full * sign, atol=1e-6)


def test_lattice_rotation_proper():
    batch = make_batch()
    batch = Structures.collate(Structures.separate(batch) * 8)
    rotated = LatticeRotation(proper=True, generator=generator())(batch)

    # the orientation of every lattice is preserved
    assert torch.allclose(
        torch.linalg.det(rotated.lattice), torch.linalg.det(batch.lattice), rtol=1e-4
    )
    assert_valid(rotated)
    assert_same_hashes(batch, rotated)


def test_rotate_lattice():
    batch = make_batch()
    rotation = random_orthogonal(3, torch.device("cpu"), generator())
    rotated = rotate_lattice(batch, rotation)

    # every lattice vector is rotated: v -> R v
    expected = torch.einsum("nij,nkj->nki", rotation, batch.lattice)
    assert torch.allclose(rotated.lattice, expected, atol=1e-5)
    assert not torch.allclose(rotated.lattice, batch.lattice, atol=1e-2)
    assert (rotated.x == batch.x).all()
    assert_valid(rotated)
    assert_same_crystals(batch, rotated)
    assert_same_hashes(batch, rotated)


def test_translate_atoms():
    batch = make_batch()
    translation = torch.tensor([[0.1, 0.2, 0.3], [-0.4, 1.5, 0.0], [0.99, 0.5, -2.2]])
    translated = translate_atoms(batch, translation)

    shift = translated.x - batch.x - translation[batch.batch]
    assert torch.allclose(shift, shift.round(), atol=1e-5)
    assert (translated.lattice == batch.lattice).all()
    assert_valid(translated)
    assert_same_crystals(batch, translated)
    assert_same_hashes(batch, translated)


def test_random_unimodular():
    basis = random_unimodular(256, 6, torch.device("cpu"), generator())

    assert basis.dtype == torch.int64
    assert (torch.linalg.det(basis.double()).round() == 1).all()
    assert (basis != torch.eye(3, dtype=torch.int64)).any(dim=(1, 2)).any()


def test_change_lattice_basis():
    batch = make_batch()
    basis = random_unimodular(3, 3, torch.device("cpu"), generator())
    sheared = change_lattice_basis(batch, basis)

    assert torch.allclose(
        torch.linalg.det(sheared.lattice), torch.linalg.det(batch.lattice), rtol=1e-4
    )
    assert not torch.allclose(sheared.lattice, batch.lattice, atol=1e-2)

    # cartesian positions are the same up to a vector of the lattice
    cart = torch.bmm(sheared.x.unsqueeze(1), sheared.lattice[sheared.batch]).squeeze(1)
    cart_ref = torch.bmm(batch.x.unsqueeze(1), batch.lattice[batch.batch]).squeeze(1)
    frac = torch.linalg.solve(
        batch.lattice[batch.batch].transpose(1, 2), cart - cart_ref
    )
    assert torch.allclose(frac, frac.round(), atol=1e-3)

    assert_valid(sheared)
    assert_same_crystals(batch, sheared)
    assert_same_hashes(batch, sheared)


def test_change_lattice_basis_not_unimodular():
    batch = make_batch()
    basis = torch.eye(3, dtype=torch.int64).repeat(3, 1, 1)
    basis[1, 0, 0] = 2

    with pytest.raises(ValueError):
        change_lattice_basis(batch, basis)


def test_make_supercell():
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 1], [1, 2, 3]])
    num_cells = repeat.prod(dim=1)
    supercell = make_supercell(batch, repeat)

    assert supercell.num_atoms.tolist() == (batch.num_atoms * num_cells).tolist()
    assert torch.allclose(supercell.lattice, batch.lattice * repeat.unsqueeze(-1))
    # extensive and intensive properties
    assert supercell.total_energy[[0, 2]].tolist() == [-16.0, -24.0]
    assert supercell.total_energy[1].isnan()
    assert (supercell.band_gap == batch.band_gap).all()

    # structure 1 is unchanged
    start, end = supercell.ptr[1], supercell.ptr[2]
    assert (supercell.x[start:end] == batch.x[batch.ptr[1] : batch.ptr[2]]).all()

    # the atoms of zincblende are repeated cell by cell
    assert (supercell.z[end:].view(6, 8) == batch.z[batch.ptr[2] :]).all()

    assert_valid(supercell)
    assert_same_crystals(batch, supercell, num_cells)
    assert_same_colors(batch, supercell, num_cells)

    # an atom of the supercell has the color of the atom it is a copy of, the
    # hash of the structure changes with its number of atoms
    local = torch.arange(len(supercell.z)) - supercell.ptr[supercell.batch]
    origin = local % batch.num_atoms[supercell.batch] + batch.ptr[supercell.batch]
    assert (colors(supercell) == colors(batch)[origin]).all()
    assert ((hashes(supercell) == hashes(batch)).all(dim=1) == (num_cells == 1)).all()


def test_make_supercell_invalid_repeat():
    with pytest.raises(ValueError):
        make_supercell(Structures.diamond(), torch.tensor([[1, 0, 1]]))


def test_random_repeat():
    num_atoms = torch.tensor([8] * 64)
    repeat = random_repeat(num_atoms, max_repeat=3, generator=generator())

    assert repeat.shape == (64, 3)
    assert repeat.min() == 1 and repeat.max() == 3

    capped = random_repeat(num_atoms, max_repeat=3, max_atoms=16, generator=generator())
    small = repeat.prod(dim=1) <= 2
    assert (capped[small] == repeat[small]).all()
    assert (capped[~small] == 1).all()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_transforms(transform):
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    assert_valid(augmented)
    assert_same_crystals(batch, augmented)
    assert_same_hashes(batch, augmented)
    assert (augmented.band_gap == batch.band_gap).all()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_hash_is_invariant(transform):
    batch = make_batch()
    expected = hashes(batch)

    # the hash has no tolerance: several draws, for the bonds close to the cutoff
    for seed in range(10):
        augmented = transform(generator=generator(seed))(batch)
        assert (hashes(augmented) == expected).all(), f"seed {seed}"


def test_supercell_transform():
    batch = make_batch()
    augmented = Supercell(max_repeat=2, generator=generator(1))(batch)
    num_cells = augmented.num_atoms // batch.num_atoms

    assert (num_cells > 1).any()
    assert_valid(augmented)
    assert_same_crystals(batch, augmented, num_cells)
    assert_same_colors(batch, augmented, num_cells)


def test_compose():
    batch = make_batch()
    gen = generator(1)
    augment = Compose(
        [
            Supercell(max_repeat=2, max_atoms=40, generator=gen),
            LatticeBasisChange(generator=gen),
            LatticeRotation(generator=gen),
            AtomTranslation(generator=gen),
            AtomPermutation(generator=gen),
        ]
    )
    augmented = augment(batch)
    num_cells = augmented.num_atoms // batch.num_atoms

    assert (augmented.num_atoms <= 40).all()
    assert_valid(augmented)
    assert_same_crystals(batch, augmented, num_cells)
    assert_same_colors(batch, augmented, num_cells)


def test_transforms_are_reproducible():
    batch = make_batch()
    first = LatticeRotation(generator=generator(3))(batch)
    second = LatticeRotation(generator=generator(3))(batch)

    assert (first.lattice == second.lattice).all()
