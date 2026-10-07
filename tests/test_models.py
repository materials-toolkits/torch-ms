"""Tests for the trainable models of the energy."""

import pytest
import torch

from torchms.data import Structures, parse_cif
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
    random_orthogonal,
    rotate_lattice,
)
from torchms.estimator.orb.model import spherical_harmonics
from torchms.models import (
    AngleBasis,
    EquivariantNetwork,
    LineGraphNetwork,
    gaunt_coefficients,
)
from torchms.models.basis import DEGREES

pytestmark = pytest.mark.timeout(30)

RADIUS = (4.0, 3.0, 2.6)


def make_structures() -> list[Structures]:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.zincblende(), triclinic]


def make_batch() -> Structures:
    return Structures.collate(make_structures())


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def line_graph_network(order: int, chiral: bool = False) -> LineGraphNetwork:
    torch.manual_seed(order)
    return LineGraphNetwork(
        order=order, radius=RADIUS, hidden=16, num_layers=2, chiral=chiral
    ).eval()


def equivariant_network(correlation: int) -> EquivariantNetwork:
    torch.manual_seed(correlation)
    return EquivariantNetwork(
        correlation=correlation, radius=RADIUS[0], channels=16, num_layers=2
    ).eval()


MODELS = [
    pytest.param(lambda: line_graph_network(1), id="bonds"),
    pytest.param(lambda: line_graph_network(2), id="angles"),
    pytest.param(lambda: line_graph_network(3), id="dihedrals"),
    pytest.param(lambda: equivariant_network(1), id="products-1"),
    pytest.param(lambda: equivariant_network(2), id="products-2"),
    pytest.param(lambda: equivariant_network(3), id="products-3"),
]


# bases


def test_angle_basis():
    angle = torch.linspace(-3.0, 3.0, 13)

    cosines = AngleBasis(4)(angle)
    assert cosines.shape == (13, 4)
    assert torch.allclose(cosines, AngleBasis(4)(-angle), atol=1e-6)

    both = AngleBasis(4, sine=True)(angle)
    assert both.shape == (13, 7)
    assert torch.allclose(both[:, :4], cosines)
    assert torch.allclose(
        both[:, 4:], -AngleBasis(4, sine=True)(-angle)[:, 4:], atol=1e-6
    )


def test_gaunt_coefficients_selection_rules():
    gaunt = gaunt_coefficients().double()

    # symmetric in its three indices
    for permutation in [(1, 0, 2), (0, 2, 1), (2, 1, 0)]:
        assert torch.allclose(gaunt, gaunt.permute(permutation), atol=1e-7)

    # the product with the constant harmonic is the identity
    assert torch.allclose(gaunt[0], torch.eye(16, dtype=torch.float64), atol=1e-6)

    # zero unless the degrees form a triangle and have an even sum
    degree = torch.tensor(DEGREES)
    l1, l2, l3 = degree[:, None, None], degree[None, :, None], degree[None, None, :]
    allowed = ((l1 + l2 + l3) % 2 == 0) & (l3 <= l1 + l2) & (l3 >= (l1 - l2).abs())
    assert gaunt[~allowed].abs().max() < 1e-6
    assert gaunt[allowed].abs().max() > 0.1


def test_gaunt_coefficients_are_equivariant():
    gaunt = gaunt_coefficients().double()
    directions = torch.randn(64, 3, dtype=torch.float64, generator=generator())
    # orthogonal in double precision
    rotation = torch.linalg.qr(
        random_orthogonal(1, torch.device("cpu"), generator())[0].double()
    ).Q

    # a function on the sphere as coefficients of harmonics, and its rotation:
    # the harmonics of rotated directions are a linear map of the harmonics
    harmonics = spherical_harmonics(directions)
    rotated = spherical_harmonics(directions @ rotation.T)
    wigner = torch.linalg.lstsq(
        harmonics, rotated
    ).solution  # rotated = harmonics @ wigner
    assert torch.allclose(harmonics @ wigner, rotated, atol=1e-8)

    a = torch.randn(16, dtype=torch.float64, generator=generator(1))
    b = torch.randn(16, dtype=torch.float64, generator=generator(2))

    product = torch.einsum("ijk,i,j->k", gaunt, a, b)
    product_of_rotated = torch.einsum("ijk,i,j->k", gaunt, a @ wigner, b @ wigner)

    assert torch.allclose(product_of_rotated, product @ wigner, atol=1e-8)


# models


def test_arguments():
    with pytest.raises(ValueError, match="order"):
        LineGraphNetwork(order=4)
    with pytest.raises(ValueError, match="radius"):
        LineGraphNetwork(order=3, radius=(5.0, 3.0))
    with pytest.raises(ValueError, match="radius"):
        LineGraphNetwork(order=2, radius=(3.0, 5.0))
    with pytest.raises(ValueError, match="mirror"):
        LineGraphNetwork(order=2, chiral=True)
    with pytest.raises(ValueError, match="correlation"):
        EquivariantNetwork(correlation=4)


def test_line_graph_network_orders():
    batch = make_batch()

    for order in (1, 2, 3):
        index, geometry, weights = line_graph_network(order).build_graph(batch)
        names = ["bonds", "triplets", "quadruplets"][:order]

        assert list(index) == list(geometry) == list(weights) == names
        for name, radius in zip(names, RADIUS):
            assert index[name].shape[1] > 0
            assert geometry[name].shape == (index[name].shape[1],)
            assert weights[name].shape == (index[name].shape[1], 1)
            assert (weights[name] > -1e-6).all() and (weights[name] <= 1).all()

            bonds = (
                index[name] if name != "bonds" else torch.arange(len(geometry["bonds"]))
            )
            assert (geometry["bonds"][bonds] <= radius).all()


@pytest.mark.parametrize("make_model", MODELS)
def test_output(make_model):
    energy = make_model()(make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float32
    assert energy.isfinite().all()


@pytest.mark.parametrize("make_model", MODELS)
def test_batch_equals_single(make_model):
    model = make_model()
    with torch.no_grad():
        alone = torch.cat([model(structure) for structure in make_structures()])

        assert torch.allclose(model(make_batch()), alone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("make_model", MODELS)
@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_invariances(make_model, transform):
    model = make_model()
    batch = make_batch()

    with torch.no_grad():
        expected = model(batch)
        for seed in range(2):
            augmented = transform(generator=generator(seed))(batch)
            assert torch.allclose(model(augmented), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_energy_of_a_supercell(make_model):
    model = make_model()
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 2]])

    with torch.no_grad():
        energy = model(make_supercell(batch, repeat))
        expected = model(batch) * repeat.prod(dim=1)

    assert torch.allclose(energy, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_depends_on_the_structure(make_model):
    model = make_model()
    batch = make_batch()
    strained = Structures.collate(make_structures())
    strained.lattice = batch.lattice * torch.tensor([1.05, 1.0, 1.0]).view(1, 3, 1)

    with torch.no_grad():
        assert not torch.allclose(model(strained), model(batch), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_gradients_reach_every_parameter(make_model):
    model = make_model().train()
    model(make_batch()).sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_higher_orders_see_more():
    # an atom with two neighbours at 2.4 Å, at 90° or at 180°: the neighbours
    # are too far from each other to be bonded, the distances are the same
    def three_atoms(third: list[float]) -> Structures:
        return Structures(
            lattice=torch.eye(3).unsqueeze(0) * 12.0,
            x=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], third],
            z=[6, 6, 6],
        )

    corner, line = three_atoms([0.0, 0.2, 0.0]), three_atoms([-0.2, 0.0, 0.0])

    # the neighbours are close to the radius, where every term is small
    radius = 3.3

    def models():
        torch.manual_seed(2)
        yield "bonds", LineGraphNetwork(1, (radius,), hidden=16, num_layers=1), True
        yield "angles", LineGraphNetwork(
            2, (radius, radius), hidden=16, num_layers=1
        ), False
        for correlation, same in [(1, True), (2, False), (3, False)]:
            yield f"products-{correlation}", EquivariantNetwork(
                correlation, radius, channels=16, num_layers=1
            ), same

    with torch.no_grad():
        for name, model, same in models():
            difference = (model.eval()(corner) - model(line)).abs().item()
            if same:
                assert difference < 1e-6, name
            else:
                assert difference > 1e-4, name


def test_chiral_model_sees_mirror_images():
    batch = make_batch()
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0])).repeat(2, 1, 1)
    proper = random_orthogonal(2, torch.device("cpu"), generator(), proper=True)

    achiral = line_graph_network(3)
    chiral = line_graph_network(3, chiral=True)

    with torch.no_grad():
        for model in (achiral, chiral):
            rotated = model(rotate_lattice(batch, proper))
            assert torch.allclose(rotated, model(batch), rtol=1e-3, atol=1e-3)

        mirrored = rotate_lattice(batch, mirror)
        assert torch.allclose(achiral(mirrored), achiral(batch), rtol=1e-3, atol=1e-3)

        # zincblende is its own mirror image, the triclinic cell is not
        difference = (chiral(mirrored) - chiral(batch)).abs()
        assert difference[0] < 1e-4
        assert difference[1] > 1e-3
