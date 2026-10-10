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
from torchms.models.layers.basis import spherical_harmonics
from torchms.models.layers import (
    AngleBasis,
    EquivariantNetwork,
    LineGraphNetwork,
    equivariant_network,
    gaunt_coefficients,
    line_graph_network,
)
from torchms.models.layers.basis import DEGREES

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


def make_line_graph_network(order: int, chiral: bool = False) -> LineGraphNetwork:
    torch.manual_seed(order)
    return LineGraphNetwork(
        line_graph_network.Config(
            order=order, radius=RADIUS, hidden=16, num_layers=2, chiral=chiral
        )
    ).eval()


def run(model, structures: Structures) -> torch.Tensor:
    """Energy of structures without graph: a network which reads a graph is
    given the one of its transform."""
    if hasattr(model, "transform"):
        structures = model.transform()(structures)

    return model(structures)


def make_equivariant_network(correlation: int) -> EquivariantNetwork:
    torch.manual_seed(correlation)
    return EquivariantNetwork(
        equivariant_network.Config(
            correlation=correlation, radius=RADIUS[0], channels=16, num_layers=2
        )
    ).eval()


MODELS = [
    pytest.param(lambda: make_line_graph_network(1), id="bonds"),
    pytest.param(lambda: make_line_graph_network(2), id="angles"),
    pytest.param(lambda: make_line_graph_network(3), id="dihedrals"),
    pytest.param(lambda: make_equivariant_network(1), id="products-1"),
    pytest.param(lambda: make_equivariant_network(2), id="products-2"),
    pytest.param(lambda: make_equivariant_network(3), id="products-3"),
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
        LineGraphNetwork(line_graph_network.Config(order=4))
    with pytest.raises(ValueError, match="radius"):
        LineGraphNetwork(line_graph_network.Config(order=3, radius=(5.0, 3.0)))
    with pytest.raises(ValueError, match="radius"):
        LineGraphNetwork(line_graph_network.Config(order=2, radius=(3.0, 5.0)))
    with pytest.raises(ValueError, match="mirror"):
        LineGraphNetwork(line_graph_network.Config(order=2, chiral=True))
    with pytest.raises(ValueError, match="correlation"):
        EquivariantNetwork(equivariant_network.Config(correlation=4))


def test_line_graph_network_orders():
    batch = make_batch()

    for order in (1, 2, 3):
        model = make_line_graph_network(order)
        structures = model.transform()(batch)
        geometry = model.geometry(structures)

        # the transform attaches what the order of the network reads, no more
        attached = [structures.graph, structures.triplets, structures.quadruplets]
        assert [item is not None for item in attached] == [n < order for n in range(3)]

        values = [geometry.distance, geometry.angle, geometry.dihedral]
        cutoffs = [
            geometry.bond_cutoff,
            geometry.triplet_cutoff,
            geometry.quadruplet_cutoff,
        ]
        assert [item is not None for item in values] == [n < order for n in range(3)]
        assert [item is not None for item in cutoffs] == [n < order for n in range(3)]

        index = [torch.arange(len(geometry.distance)).unsqueeze(0)]
        index += [item.index for item in attached[1:order]]

        for n in range(order):
            assert index[n].shape[1] > 0
            assert values[n].shape == (index[n].shape[1],)
            assert cutoffs[n].shape == (index[n].shape[1], 1)
            assert (cutoffs[n] > -1e-6).all() and (cutoffs[n] <= 1).all()
            # the bonds of each order are within its radius
            assert (geometry.distance[index[n]] <= RADIUS[n]).all()


def test_line_graph_network_does_not_build_graphs():
    from torchms.graph.transforms import AddRadiusGraph

    batch = make_batch()
    model = make_line_graph_network(3)

    with pytest.raises(ValueError, match="graph, triplets, quadruplets"):
        model(batch)

    # a graph alone is not completed with the tuples of edges of the network
    with pytest.raises(ValueError, match="triplets, quadruplets.*transform"):
        model(AddRadiusGraph(RADIUS[0], self_images=True)(batch))

    # a network of a lower order reads less
    structures = make_line_graph_network(2).transform()(batch)
    assert structures.quadruplets is None
    assert make_line_graph_network(2)(structures).shape == (2,)


def test_line_graph_network_cutoffs_of_tuples():
    """The cutoff of a triplet or a quadruplet is the product of its bonds'."""
    model = make_line_graph_network(3)
    structures = model.transform()(make_batch())
    geometry = model.geometry(structures)

    for tuples, cutoff, radius in (
        (structures.triplets, geometry.triplet_cutoff, RADIUS[1]),
        (structures.quadruplets, geometry.quadruplet_cutoff, RADIUS[2]),
    ):
        envelope = model.envelope(geometry.distance, radius)
        assert torch.allclose(cutoff, envelope[tuples.index].prod(dim=0))


def test_equivariant_network_does_not_build_graphs():
    batch = make_batch()
    model = make_equivariant_network(3)

    with pytest.raises(ValueError, match="graph.*transform"):
        model(batch)

    structures = model.transform()(batch)
    assert structures.triplets is None and structures.line_graph is None

    graph, radial, harmonics = model.edges(structures)
    assert graph is structures.graph
    assert (graph.edges_distances() <= RADIUS[0]).all()
    assert radial.shape == (int(graph.num_edges.sum()), 8)
    assert harmonics.shape == (int(graph.num_edges.sum()), 16)


@pytest.mark.parametrize("make_model", MODELS)
def test_output(make_model):
    energy = run(make_model(), make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float32
    assert energy.isfinite().all()


@pytest.mark.parametrize("make_model", MODELS)
def test_batch_equals_single(make_model):
    model = make_model()
    with torch.no_grad():
        alone = torch.cat([run(model, structure) for structure in make_structures()])

        assert torch.allclose(run(model, make_batch()), alone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("make_model", MODELS)
@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_invariances(make_model, transform):
    model = make_model()
    batch = make_batch()

    with torch.no_grad():
        expected = run(model, batch)
        for seed in range(2):
            augmented = transform(generator=generator(seed))(batch)
            assert torch.allclose(run(model, augmented), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_energy_of_a_supercell(make_model):
    model = make_model()
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 2]])

    with torch.no_grad():
        energy = run(model, make_supercell(batch, repeat))
        expected = run(model, batch) * repeat.prod(dim=1)

    assert torch.allclose(energy, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("make_model", MODELS)
def test_depends_on_the_structure(make_model):
    model = make_model()
    batch = make_batch()
    strained = Structures.collate(make_structures())
    strained.lattice = batch.lattice * torch.tensor([1.05, 1.0, 1.0]).view(1, 3, 1)

    with torch.no_grad():
        assert not torch.allclose(
            run(model, strained), run(model, batch), rtol=1e-3, atol=1e-3
        )


@pytest.mark.parametrize("make_model", MODELS)
def test_gradients_reach_every_parameter(make_model):
    model = make_model().train()
    run(model, make_batch()).sum().backward()

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
        yield "bonds", LineGraphNetwork(
            line_graph_network.Config(1, (radius,), hidden=16, num_layers=1)
        ), True
        yield "angles", LineGraphNetwork(
            line_graph_network.Config(2, (radius, radius), hidden=16, num_layers=1)
        ), False
        for correlation, same in [(1, True), (2, False), (3, False)]:
            yield f"products-{correlation}", EquivariantNetwork(
                equivariant_network.Config(
                    correlation, radius, channels=16, num_layers=1
                )
            ), same

    with torch.no_grad():
        for name, model, same in models():
            difference = (run(model.eval(), corner) - run(model, line)).abs().item()
            if same:
                assert difference < 1e-6, name
            else:
                assert difference > 1e-4, name


def test_chiral_model_sees_mirror_images():
    batch = make_batch()
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0])).repeat(2, 1, 1)
    proper = random_orthogonal(2, torch.device("cpu"), generator(), proper=True)

    achiral = make_line_graph_network(3)
    chiral = make_line_graph_network(3, chiral=True)

    with torch.no_grad():
        for model in (achiral, chiral):
            rotated = run(model, rotate_lattice(batch, proper))
            assert torch.allclose(rotated, run(model, batch), rtol=1e-3, atol=1e-3)

        mirrored = rotate_lattice(batch, mirror)
        assert torch.allclose(
            run(achiral, mirrored), run(achiral, batch), rtol=1e-3, atol=1e-3
        )

        # zincblende is its own mirror image, the triclinic cell is not
        difference = (run(chiral, mirrored) - run(chiral, batch)).abs()
        assert difference[0] < 1e-4
        assert difference[1] > 1e-3


def test_networks_hyperparameters():
    """The number of elements and the cutoff are arguments of the networks."""
    batch = make_batch()

    for make in (
        lambda **kwargs: LineGraphNetwork(
            line_graph_network.Config(order=2, radius=RADIUS, hidden=16, **kwargs)
        ),
        lambda **kwargs: EquivariantNetwork(
            equivariant_network.Config(radius=RADIUS[0], channels=16, **kwargs)
        ),
    ):
        torch.manual_seed(0)
        default = make().eval()
        assert default.atom_embedding.num_embeddings == 119
        assert default.cutoff_power == 4

        torch.manual_seed(0)
        other = make(num_elements=60, cutoff_power=6).eval()
        assert other.atom_embedding.num_embeddings == 61
        assert other.cutoff_power == 6

        # the same weights with another cutoff give another energy
        other = make(cutoff_power=6).eval()
        other.load_state_dict(default.state_dict())
        with torch.no_grad():
            assert not torch.allclose(run(other, batch), run(default, batch))
