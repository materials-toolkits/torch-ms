"""Tests for the rewrite of the energy prediction of M3GNet."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import spherical_jn

from torchms.data import Structures, parse_cif
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
)
from torchms.estimator.m3gnet import InvalidWeights, M3gnetEnergyModel, weights
from torchms.estimator.m3gnet.model import (
    ELEMENTS,
    MAX_L,
    MAX_N,
    MODEL_ARGS,
    RADIUS,
    THREE_BODY_RADIUS,
    SphericalBesselFunction,
    legendre,
    spherical_bessel,
    spherical_bessel_roots,
)

DATA = Path(__file__).parent / "data"
NAME = "m3gnet-matpes-pbe-2025.2"

# thousands of angles per structure: slower than the default timeout
pytestmark = pytest.mark.timeout(30)


def make_structures() -> list[Structures]:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.zincblende(), triclinic]


def make_batch() -> Structures:
    return Structures.collate(make_structures())


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def model() -> M3gnetEnergyModel:
    """Model with random weights."""
    torch.manual_seed(0)
    return M3gnetEnergyModel().eval().requires_grad_(False)


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS) == {NAME}
    for checkpoint in weights.CHECKPOINTS[NAME]:
        assert checkpoint.url.startswith("https://")
        assert len(checkpoint.sha256) == 64

    state, args = weights.checkpoint_path(NAME, tmp_path)
    assert state == tmp_path / "m3gnet" / NAME / "state.pt"
    assert args == tmp_path / "m3gnet" / NAME / "model.pt"

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_path("m3gnet-9")


# bases


def test_spherical_bessel_roots():
    roots = spherical_bessel_roots(4, 5)

    assert roots.shape == (4, 5)
    for degree in range(4):
        assert np.allclose(spherical_jn(degree, roots[degree]), 0.0, atol=1e-10)
        assert (np.diff(roots[degree]) > 0).all()
    # the first roots, not later ones
    assert np.allclose(roots[:, 0], [math.pi, 4.493409, 5.763459, 6.987932], atol=1e-5)


def test_spherical_bessel():
    x = torch.linspace(0.3, 15.0, 50, dtype=torch.float64)
    functions = spherical_bessel(4, x)

    assert len(functions) == 5
    for degree, values in enumerate(functions):
        assert np.allclose(values.numpy(), spherical_jn(degree, x.numpy()), atol=1e-9)


def test_legendre():
    cosine = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    harmonics = legendre(3, cosine)

    assert harmonics.shape == (9, 3)
    expected = torch.stack(
        [
            torch.full_like(cosine, math.sqrt(1 / (4 * math.pi))),
            math.sqrt(3 / (4 * math.pi)) * cosine,
            math.sqrt(5 / (16 * math.pi)) * (3 * cosine**2 - 1),
        ],
        dim=-1,
    )
    assert torch.allclose(harmonics, expected)


def test_radial_basis_is_orthonormal():
    basis = SphericalBesselFunction().double()
    distance = torch.linspace(1e-4, RADIUS, 20001, dtype=torch.float64)
    values = basis(distance)

    assert values.shape == (len(distance), MAX_L * MAX_N)

    # functions of the same degree are orthonormal for the measure r^2 dr
    weight = distance**2 * (distance[1] - distance[0])
    for degree in range(MAX_L):
        block = values[:, degree * MAX_N : (degree + 1) * MAX_N]
        gram = block.T @ (block * weight.unsqueeze(-1))
        assert torch.allclose(gram, torch.eye(MAX_N, dtype=torch.float64), atol=1e-3)


# the model with random weights


def test_model_output(model):
    energy = model(make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert energy.isfinite().all()


def test_model_graphs(model):
    batch = make_batch()
    graph, distance, triplet_index, cosine = model.graphs(batch)

    assert (distance <= RADIUS).all()
    assert torch.allclose(distance, graph.edges_distances())

    assert triplet_index.shape[1] > 0
    assert (distance[triplet_index] <= THREE_BODY_RADIUS).all()
    assert (triplet_index[0] != triplet_index[1]).all()
    assert (
        graph.edge_index[0, triplet_index[0]] == graph.edge_index[0, triplet_index[1]]
    ).all()
    assert cosine.abs().max() < 1.0


def test_model_batch_equals_single(model):
    alone = torch.cat([model(structure) for structure in make_structures()])

    assert torch.allclose(model(make_batch()), alone, rtol=1e-5)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_model_invariances(model, transform):
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    assert torch.allclose(model(augmented), model(batch), rtol=1e-4)


def test_model_energy_of_a_supercell(model):
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 1]])
    supercell = make_supercell(batch, repeat)

    assert torch.allclose(
        model(supercell), model(batch) * repeat.prod(dim=1), rtol=1e-4
    )


def test_model_unknown_element(model):
    # polonium is not an element of the model
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[84]
    )

    with pytest.raises(ValueError, match="not supported"):
        model(structures)

    assert len(ELEMENTS) == 89 and 84 not in ELEMENTS and 94 in ELEMENTS


def test_load_pretrained_checks_the_parameters(model):
    state_dict = dict(model.state_dict())

    other = M3gnetEnergyModel()
    other.load_pretrained(state_dict, MODEL_ARGS)
    other.eval()
    with torch.no_grad():
        assert torch.allclose(other(make_batch()), model(make_batch()))

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(state_dict, {**MODEL_ARGS, "threebody_cutoff": 3.0})

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(state_dict, {**MODEL_ARGS, "element_types": ("H", "C")})

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "unknown.weight": torch.zeros(1)})

    del state_dict["data_std"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


# the pretrained model against the real matgl


def reference_structures() -> tuple[Structures, dict[str, np.ndarray]]:
    data = dict(np.load(DATA / "orb_reference.npz"))
    structures = Structures(
        lattice=torch.from_numpy(data["lattice"]),
        x=torch.from_numpy(data["x"]),
        z=torch.from_numpy(data["z"]),
        num_atoms=torch.from_numpy(data["num_atoms"]),
    )
    return structures, dict(np.load(DATA / "m3gnet_reference.npz"))


def test_bessel_roots_match_matgl():
    _, data = reference_structures()

    assert np.allclose(spherical_bessel_roots(4, 4), data["bessel_roots"], atol=1e-5)


@pytest.fixture(scope="module")
def pretrained() -> M3gnetEnergyModel:
    if not all(path.exists() for path in weights.checkpoint_path(NAME)):
        pytest.skip(f"weights of {NAME} are not in the cache")

    return M3gnetEnergyModel.from_pretrained(NAME, device="cpu")


@pytest.mark.timeout(120)
def test_pretrained_graphs_match_matgl(pretrained):
    """Graphs of matgl 4.1.0, computed once outside of the project."""
    structures, data = reference_structures()
    graph, _, triplet_index, cosine = pretrained.graphs(structures)
    n_struct = len(structures.num_atoms)

    assert graph.num_edges.tolist() == data["num_edges"].tolist()

    triplet_batch = graph.edge_batch[triplet_index[0]]
    num_angles = torch.bincount(triplet_batch, minlength=n_struct)
    assert num_angles.tolist() == data["num_angles"].tolist()

    cosine_sum = torch.zeros(n_struct, dtype=torch.float64).index_add_(
        0, triplet_batch, cosine.double()
    )
    assert np.allclose(cosine_sum.numpy(), data["cosine_sum"], atol=2e-2)


@pytest.mark.timeout(120)
def test_pretrained_matches_matgl(pretrained):
    """Energies of matgl 4.1.0, computed once outside of the project."""
    structures, data = reference_structures()

    energy = pretrained(structures).numpy()
    error = np.abs(energy - data["energy"]) / structures.num_atoms.numpy()

    assert error.max() < 1e-4
