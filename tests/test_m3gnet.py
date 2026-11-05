"""Tests for the rewrite of the energy prediction of M3GNet."""

from pathlib import Path

import numpy as np
import pytest
import torch

from torchms.data import Structures, parse_cif
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
)
from torchms.models.estimator import weights
from torchms.models.m3gnet import Config, InvalidWeights, M3GNet


def predict(model, structures: Structures) -> torch.Tensor:
    """Prediction of a model for structures without graph."""
    return model(model.transform()(structures))


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
def model() -> M3GNet:
    """Model with random weights."""
    torch.manual_seed(0)
    return M3GNet().eval().requires_grad_(False)


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["m3gnet"]) == {NAME}
    for checkpoint in weights.CHECKPOINTS["m3gnet"][NAME]:
        assert checkpoint.url.startswith("https://")
        assert len(checkpoint.sha256) == 64

    state, args = weights.checkpoint_paths("m3gnet", NAME, tmp_path)
    assert state == tmp_path / "m3gnet" / NAME / "state.pt"
    assert args == tmp_path / "m3gnet" / NAME / "model.pt"

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("m3gnet", "m3gnet-9")


# bases


# the model with random weights


def test_model_output(model):
    energy = predict(model, make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert energy.isfinite().all()


def test_model_graphs(model):
    config = Config.load()
    batch = make_batch()
    graph, distance, angles, cosine = model.graphs(model.transform()(batch))
    triplet_index = angles.index

    assert (distance <= config.graph.radius).all()
    assert torch.allclose(distance, graph.edges_distances())

    assert triplet_index.shape[1] > 0
    assert (distance[triplet_index] <= config.graph.three_body_radius).all()
    assert (triplet_index[0] != triplet_index[1]).all()
    assert (
        graph.edge_index[0, triplet_index[0]] == graph.edge_index[0, triplet_index[1]]
    ).all()
    assert cosine.abs().max() < 1.0


def test_model_batch_equals_single(model):
    alone = torch.cat([predict(model, structure) for structure in make_structures()])

    assert torch.allclose(predict(model, make_batch()), alone, rtol=1e-5)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_model_invariances(model, transform):
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    assert torch.allclose(predict(model, augmented), predict(model, batch), rtol=1e-4)


def test_model_energy_of_a_supercell(model):
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 1]])
    supercell = make_supercell(batch, repeat)

    assert torch.allclose(
        predict(model, supercell), predict(model, batch) * repeat.prod(dim=1), rtol=1e-4
    )


def test_model_unknown_element(model):
    # polonium is not an element of the model
    config = Config.load()
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[84]
    )

    with pytest.raises(ValueError, match="not supported"):
        predict(model, structures)

    assert (
        len(config.model.atomic_numbers()) == 89
        and 84 not in config.model.atomic_numbers()
        and 94 in config.model.atomic_numbers()
    )


def test_load_pretrained_checks_the_parameters(model):
    config = Config.load()
    state_dict = dict(model.state_dict())

    other = M3GNet()
    other.load_pretrained(state_dict, config.upstream.model_args)
    other.eval()
    with torch.no_grad():
        assert torch.allclose(
            predict(other, make_batch()), predict(model, make_batch())
        )

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(
            state_dict, {**config.upstream.model_args, "threebody_cutoff": 3.0}
        )

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(
            state_dict, {**config.upstream.model_args, "element_types": ("H", "C")}
        )

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


@pytest.fixture(scope="module")
def pretrained() -> M3GNet:
    if not all(path.exists() for path in weights.checkpoint_paths("m3gnet", NAME)):
        pytest.skip(f"weights of {NAME} are not in the cache")

    return M3GNet.from_pretrained(NAME, device="cpu")


@pytest.mark.timeout(120)
def test_pretrained_graphs_match_matgl(pretrained):
    """Graphs of matgl 4.1.0, computed once outside of the project."""
    structures, data = reference_structures()
    graph, _, angles, cosine = pretrained.graphs(pretrained.transform()(structures))
    triplet_index = angles.index
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

    energy = predict(pretrained, structures).numpy()
    error = np.abs(energy - data["energy"]) / structures.num_atoms.numpy()

    assert error.max() < 1e-4
