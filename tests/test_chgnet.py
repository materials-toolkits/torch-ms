"""Tests for the rewrite of the energy prediction of CHGNet."""

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
from torchms.models.chgnet import CHGNet, InvalidWeights
from torchms.models.chgnet.model import BOND_RADIUS, MODEL_ARGS, RADIUS
from torchms.graph.line_graph import reverse_edges


def predict(model, structures: Structures) -> torch.Tensor:
    """Prediction of a model for structures without graph."""
    return model(model.transform()(structures))


DATA = Path(__file__).parent / "data"
NAME = "chgnet-0.3.0"

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
def model() -> CHGNet:
    """Model with random weights."""
    torch.manual_seed(0)
    return CHGNet().eval().requires_grad_(False)


def per_atom(model: CHGNet, structures: Structures) -> torch.Tensor:
    return predict(model, structures) / structures.num_atoms


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["chgnet"]) == {NAME}
    (checkpoint,) = weights.CHECKPOINTS["chgnet"][NAME]
    assert checkpoint.url.startswith("https://")
    assert len(checkpoint.sha256) == 64
    (path,) = weights.checkpoint_paths("chgnet", NAME, tmp_path)
    assert path.parent == tmp_path / "chgnet"

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("chgnet", "chgnet-9")


def test_model_output(model):
    batch = make_batch()
    energy = predict(model, batch)

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert energy.isfinite().all()


def test_model_graphs(model):
    batch = make_batch()
    graph, undirected, bond_length, triplets, cosines = model.graphs(
        model.transform()(batch)
    )
    edge_bond = undirected.edge_bond
    triplet_index = triplets.index
    distance = graph.edges_distances()

    assert (distance <= RADIUS).all()

    # an edge and its reverse are one bond
    assert (edge_bond[reverse_edges(graph)] == edge_bond).all()
    assert len(bond_length) * 2 == graph.edge_index.shape[1]
    assert (torch.bincount(edge_bond) == 2).all()
    assert torch.allclose(bond_length[edge_bond], distance, atol=1e-5)

    # angles between two different short edges around the same atom
    assert triplet_index.shape[1] > 0
    assert (distance[triplet_index] < BOND_RADIUS).all()
    assert (triplet_index[0] != triplet_index[1]).all()
    assert (
        graph.edge_index[0, triplet_index[0]] == graph.edge_index[0, triplet_index[1]]
    ).all()
    assert cosines.shape == (triplet_index.shape[1],)


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


def test_model_without_angles(model):
    # no bond shorter than 3 Å: the bonds are not updated
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 4.0, x=torch.zeros(1, 3), z=[29]
    )
    _, _, _, triplets, _ = model.graphs(model.transform()(structures))

    assert triplets.index.shape[1] == 0
    assert predict(model, structures).isfinite().all()


def test_load_pretrained_checks_the_parameters(model):
    state_dict = dict(model.state_dict())
    state_dict["site_wise.weight"] = torch.zeros(1, 64)

    other = CHGNet()
    other.load_pretrained(state_dict, MODEL_ARGS)
    other.eval()
    with torch.no_grad():
        assert torch.allclose(
            predict(other, make_batch()), predict(model, make_batch())
        )

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(state_dict, {**MODEL_ARGS, "atom_graph_cutoff": 5})

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "unknown.weight": torch.zeros(1)})

    del state_dict["bond_embedding.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


# the pretrained model against the real chgnet


def reference_structures() -> tuple[Structures, dict[str, np.ndarray]]:
    data = dict(np.load(DATA / "orb_reference.npz"))
    structures = Structures(
        lattice=torch.from_numpy(data["lattice"]),
        x=torch.from_numpy(data["x"]),
        z=torch.from_numpy(data["z"]),
        num_atoms=torch.from_numpy(data["num_atoms"]),
    )
    return structures, dict(np.load(DATA / "chgnet_reference.npz"))


@pytest.fixture(scope="module")
def pretrained() -> CHGNet:
    if not weights.checkpoint_paths("chgnet", NAME)[0].exists():
        pytest.skip(f"weights of {NAME} are not in the cache")

    return CHGNet.from_pretrained(NAME, device="cpu")


@pytest.mark.timeout(120)
def test_pretrained_graphs_match_chgnet(pretrained):
    """Graphs of chgnet 0.4.2, computed once outside of the project."""
    structures, data = reference_structures()
    graph, undirected, _, triplets, cosines = pretrained.graphs(
        pretrained.transform()(structures)
    )
    edge_bond = undirected.edge_bond
    triplet_index = triplets.index
    n_struct = len(structures.num_atoms)

    assert graph.num_edges.tolist() == data["num_edges"].tolist()

    num_bonds = torch.bincount(graph.edge_batch, minlength=n_struct) // 2
    assert int(edge_bond.max()) + 1 == int(data["num_bonds"].sum())
    assert num_bonds.tolist() == data["num_bonds"].tolist()

    triplet_batch = graph.edge_batch[triplet_index[0]]
    num_angles = torch.bincount(triplet_batch, minlength=n_struct)
    assert num_angles.tolist() == data["num_angles"].tolist()

    cosine_sum = torch.zeros(n_struct, dtype=torch.float64).index_add_(
        0, triplet_batch, cosines.double()
    )
    assert np.allclose(cosine_sum.numpy(), data["cosine_sum"], atol=1e-2)


@pytest.mark.timeout(120)
def test_pretrained_matches_chgnet(pretrained):
    """Energies of chgnet 0.4.2, computed once outside of the project."""
    structures, data = reference_structures()

    energy = predict(pretrained, structures).numpy()
    error = np.abs(energy - data["energy"]) / data["num_atoms"]

    assert error.max() < 1e-4


@pytest.mark.timeout(120)
def test_pretrained_state_dict_is_fully_used(pretrained):
    checkpoint = torch.load(
        weights.checkpoint_paths("chgnet", NAME)[0],
        map_location="cpu",
        weights_only=True,
    )
    unused = set(checkpoint["model"]["state_dict"]) - set(pretrained.state_dict())

    assert unused == {"site_wise.weight", "site_wise.bias"}


def test_unknown_element(model):
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 4.0, x=torch.zeros(1, 3), z=[95]
    )

    with pytest.raises(ValueError, match=r"elements \[95\] are not supported"):
        predict(model, structures)
