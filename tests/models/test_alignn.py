"""Tests for the rewrite of the prediction of ALIGNN."""

import io
import json
import zipfile

import numpy as np
import pytest
import torch

from helpers.models import predict
from helpers.references import load_reference
from helpers.references import reference_structures as orb_structures
from helpers.structures import generator
from helpers.structures import make_batch as make_batch_of
from helpers.structures import make_structures as make_structures_of
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
)
from torchms.models.estimator import weights
from torchms.models.alignn import ALIGNN, Config, InvalidWeights
from torchms.models.alignn.config import BasisConfig
from torchms.models.estimator.weights import Checkpoint, sha256
from torchms.models.alignn.model import rbf_expansion, element_features
from torchms.graph.graph import Graph
from torchms.graph.line_graph import reverse_edges
from torchms.graph.transforms import AddLineGraph, attach_graph

NAME = "alignn-formation-energy-radius"

# layers of 512 features: slower than the default timeout
pytestmark = pytest.mark.timeout(30)


def copper() -> Structures:
    """Face centered cubic copper: 12 nearest neighbours, then 6 farther."""
    return Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.6,
        x=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        z=[29, 29, 29, 29],
    )


def make_structures() -> list[Structures]:
    return make_structures_of(copper)


def make_batch() -> Structures:
    return make_batch_of(copper)


def with_graph(structures: Structures, graph: Graph) -> Structures:
    """Structures with a given graph of the atoms, and the line graph of the model."""
    config = Config.load()
    line_graph = AddLineGraph(radius=config.graph.three_body_radius, backtrack=True)
    return line_graph(attach_graph(structures, graph, False))


@pytest.fixture(scope="module")
def model() -> ALIGNN:
    """Model with random weights."""
    torch.manual_seed(0)
    return ALIGNN().eval().requires_grad_(False)


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["alignn"]) == {NAME}
    (checkpoint,) = weights.CHECKPOINTS["alignn"][NAME]
    assert checkpoint.url.startswith("https://")
    assert len(checkpoint.sha256) == 64
    assert weights.checkpoint_paths("alignn", NAME, tmp_path)[0] == (
        tmp_path / "alignn" / "formation_energy_peratom_radius.zip"
    )

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("alignn", "alignn-9")


def test_element_features():
    features = element_features(Config.load().element_features_path())

    assert features.shape == (119, 92)
    assert set(features.unique().tolist()) == {0.0, 1.0}
    assert (features[0] == 0).all()
    # every element has features, a few elements share the same ones
    assert (features[1:].sum(dim=1) > 0).all()
    assert features[1:101].unique(dim=0).shape[0] > 90
    assert (features[6] != features[8]).any()


def test_rbf_expansion():
    expansion = rbf_expansion(BasisConfig(start=0.0, stop=8.0, num_basis=80))
    values = expansion(torch.tensor([0.0, 4.0, 8.0]))

    assert values.shape == (3, 80)
    assert values[0, 0] == 1.0 and values[2, -1] == 1.0
    # gaussian exp(-(d - c)^2 / step) with a step of 8 / 79
    step = 8.0 / 79.0
    assert torch.isclose(values[0, 1], torch.exp(torch.tensor(-step)), atol=1e-6)


# the model with random weights


def test_model_output(model):
    prediction = predict(model, make_batch())

    assert prediction.shape == (2,)
    assert prediction.isfinite().all()


def test_model_graphs(model):
    config = Config.load()
    batch = make_batch()
    graph, distance, pairs, cosine = model.graphs(model.transform()(batch))
    pair_index = pairs.index

    assert (distance <= config.graph.radius).all()
    degree = torch.bincount(graph.edge_index[0], minlength=len(batch.z))
    assert (degree == config.graph.max_neighbors).all()

    # a bond is followed by the bonds starting from its end, itself reversed too
    first, second = pair_index
    assert pair_index.shape[1] > 0
    assert (distance[pair_index] <= config.graph.three_body_radius).all()
    assert (graph.edge_index[1, first] == graph.edge_index[0, second]).all()

    backward = reverse_edges(graph)[first] == second
    assert backward.any()
    assert torch.allclose(cosine[backward], torch.ones(int(backward.sum())), atol=1e-5)

    # in copper the angles between nearest neighbours are 60, 90, 120 and 180°
    in_copper = graph.edge_batch[first] == 0
    assert sorted(
        cosine[in_copper & ~backward].round(decimals=3).unique().tolist()
    ) == [
        -1.0,
        -0.5,
        0.0,
        0.5,
    ]


def test_model_batch_equals_single(model):
    alone = torch.cat([predict(model, structure) for structure in make_structures()])

    assert torch.allclose(predict(model, make_batch()), alone, atol=1e-5)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_model_invariances(model, transform):
    # structures where the 12 nearest neighbours are not chosen among
    # neighbours at the same distance
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    assert torch.allclose(predict(model, augmented), predict(model, batch), atol=1e-4)


def test_model_prediction_of_a_supercell(model):
    # the prediction is a mean over the atoms
    batch = make_batch()
    supercell = make_supercell(batch, torch.tensor([[2, 1, 1], [1, 1, 1]]))

    assert torch.allclose(predict(model, supercell), predict(model, batch), atol=1e-4)


def test_model_with_a_given_graph(model):
    batch = make_batch()
    graph, _, _, _ = model.graphs(model.transform()(batch))

    assert torch.equal(model(with_graph(batch, graph)), predict(model, batch))

    # a graph of the atoms without its line graph is not completed
    with pytest.raises(ValueError, match="needs structures with their line_graph"):
        model(attach_graph(batch, graph, False))


def state_and_config(model: ALIGNN) -> tuple[dict, dict]:
    config = Config.load()
    state_dict = dict(model.state_dict())
    state_dict["_species_table"] = torch.zeros(120, 92)
    return state_dict, {
        **config.upstream.graph_args,
        "model": dict(config.upstream.model_args),
    }


def test_load_pretrained_checks_the_parameters(model):
    config = Config.load()
    state_dict, upstream = state_and_config(model)

    other = ALIGNN()
    other.load_pretrained(state_dict, upstream)
    other.eval()
    with torch.no_grad():
        assert torch.allclose(
            predict(other, make_batch()), predict(model, make_batch())
        )

    with pytest.raises(InvalidWeights, match="another configuration"):
        other.load_pretrained(state_dict, {**upstream, "cutoff": 8.0})

    with pytest.raises(InvalidWeights, match="another configuration"):
        other.load_pretrained(
            state_dict,
            {
                **upstream,
                "model": {**config.upstream.model_args, "alignn_layers": 2},
            },
        )

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "unknown.weight": torch.zeros(1)})

    del state_dict["fc.bias"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


def test_from_pretrained_reads_the_archive(model, tmp_path, monkeypatch):
    state_dict, config = state_and_config(model)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)

    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as fp:
        fp.writestr("config.json", json.dumps(config))
        fp.writestr("best_model.pt", buffer.getvalue())

    checkpoint = Checkpoint(archive.as_uri(), sha256(archive), "model.zip")
    monkeypatch.setitem(weights.CHECKPOINTS["alignn"], "local", (checkpoint,))

    loaded = ALIGNN.from_pretrained("local", cache=tmp_path / "cache")
    assert torch.allclose(predict(loaded, make_batch()), predict(model, make_batch()))


# the pretrained model against the real alignn


def reference_structures() -> tuple[Structures, dict[str, np.ndarray], Graph]:
    """The structures, what alignn computed for them, and the graph it built."""
    structures = orb_structures()
    reference = load_reference("alignn_reference.npz").numpy()

    # graph of the atoms built by alignn
    num_edges = torch.from_numpy(reference["num_edges"])
    offset = structures.ptr[torch.arange(len(num_edges)).repeat_interleave(num_edges)]
    graph = Graph(
        structures=structures,
        edge_index=torch.stack(
            [
                torch.from_numpy(reference["sender"]).long() + offset,
                torch.from_numpy(reference["receiver"]).long() + offset,
            ]
        ),
        edge_shift=torch.from_numpy(reference["shift"]).long(),
        num_atoms=structures.num_atoms,
        num_edges=num_edges,
    )
    return structures, reference, graph


def edges_of(graph: Graph) -> list[set[tuple[int, ...]]]:
    """Edges of each structure as sets of (sender, receiver, shift)."""
    edges = torch.cat([graph.edge_index.t(), graph.edge_shift], dim=1).tolist()
    result: list[set[tuple[int, ...]]] = [set() for _ in graph.num_edges]
    for edge, batch in zip(edges, graph.edge_batch.tolist()):
        result[batch].add(tuple(edge))
    return result


@pytest.fixture(scope="module")
def pretrained() -> ALIGNN:
    return load_pretrained(ALIGNN, "alignn", NAME)


@pytest.mark.timeout(120)
def test_pretrained_line_graph_matches_alignn(pretrained):
    """Line graphs of alignn 2026.8.11, computed once outside of the project."""
    structures, reference, graph = reference_structures()
    _, _, pairs, cosine = pretrained.graphs(with_graph(structures, graph))
    pair_index = pairs.index
    n_struct = len(structures.num_atoms)

    pair_batch = graph.edge_batch[pair_index[0]]
    num_angles = torch.bincount(pair_batch, minlength=n_struct)
    assert num_angles.tolist() == reference["num_angles"].tolist()

    cosine_sum = torch.zeros(n_struct, dtype=torch.float64).index_add_(
        0, pair_batch, cosine.double()
    )
    assert np.allclose(cosine_sum.numpy(), reference["cosine_sum"], atol=1e-2)


@pytest.mark.timeout(120)
def test_pretrained_matches_alignn_on_its_graph(pretrained):
    """Predictions of alignn 2026.8.11, computed once outside of the project."""
    structures, reference, graph = reference_structures()

    prediction = pretrained(with_graph(structures, graph)).numpy()

    assert np.abs(prediction - reference["prediction"]).max() < 1e-4


@pytest.mark.timeout(120)
def test_pretrained_matches_alignn_without_equidistant_neighbours(pretrained):
    structures, reference, graph = reference_structures()
    own, _, _, _ = pretrained.graphs(pretrained.transform()(structures))

    assert own.num_edges.tolist() == reference["num_edges"].tolist()

    # among neighbours at the same distance, alignn does not keep the same ones
    same = np.array([a == b for a, b in zip(edges_of(own), edges_of(graph))])
    assert same.sum() >= 10 and (~same).sum() >= 10

    error = np.abs(predict(pretrained, structures).numpy() - reference["prediction"])
    assert error[same].max() < 1e-4
    assert error[~same].max() < 0.05
