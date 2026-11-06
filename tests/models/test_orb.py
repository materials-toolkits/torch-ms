"""Tests for the rewrite of orb-v3: its blocks, random weights, and against orb-models."""

import numpy as np
import pytest
import torch

from helpers.models import predict
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, tilted
from helpers.weights import cached_or_skip, pretrained
from torchms.data import Structures
from torchms.models import OrbV3
from torchms.models.estimator import weights
from torchms.models.orb import Config
from torchms.models.orb.model import InvalidWeights, ZBLBasis, build_mlp
from torchms.graph.graph import Graph


def test_pinned_checkpoints():
    assert set(weights.CHECKPOINTS["orb"]) == {
        "orb-v3-conservative-inf-omat",
        "orb-v3-conservative-inf-mpa",
    }
    for (checkpoint,) in weights.CHECKPOINTS["orb"].values():
        assert checkpoint.url.startswith("https://")
        assert len(checkpoint.sha256) == 64


# building blocks of the network


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
    config = Config.load()
    repulsion = ZBLBasis(config.repulsion)
    z = torch.tensor([6, 6])
    graph = Graph.from_edges(
        None,
        torch.tensor([2]),
        torch.tensor([[0, 1], [1, 0]]),
        torch.zeros(2, 3, dtype=torch.int64),
        edge_batch=torch.tensor([0, 0]),
    )

    def energy(distance: float) -> float:
        vectors = torch.tensor([[distance, 0.0, 0.0], [-distance, 0.0, 0.0]])
        return float(repulsion(z, vectors, graph))

    # cut at the covalent radii of the next element (nitrogen), see ZBLBasis
    limit = 2 * config.repulsion.covalent_radii[7]
    assert energy(limit + 0.01) == 0.0
    assert energy(1.0) > energy(1.2) > 0.0
    assert len(config.repulsion.covalent_radii) == 119


# the model, with random weights


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return OrbV3().double().eval().requires_grad_(False)


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


def supercell(structures: Structures) -> Structures:
    """Two cells along the first lattice vector."""
    half = structures.x * torch.tensor([0.5, 1.0, 1.0])
    return Structures(
        lattice=structures.lattice * torch.tensor([2.0, 1.0, 1.0]).view(1, 3, 1),
        x=torch.cat([half, half + torch.tensor([0.5, 0.0, 0.0])]),
        z=structures.z.repeat(2),
    )


def repulsion(model: OrbV3, structures: Structures) -> torch.Tensor:
    graph, vectors = model.edges(model.transform()(structures))
    return model.pair_repulsion_fn(structures.z, vectors, graph)


def test_model_output(model):
    structures = double(Structures.collate([Structures.diamond(), tilted()]))

    energy = predict(model, structures)

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert torch.isfinite(energy).all()


def test_model_graph(model):
    config = Config.load()
    graph, vectors = model.edges(model.transform()(double(Structures.diamond())))
    degree = torch.bincount(graph.edge_index[0], minlength=8)

    # diamond has more than 120 atoms closer than 6 Å
    assert degree.tolist() == [config.graph.max_neighbors] * 8
    assert vectors.norm(dim=1).max() <= config.graph.radius
    assert torch.allclose(vectors.norm(dim=1), graph.edges_distances())

    graph, vectors = model.edges(model.transform()(double(tilted())))
    assert (
        torch.bincount(graph.edge_index[0], minlength=3).max()
        < config.graph.max_neighbors
    )
    assert torch.allclose(vectors.norm(dim=1), graph.edges_distances())


def test_model_batch_equals_single(model):
    parts = [double(Structures.zincblende()), double(tilted())]

    batched = predict(model, Structures.collate(parts))
    single = torch.cat([predict(model, part) for part in parts])

    assert torch.allclose(batched, single, rtol=1e-10)


def test_model_invariances(model):
    structures = double(tilted())
    reference = predict(model, structures)

    def variant(lattice=None, x=None, z=None) -> torch.Tensor:
        return predict(
            model,
            Structures(
                lattice=structures.lattice if lattice is None else lattice,
                x=structures.x if x is None else x,
                z=structures.z if z is None else z,
            ),
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
        predict(model, supercell(structures)), 2 * predict(model, structures), rtol=1e-4
    )


def test_model_pair_repulsion_is_a_mean_over_atoms(model):
    """As in orb-models for these weights, the repulsion does not grow with the cell."""
    structures = double(tilted())
    doubled = supercell(structures)

    assert repulsion(model, structures) > 0.0
    assert torch.allclose(repulsion(model, doubled), repulsion(model, structures))

    rest = predict(model, structures) - repulsion(model, structures)
    assert torch.allclose(
        predict(model, doubled) - repulsion(model, doubled), 2 * rest, rtol=1e-4
    )


def test_load_pretrained_checks_the_parameters(model):
    config = Config.load()
    state_dict = dict(model.state_dict())
    state_dict["heads.energy.reference.linear.weight"] = state_dict[
        "heads.energy.reference.linear.weight"
    ].flatten()
    state_dict.update(
        {f"{prefix}weight": torch.zeros(1) for prefix in config.unused_parameters}
    )

    other = OrbV3().double()
    other.load_pretrained(state_dict)
    assert torch.allclose(
        predict(other, double(tilted())), predict(model, double(tilted()))
    )

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "model.unknown.weight": torch.zeros(1)})

    del state_dict["model.atom_emb.embeddings.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


# the pretrained model against the real orb-models


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", ["omat", "mpa"])
def test_pretrained_matches_orb_models(name):
    """Energies of orb-models 0.7.0, computed once outside of the project."""
    config = Config.load()
    model = pretrained(OrbV3, "orb", f"orb-v3-conservative-inf-{name}")
    structures, data = (
        reference_structures(),
        load_reference("orb_reference.npz").numpy(),
    )

    graph, _ = model.edges(model.transform()(structures))
    with torch.no_grad():
        energy = predict(model, structures).numpy()

    assert graph.num_edges.tolist() == data["num_edges"].tolist()

    error = np.abs(energy - data[f"energy_{name}"]) / data["num_atoms"]
    # atoms with more than 120 neighbours: equidistant ones are not kept in
    # the same order as orb-models
    capped = data["max_degree"] >= config.graph.max_neighbors
    assert capped.sum() >= 3
    assert error[~capped].max() < 2e-5
    assert error[capped].max() < 5e-4


@pytest.mark.timeout(300)
def test_pretrained_is_almost_invariant_to_rotations():
    """The network is not equivariant: it learned the invariance of the energy."""
    model = pretrained(OrbV3, "orb", "orb-v3-conservative-inf-omat")
    structures = reference_structures()
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator()))[0]
    rotated = Structures(
        lattice=structures.lattice @ rotation,
        x=structures.x,
        z=structures.z,
        num_atoms=structures.num_atoms,
    )

    with torch.no_grad():
        change = (
            predict(model, rotated) - predict(model, structures)
        ).abs() / structures.num_atoms

    assert change.max() > 1e-7
    assert change.mean() < 1e-3
    assert change.max() < 2e-2


def test_pretrained_state_dict_is_fully_used():
    config = Config.load()
    cached_or_skip("orb", "orb-v3-conservative-inf-omat")
    path = weights.checkpoint_paths("orb", "orb-v3-conservative-inf-omat")[0]

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    used = {k for k in state_dict if not k.startswith(config.unused_parameters)}

    assert used == set(OrbV3().state_dict())
    assert len(state_dict) - len(used) == 17
