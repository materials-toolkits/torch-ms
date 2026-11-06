"""Tests for the rewrite of the energy prediction of MatterSim."""

import math

import pytest
import torch

from helpers.compare import assert_close
from helpers.models import predict
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, make_batch, make_structures
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
from torchms.models.layers.basis import SphericalBesselBasis, legendre
from torchms.models.mattersim import Config, InvalidWeights, MatterSim
from torchms.models.mattersim.model import SphericalBasisLayer

NAME = "mattersim-v1.0.0-1m"
LARGE = "mattersim-v1.0.0-5m"

# the weights and the file of their reference
REFERENCES = {
    NAME: "mattersim_reference.npz",
    LARGE: "mattersim_5m_reference.npz",
}

# thousands of angles per structure: slower than the default timeout
pytestmark = pytest.mark.timeout(30)


@pytest.fixture(scope="module")
def model() -> MatterSim:
    """Model with random weights, and a plausible scale of the atoms."""
    torch.manual_seed(0)
    model = MatterSim().eval().requires_grad_(False)
    model.normalizer.shift.fill_(-3.0)

    return model


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["mattersim"]) == {NAME, LARGE}
    assert Config.load().default_weights == NAME

    for name, filename in (
        (NAME, "mattersim-v1.0.0-1M.pth"),
        (LARGE, "mattersim-v1.0.0-5M.pth"),
    ):
        (checkpoint,) = weights.CHECKPOINTS["mattersim"][name]
        # a commit of the repository of the authors, not a branch
        assert checkpoint.url.startswith(
            "https://raw.githubusercontent.com/microsoft/mattersim/"
        )
        assert "/main/" not in checkpoint.url
        assert len(checkpoint.sha256) == 64

        (path,) = weights.checkpoint_paths("mattersim", name, tmp_path)
        assert path == tmp_path / "mattersim" / filename


def test_network_of_each_weights():
    config = Config.load()
    small, large = MatterSim.for_weights(), MatterSim.for_weights(LARGE)

    assert small.config is config and MatterSim.for_weights(NAME).config == config
    assert len(small.graph_conv) == 3 and small.final.g[0].linear.in_features == 128
    assert len(large.graph_conv) == 4 and large.final.g[0].linear.in_features == 256
    assert large.default_weights() == LARGE
    assert large.config.upstream.model_args["units"] == 256
    # the graph does not change
    assert MatterSim.transform(large.config) == MatterSim.transform()

    with pytest.raises(KeyError, match="no network"):
        MatterSim.for_weights("mattersim-v9")

    # the weights of one size are refused by the network of the other one
    with pytest.raises(InvalidWeights):
        large.load_pretrained(small.state_dict(), config.upstream.model_args)
    with pytest.raises(InvalidWeights, match="other arguments"):
        small.load_pretrained(large.state_dict(), large.config.upstream.model_args)


# the basis of the angles


def test_closed_forms_are_the_spherical_bessel_functions():
    # the closed forms of MatterSim and the recurrence of torchms agree, in
    # double precision, away from the short lengths where both lose digits
    network, radius = Config.load().model, 5.0
    closed = SphericalBasisLayer(network, radius).double()
    recurrence = SphericalBesselBasis(network.max_l, network.max_n, radius).double()
    distance = torch.linspace(1.0, 4.0, 50, dtype=torch.float64)
    angle = torch.linspace(0.1, 3.0, 50, dtype=torch.float64)

    expected = recurrence(distance) * legendre(
        network.max_l, torch.cos(angle)
    ).repeat_interleave(network.max_n, dim=1)

    assert torch.allclose(closed(distance, angle), expected, atol=1e-6)


def test_closed_forms_at_short_lengths():
    network = Config.load().model
    closed = SphericalBasisLayer(network, 5.0)
    values = closed(torch.tensor([0.5, 1.0]), torch.tensor([math.pi / 3, math.pi / 2]))

    assert values.shape == (2, network.basis_dim)
    assert values.isfinite().all()


# the model with random weights


def test_model_output(model):
    energy = predict(model, make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float32
    assert energy.isfinite().all()


def test_atom_energies_sum_to_the_energy(model):
    structures = model.transform()(make_batch())
    per_atom = model.atom_energies(structures)

    assert per_atom.shape == structures.z.shape
    assert torch.allclose(
        structures.graph.aggregate_nodes_to_structures(per_atom), model(structures)
    )


def test_scaling_by_element(model):
    structures = model.transform()(Structures.diamond())
    before = model.atom_energies(structures)

    scale, shift = model.normalizer.scale.clone(), model.normalizer.shift.clone()
    try:
        model.normalizer.scale[6] = 2.0
        model.normalizer.shift[6] = 1.0
        after = model.atom_energies(structures)
    finally:
        model.normalizer.scale.copy_(scale)
        model.normalizer.shift.copy_(shift)

    raw = before - shift[6]
    assert torch.allclose(after, 2.0 * raw + 1.0, atol=1e-5)


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


def test_a_molecule(model):
    # without periodicity: no periodic image, the bonds of the molecule only
    energy = predict(model, Structures.water())

    assert energy.shape == (1,) and energy.isfinite().all()


def test_model_unknown_element(model):
    # americium is after the last atomic number of the embedding
    assert Config.load().model.max_z == 94
    predict(
        model,
        Structures(
            lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[94]
        ),
    )

    for z in (0, 95):
        structures = Structures(
            lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[z]
        )
        with pytest.raises(ValueError, match="not supported"):
            predict(model, structures)


def test_needs_its_graphs(model):
    with pytest.raises(ValueError, match=r"apply MatterSim.transform\(\)"):
        model(make_batch())


def test_load_pretrained_checks_the_parameters(model):
    config = Config.load()
    state_dict = dict(model.state_dict())

    other = MatterSim().eval()
    other.load_pretrained(state_dict, config.upstream.model_args, "m3gnet")
    assert torch.allclose(predict(other, make_batch()), predict(model, make_batch()))

    with pytest.raises(InvalidWeights, match="other arguments"):
        other.load_pretrained(state_dict, {**config.upstream.model_args, "units": 256})
    with pytest.raises(InvalidWeights, match="graphormer"):
        other.load_pretrained(state_dict, model_name="graphormer")
    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state_dict, "unknown.weight": torch.zeros(1)})

    del state_dict["normalizer.scale"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict)


# the pretrained model against MatterSim


@pytest.fixture(scope="module", params=list(REFERENCES))
def weights_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def pretrained(weights_name) -> MatterSim:
    return load_pretrained(MatterSim, "mattersim", weights_name)


@pytest.fixture(scope="module")
def reference(weights_name):
    return load_reference(REFERENCES[weights_name])


@pytest.mark.timeout(120)
def test_pretrained_graphs_match_mattersim(pretrained, reference):
    """Bonds and angles of MatterSim 1.2.5, built by pymatgen."""
    structures = pretrained.transform()(reference_structures())

    assert torch.equal(structures.graph.num_edges, reference["num_edges"])
    assert torch.equal(structures.triplets.num_tuples, reference["num_triplets"])

    lengths = structures.graph.edges_distances().double()
    assert_close(
        structures.graph.aggregate_edges_to_structures(lengths),
        reference["edge_length_sum"],
        name="sum of the lengths",
    )


@pytest.mark.timeout(120)
def test_pretrained_matches_mattersim(pretrained, reference):
    """Energies of MatterSim 1.2.5, one structure at a time as its calculator."""
    structures = pretrained.transform()(reference_structures())

    per_atom = pretrained.atom_energies(structures)
    assert_close(per_atom, reference["atom_energy"], name="energy of the atoms")

    energy = pretrained(structures)
    error = (energy.double() - reference["energy"]).abs() / structures.num_atoms
    assert error.max() < 1e-5


@pytest.mark.timeout(120)
def test_pretrained_is_as_near_to_double_precision_as_mattersim(pretrained, reference):
    structures = pretrained.transform()(reference_structures())
    energy = pretrained(structures).double()

    ours = (energy - reference["energy_double"]).abs() / structures.num_atoms
    theirs = (
        reference["energy"] - reference["energy_double"]
    ).abs() / structures.num_atoms
    assert ours.max() <= 3 * theirs.max()
