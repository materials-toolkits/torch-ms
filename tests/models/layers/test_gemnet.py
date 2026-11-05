"""Each layer of GemNet-T against MatterGen and cdvae, with their pretrained weights.

The inputs and outputs of each layer were recorded from the original codes by
tests/references/reference_layers_mattergen.py and reference_layers_cdvae.py,
on small crystals. Each test gives a layer of torchms, loaded from the same
checkpoint, the inputs the original layer received, on the graph the original
built (helpers.gemnet), and compares its outputs with the original's.

The layers both models have are tested on both: MatterGen (512 features, the
cosines with the lattice and its score) and the decoder of cdvae (128). The
layers of the lattice and the adapters of a property are MatterGen's alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from helpers.compare import assert_close
from helpers.gemnet import UpstreamGraph, on_upstream_graph
from helpers.references import Reference, load_reference
from helpers.weights import pretrained
from torchms.graph.geometry import triplet_cosines
from torchms.models import CDVAE, MatterGen, MatterGenChemicalSystem
from torchms.models.layers.gemnet import Condition, GemNetT, Output, lattice_cosines

# loading a checkpoint of 460 MB
pytestmark = pytest.mark.timeout(60)

# the move of an atom sums the scores of its edges times their directions,
# which cancel (to 0 for a crystal of one atom): its rounding is the one of the
# scores, each a sum of products of features of order 1. It is held to 1e-5
# of 1, where each output block is held to 1e-5 of its largest score.
MOVE_SCALE = 1.0


@dataclass(frozen=True)
class Recorded:
    """A GemNet-T of torchms, what the original's layers did, and its graph.

    Attributes:
        model: "mattergen" or "cdvae".
        gemnet: The network of torchms, with the pretrained weights.
        reference: What the layers of the original received and returned.
        upstream: The recorded crystals on the graph of the original.
    """

    model: str
    gemnet: GemNetT
    reference: Reference
    upstream: UpstreamGraph


def mattergen_gemnet() -> GemNetT:
    return pretrained(MatterGen, "mattergen", "mattergen-base").gemnet


def cdvae_gemnet() -> GemNetT:
    return pretrained(CDVAE, "cdvae", "cdvae-mp20-cowboys").decoder.gemnet


MODELS = {
    "mattergen": (mattergen_gemnet, "layers/mattergen_gemnet.npz"),
    "cdvae": (cdvae_gemnet, "layers/cdvae_gemnet.npz"),
}


def load(model: str) -> Recorded:
    gemnet, path = MODELS[model]
    reference = load_reference(path)

    return Recorded(model, gemnet(), reference, on_upstream_graph(reference))


@pytest.fixture(scope="module", params=sorted(MODELS))
def recorded(request) -> Recorded:
    return load(request.param)


@pytest.fixture(scope="module")
def mattergen() -> Recorded:
    return load("mattergen")


def cbf_in_torchms_order(recorded: Recorded) -> torch.Tensor:
    """Harmonics of each triplet the original computed, in torchms order."""
    harmonics = recorded.reference.call("cbf_basis3").output("1")

    return harmonics[recorded.upstream.triplet_order]


def cosines_in_torchms_order(recorded: Recorded) -> torch.Tensor:
    """Cosine of each triplet the original computed, in torchms order."""
    cosines = recorded.reference.call("cbf_basis3").input("cosφ_cab")

    return cosines[recorded.upstream.triplet_order]


def towards(upstream: UpstreamGraph) -> torch.Tensor:
    """Direction of each edge as GemNet-T reads it: V_st of the original."""
    graph = upstream.structures.graph
    assert graph is not None and graph.edge_vector is not None

    return -graph.edge_vector / graph.edges_distances().unsqueeze(-1)


def test_graph(recorded):
    """torchms finds the triplets and the bonds of the original on its edges."""
    structures = recorded.upstream.structures
    assert structures.graph is not None and structures.triplets is not None
    assert structures.bonds is not None

    swap = torch.empty_like(recorded.upstream.id_swap)
    swap[structures.bonds.index[0]] = structures.bonds.index[1]
    swap[structures.bonds.index[1]] = structures.bonds.index[0]
    assert torch.equal(swap, recorded.upstream.id_swap)

    cosines = triplet_cosines(structures.graph.edge_vector, structures.triplets)
    assert_close(
        cosines, cosines_in_torchms_order(recorded), name="cosines of the triplets"
    )


def test_radial_basis(recorded):
    call = recorded.reference.call("radial_basis")
    found = recorded.gemnet.radial_basis(call.input("d"))
    assert_close(found, call.output(), name="rbf")


def test_circular_basis(recorded):
    call = recorded.reference.call("cbf_basis3")

    radial, harmonics = recorded.gemnet.cbf_basis3(
        call.input("D_ca"), cosines_in_torchms_order(recorded)
    )

    assert_close(radial, call.output("0")[0], name="radial part")
    assert_close(harmonics, cbf_in_torchms_order(recorded), name="cbf")


@pytest.mark.parametrize("name", ["mlp_rbf3", "mlp_rbf_h", "mlp_rbf_out"])
def test_radial_projection(recorded, name):
    call = recorded.reference.call(name)
    found = getattr(recorded.gemnet, name)(call.input("x"))
    assert_close(found, call.output(), name=name)


def test_circular_projection(recorded):
    """EfficientInteractionDownProjection: its radial part; torchms pads nothing."""
    call = recorded.reference.call("mlp_cbf3")
    found = recorded.gemnet.mlp_cbf3(call.input("rbf")[0])
    assert_close(found, call.output("0"), name="rbf_W1")


def test_atom_embedding(recorded):
    call = recorded.reference.call("atom_emb")
    # the original subtracts 1 from the atomic number, the models of torchms
    # before GemNet-T
    rows = call.input("Z") - 1
    found = recorded.gemnet.atom_emb(rows)
    assert_close(found, call.output(), name="atom embedding")


def test_atom_latent_embedding(recorded):
    call = recorded.reference.call("atom_latent_emb")
    found = recorded.gemnet.atom_latent_emb(call.input("input"))
    assert_close(found, call.output(), name="atom and latent embedding")


def test_edge_embedding(recorded):
    call = recorded.reference.call("edge_emb")
    graph = recorded.upstream.structures.graph
    found = recorded.gemnet.edge_emb(call.input("h"), call.input("m_rbf"), graph)
    assert_close(found, call.output(), name="edge embedding")


def test_output_block(recorded):
    call = recorded.reference.call("out_blocks.0")
    found = recorded.gemnet.out_blocks[0](call.input("m"), call.input("rbf"))
    assert_close(found, call.output("1").squeeze(-1), name="score of the edges")


def test_triplet_interaction(recorded):
    call = recorded.reference.call("int_blocks.0.trip_interaction")
    structures = recorded.upstream.structures
    assert structures.triplets is not None and structures.bonds is not None

    found = recorded.gemnet.int_blocks[0].trip_interaction(
        call.input("m"),
        call.input("rbf3"),
        call.input("cbf3:0"),
        cbf_in_torchms_order(recorded),
        structures.triplets,
        structures.bonds,
    )
    assert_close(found, call.output(), name="triplet interaction")


def test_atom_update(recorded):
    call = recorded.reference.call("int_blocks.0.atom_update")
    found = recorded.gemnet.int_blocks[0].atom_update(
        call.input("m"), call.input("rbf"), recorded.upstream.structures.graph
    )
    assert_close(found, call.output(), name="atom update")


def test_interaction_block(recorded):
    call = recorded.reference.call("int_blocks.0")
    structures = recorded.upstream.structures
    assert structures.triplets is not None and structures.bonds is not None

    h, m = recorded.gemnet.int_blocks[0](
        call.input("h"),
        call.input("m"),
        call.input("rbf3"),
        call.input("cbf3:0"),
        cbf_in_torchms_order(recorded),
        call.input("rbf_h"),
        structures.graph,
        structures.triplets,
        structures.bonds,
    )
    assert_close(h, call.output("0"), name="atoms after the block")
    assert_close(m, call.output("1"), name="edges after the block")


def run_network(recorded: Recorded, conditions: tuple[Condition, ...] = ()) -> Output:
    """The whole GemNet-T on the inputs the original gave it, on its graph."""
    call = recorded.reference.call("network")
    structures = recorded.upstream.structures
    graph = structures.graph
    assert graph is not None and graph.edge_vector is not None
    assert structures.triplets is not None and structures.bonds is not None

    return recorded.gemnet(
        call.input("z"),
        call.input("atom_types") - 1,
        graph,
        structures.triplets,
        structures.bonds,
        graph.edge_vector,
        cosines_in_torchms_order(recorded),
        structures.lattice,
        conditions,
    )


def test_network(recorded):
    """GemNet-T as a whole, on the recorded graph: 4 blocks, 3 for cdvae."""
    call = recorded.reference.call("network")
    output = run_network(recorded)

    if recorded.model == "mattergen":
        features, move = call.output("node_embeddings"), call.output("forces")
    else:
        # cdvae returns the features of the atoms and their moves
        features, move = call.output("0"), call.output("1")
    assert_close(output.features, features, name="features")
    assert_close(output.move, move, name="move of the atoms", scale=MOVE_SCALE)

    if recorded.model == "mattergen":
        assert output.lattice is not None
        assert_close(output.lattice, call.output("stress"), name="lattice score")
    else:
        assert output.lattice is None


# the cosines with the lattice and its score, of MatterGen


def test_lattice_cosines(mattergen):
    """The cosines of each edge with the lattice vectors MatterGen appends."""
    call = mattergen.reference.call("angle_edge_emb")
    structures = mattergen.upstream.structures
    found = lattice_cosines(
        towards(mattergen.upstream), structures.lattice, structures.graph
    )
    assert_close(found, call.input("input")[:, -3:], name="lattice cosines")


def test_angle_edge_embedding(mattergen):
    call = mattergen.reference.call("angle_edge_emb")
    found = mattergen.gemnet.angle_edge_emb(call.input("input"))
    assert_close(found, call.output(), name="edge embedding with the cosines")


def test_lattice_radial_projection(mattergen):
    call = mattergen.reference.call("mlp_rbf_lattice")
    found = mattergen.gemnet.mlp_rbf_lattice(call.input("x"))
    assert_close(found, call.output(), name="mlp_rbf_lattice")


def test_lattice_output_block(mattergen):
    call = mattergen.reference.call("lattice_out_blocks.0")
    direction = towards(mattergen.upstream)
    outer = direction.unsqueeze(2) * direction.unsqueeze(1)
    found = mattergen.gemnet.lattice_out_blocks[0](
        call.input("edge_emb"),
        call.input("rbf"),
        outer,
        mattergen.upstream.structures.graph,
    )
    assert_close(found, call.output(), name="score of the lattice")


# the adapters of a property, from MatterGen fine-tuned on the chemical system


@pytest.fixture(scope="module")
def adapters() -> Recorded:
    gemnet = pretrained(
        MatterGenChemicalSystem, "mattergen", "mattergen-chemical-system"
    ).gemnet
    reference = load_reference("layers/mattergen_gemnet_adapters.npz")

    return Recorded("mattergen", gemnet, reference, on_upstream_graph(reference))


def chemical_system(reference: Reference) -> Condition:
    """The condition MatterGen gave: active where its mask is not set."""
    call = reference.call("network")
    mask = call.input("cond_adapt_mask:chemical_system").squeeze(-1)

    return Condition(call.input("cond_adapt:chemical_system"), ~mask)


def test_adapter_layers(adapters):
    adapt = adapters.reference.call("cond_adapt_layers.chemical_system.0")
    mixin = adapters.reference.call("cond_mixin_layers.chemical_system.0")

    found = adapters.gemnet.cond_adapt_layers["chemical_system"][0](
        adapt.input("input")
    )
    assert_close(found, adapt.output(), name="adapter")
    found = adapters.gemnet.cond_mixin_layers["chemical_system"][0](
        mixin.input("input")
    )
    assert_close(found, mixin.output(), name="mixin")


def test_adapt(adapters):
    """The atoms before the first block, with the adapters added."""
    adapt = adapters.reference.call("cond_adapt_layers.chemical_system.0")
    mixin = adapters.reference.call("cond_mixin_layers.chemical_system.0")
    condition = chemical_system(adapters.reference)
    h = adapt.input("input")[:, : adapters.gemnet.config.hidden_dim]
    graph = adapters.upstream.structures.graph
    assert graph is not None

    found = adapters.gemnet.adapt(0, h, (condition,), graph)

    active = graph.unpool_structures(condition.active.float()).unsqueeze(-1)
    assert_close(found, h + active * mixin.output(), name="adapted atoms")


def test_conditioned_network(adapters):
    call = adapters.reference.call("network")
    output = run_network(adapters, (chemical_system(adapters.reference),))

    assert_close(output.features, call.output("node_embeddings"), name="features")
    assert_close(
        output.move, call.output("forces"), name="move of the atoms", scale=MOVE_SCALE
    )
    assert output.lattice is not None
    assert_close(output.lattice, call.output("stress"), name="lattice score")
