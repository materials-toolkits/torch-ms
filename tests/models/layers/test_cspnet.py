"""Each layer of CSPNet against DiffCSP, with its pretrained weights.

The inputs and outputs of each layer were recorded from DiffCSP by
tests/references/reference_layers_diffcsp.py, for its two models: the one
given the elements (diffcsp-mp20) and the one which also generates them
(diffcsp-gen-mp20). Each test gives a layer of torchms, loaded from the same
checkpoint, the inputs the original layer received, on the complete graph
DiffCSP built, and compares its outputs with the original's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from helpers.compare import assert_close
from helpers.matching import permutation
from helpers.references import Reference, load_reference, submodule_inputs
from helpers.weights import pretrained
from torchms.data.batching import Structures
from torchms.graph.graph import Graph
from torchms.graph.transforms import AddCompleteGraph
from torchms.models import DiffCSP, DiffCSPGeneration
from torchms.models.layers.basis import periodic_fourier
from torchms.models.layers.cspnet import CSPNet

# loading a checkpoint
pytestmark = pytest.mark.timeout(60)


@dataclass(frozen=True)
class Recorded:
    """A model of DiffCSP, what its layers did, and the crystals on its graph."""

    decoder: CSPNet
    reference: Reference
    structures: Structures
    generation: bool


MODELS = {
    "diffcsp-mp20": (DiffCSP, "layers/diffcsp_cspnet.npz"),
    "diffcsp-gen-mp20": (DiffCSPGeneration, "layers/diffcsp_cspnet_generation.npz"),
}


def on_upstream_graph(reference: Reference) -> Structures:
    """The recorded crystals, with the complete graph DiffCSP built (gen_edges).

    An edge (i, j) of DiffCSP gives the message of i, its sender here, from the
    difference x_j - x_i, its receiver: no periodic image, a shift of zero.
    """
    call = reference.call("network")
    structures = Structures(
        lattice=call.input("lattices"),
        x=call.input("frac_coords"),
        z=torch.zeros(len(call.input("frac_coords")), dtype=torch.long),
        num_atoms=call.input("num_atoms"),
    )
    edges = reference.call("graph").output("0")
    graph = Graph.from_edges(
        structures,
        structures.num_atoms,
        edges,
        torch.zeros(edges.shape[1], 3, dtype=torch.long),
        edge_batch=structures.batch[edges[0]],
    )

    return replace(structures, graph=graph)


@pytest.fixture(scope="module", params=sorted(MODELS))
def recorded(request) -> Recorded:
    model_class, path = MODELS[request.param]
    model = pretrained(model_class, "diffcsp", request.param)
    reference = load_reference(path)

    return Recorded(
        model.decoder,
        reference,
        on_upstream_graph(reference),
        model_class is DiffCSPGeneration,
    )


def edge_features(recorded: Recorded) -> torch.Tensor:
    """What a layer of torchms reads on an edge, from what DiffCSP computed.

    The inner products of the lattice vectors of its structure, then the
    sines and cosines of the difference of the coordinates (dis_emb).
    """
    graph = recorded.structures.graph
    assert graph is not None
    lattice = recorded.structures.lattice
    products = (lattice @ lattice.transpose(-1, -2)).flatten(start_dim=1)

    return torch.cat(
        [
            graph.unpool_senders(graph.unpool_structures(products)),
            recorded.reference.call("dis_emb").output(),
        ],
        dim=1,
    )


def element(recorded: Recorded) -> torch.Tensor:
    """The element of each atom as torchms reads it: from 0 for hydrogen."""
    atom_types = recorded.reference.call("network").input("atom_types")
    # the generation reads a vector of the elements as it is
    return atom_types if recorded.generation else atom_types - 1


def test_graph(recorded):
    """AddCompleteGraph builds the edges of DiffCSP, with the same differences."""
    graph = recorded.structures.graph
    assert graph is not None
    complete = AddCompleteGraph(self_loops=True)(recorded.structures).graph
    assert complete is not None
    permutation(complete.edge_index.T, graph.edge_index.T)

    x = recorded.structures.x
    difference = (graph.unpool_receivers(x) - graph.unpool_senders(x)) % 1.0
    expected = recorded.reference.call("graph").output("1")
    assert_close(difference, expected, name="differences of the coordinates")


def test_periodic_fourier(recorded):
    """SinusoidsEmbedding of DiffCSP: sines then cosines, by coordinate."""
    call = recorded.reference.call("dis_emb")
    found = periodic_fourier(call.input("x"), recorded.decoder.config.num_freqs)
    assert_close(found, call.output(), name="basis of the differences")


def test_node_embedding(recorded):
    call = recorded.reference.call("node_embedding")
    found = recorded.decoder.node_embedding(element(recorded))
    assert_close(found, call.output(), name="embedding of the elements")


def test_atom_latent_embedding(recorded):
    call = recorded.reference.call("atom_latent_emb")
    found = recorded.decoder.atom_latent_emb(call.input("input"))
    assert_close(found, call.output(), name="embedding with the time")


@pytest.mark.parametrize("layer", ["csp_layer_0", "csp_layer_5"])
def test_csp_layer(recorded, layer):
    """A layer, and what it gives its two networks.

    The last layer adds an update of 350 to features of 50000: its output
    alone would hide an error of the update. What the layer gives its edge
    network (normalized features of the two atoms, edge features) and its
    node network (the mean of the messages of each atom) is compared too.
    """
    call = recorded.reference.call(layer)
    graph = recorded.structures.graph
    assert graph is not None
    csp_layer = recorded.decoder.get_submodule(layer)

    with submodule_inputs(csp_layer, ["edge_mlp", "node_mlp"]) as inputs:
        found = csp_layer(graph, call.input("node_features"), edge_features(recorded))

    assert_close(found, call.output(), name=layer)
    for network in ("edge_mlp", "node_mlp"):
        expected = recorded.reference.call(f"{layer}.{network}").input("input")
        assert_close(inputs[network], expected, name=f"input of {layer}.{network}")


@pytest.mark.parametrize("layer", ["csp_layer_0", "csp_layer_5"])
@pytest.mark.parametrize("network", ["edge_mlp", "node_mlp"])
def test_csp_layer_networks(recorded, layer, network):
    call = recorded.reference.call(f"{layer}.{network}")
    found = recorded.decoder.get_submodule(f"{layer}.{network}")(call.input("input"))
    assert_close(found, call.output(), name=f"{layer}.{network}")


def test_read_out(recorded):
    """The last layer norm and the linear read-outs, of the elements too."""
    layers = ["final_layer_norm", "coord_out", "lattice_out"]
    for layer in layers + (["type_out"] if recorded.generation else []):
        call = recorded.reference.call(layer)
        found = recorded.decoder.get_submodule(layer)(call.input("input"))
        assert_close(found, call.output(), name=layer)


def test_network(recorded):
    """CSPNet as a whole: 6 layers, on the recorded graph."""
    call = recorded.reference.call("network")
    graph = recorded.structures.graph
    assert graph is not None

    lattice_out, coord_out, type_out = recorded.decoder(
        call.input("t"),
        element(recorded),
        graph,
        call.input("frac_coords"),
        call.input("lattices"),
    )

    assert_close(lattice_out, call.output("0"), name="lattice")
    assert_close(coord_out, call.output("1"), name="coordinates")
    if recorded.generation:
        assert type_out is not None
        assert_close(type_out, call.output("2"), name="elements")
    else:
        assert type_out is None
