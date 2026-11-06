"""Each layer of DimeNet++ against the encoder of cdvae, with its pretrained weights.

The inputs and outputs of each layer were recorded from cdvae by
tests/references/reference_layers_cdvae.py, on small crystals with their
CrystalNN graphs. Each test gives a layer of torchms, loaded from the same
checkpoint, the inputs the original layer received, on the graph and the pairs
of bonds of cdvae, and compares its outputs with the original's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import pytest
import torch

from helpers.compare import assert_close
from helpers.matching import permutation
from helpers.references import Reference, load_reference
from helpers.weights import pretrained
from torchms.data.batching import Structures
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import Graph
from torchms.models import CDVAE
from torchms.models.layers.dimenet import DimeNetPlusPlus

# loading a checkpoint
pytestmark = pytest.mark.timeout(60)

# under this length (Å), the spherical Bessel functions lose digits in float32
SHORT_BOND = 2.0


@dataclass(frozen=True)
class Recorded:
    """The encoder of torchms, what cdvae's layers did, and its graph.

    Attributes:
        encoder: DimeNet++ of torchms, with the pretrained weights.
        reference: What the layers of cdvae received and returned.
        structures: The crystals with the CrystalNN graph of cdvae and its pairs
            of bonds, as torchms builds them.
        pair_order: Index of the pair of cdvae for each pair of torchms.
    """

    encoder: DimeNetPlusPlus
    reference: Reference
    structures: Structures
    pair_order: torch.Tensor

    def in_torchms_order(self, values: torch.Tensor) -> torch.Tensor:
        """Values of the pairs of cdvae, in the order of the pairs of torchms."""
        return values[self.pair_order]

    def cosines(self) -> torch.Tensor:
        """Cosine of the angle of each pair, from the one cdvae computed."""
        return self.in_torchms_order(
            torch.cos(self.reference.call("sbf").input("angle"))
        )


def on_upstream_graph(reference: Reference) -> tuple[Structures, torch.Tensor]:
    """The recorded crystals on their CrystalNN graph, and the order of the pairs.

    An edge (j, i) of cdvae, with the image of j, is the edge i → j: its
    messages go to i, the sender. A pair of cdvae is two edges (k → j, j → i),
    idx_kj and idx_ji, the edges i → j, j → k here: the first one gets the
    features of the second, as the pairs of AddLineGraph.
    """
    structures = Structures(
        lattice=reference["lattice"],
        x=reference["frac_coords"],
        z=reference["atom_types"],
        num_atoms=reference["num_atoms"],
    )
    receiver, sender = reference["edge_index"]
    graph = Graph.from_edges(
        structures,
        structures.num_atoms,
        torch.stack([sender, receiver]),
        reference["to_jimages"],
        edge_batch=structures.batch[sender],
    )
    structures = CDVAE.transform()(replace(structures, graph=graph))
    assert structures.line_graph is not None

    pairs = reference.call("pairs")
    upstream = torch.stack([pairs.output("6"), pairs.output("5")], dim=1)

    return structures, permutation(upstream, structures.line_graph.index.T)


@pytest.fixture(scope="module")
def recorded() -> Recorded:
    encoder = pretrained(CDVAE, "cdvae", "cdvae-mp20-cowboys").encoder
    reference = load_reference("layers/cdvae_dimenet.npz")

    return Recorded(encoder, reference, *on_upstream_graph(reference))


def sbf(recorded: Recorded) -> torch.Tensor:
    """Basis of each pair of bonds cdvae computed, in torchms order."""
    return recorded.in_torchms_order(recorded.reference.call("sbf").output())


def test_graph(recorded):
    """The pairs of AddLineGraph are the ones of cdvae, with the same angles."""
    graph, pairs = recorded.structures.graph, recorded.structures.line_graph
    assert graph is not None and pairs is not None

    # the lengths of the edges from the coordinates and the shifts of cdvae
    distance = recorded.reference.call("rbf").input("dist")
    assert_close(graph.edges_distances(), distance, name="lengths of the bonds")

    cosines = triplet_cosines(graph.edges_vectors(), pairs)
    assert_close(cosines, recorded.cosines(), name="cosines of the pairs")


def test_bessel_basis_layer(recorded):
    call = recorded.reference.call("rbf")
    assert_close(recorded.encoder.rbf(call.input("dist")), call.output(), name="rbf")


def test_spherical_basis_layer(recorded):
    """The radial part of the second bond of a pair, times the harmonics of its angle.

    The spherical Bessel functions lose digits in float32 for short bonds, in
    both codes: under 2 Å, cdvae is up to 3e-3 from the values in float64
    (on values up to 33), torchms 9e-4. The pairs whose second bond is longer
    are compared at 1e-5; the other ones are as near to float64 as cdvae.
    """
    pairs = recorded.structures.line_graph
    assert pairs is not None
    distance = recorded.reference.call("sbf").input("dist")

    def basis(sbf, distance, cosine):
        radial = pairs.unpool_edges(sbf.radial(distance), position=1)
        return sbf(radial, cosine)

    found = basis(recorded.encoder.sbf, distance, recorded.cosines())
    expected = sbf(recorded)

    long = pairs.unpool_edges(distance, position=1) >= SHORT_BOND
    assert long.sum() > 100 and (~long).sum() > 10
    assert_close(found[long], expected[long], name="sbf, bonds of 2 Å or more")

    exact = basis(
        copy.deepcopy(recorded.encoder.sbf).double(),
        distance.double(),
        recorded.cosines().double(),
    )
    ours = (found[~long].double() - exact[~long]).abs().max()
    theirs = (expected[~long].double() - exact[~long]).abs().max()
    assert ours <= theirs, f"short bonds: {ours:.1e} from float64, cdvae {theirs:.1e}"


def test_embedding_block(recorded):
    call = recorded.reference.call("emb")
    found = recorded.encoder.emb(
        call.input("x"), call.input("rbf"), recorded.structures.graph
    )
    assert_close(found, call.output(), name="embedding of the bonds")


@pytest.mark.parametrize("block", [0, 3])
def test_interaction_block(recorded, block):
    call = recorded.reference.call(f"interaction_blocks.{block}")
    pairs = recorded.structures.line_graph
    assert pairs is not None

    found = recorded.encoder.interaction_blocks[block](
        call.input("x"), call.input("rbf"), sbf(recorded), pairs
    )
    assert_close(found, call.output(), name=f"interaction block {block}")


@pytest.mark.parametrize("block", [0, 4])
def test_output_block(recorded, block):
    call = recorded.reference.call(f"output_blocks.{block}")
    found = recorded.encoder.output_blocks[block](
        call.input("x"), call.input("rbf"), recorded.structures.graph
    )
    assert_close(found, call.output(), name=f"output block {block}")


def test_network(recorded):
    """DimeNet++ as a whole: 4 blocks, on the CrystalNN graph of cdvae."""
    graph, pairs = recorded.structures.graph, recorded.structures.line_graph
    assert graph is not None and pairs is not None
    distance = recorded.reference.call("rbf").input("dist")

    found = recorded.encoder(
        recorded.structures.z, graph, pairs, distance, recorded.cosines()
    )
    assert_close(found, recorded.reference.call("network").output(), name="hidden")
