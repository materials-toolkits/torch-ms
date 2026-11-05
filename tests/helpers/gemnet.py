"""The graph GemNet-T of an original code built, as torchms structures.

MatterGen and cdvae record the graph of GemNet-T with
generate_interaction_graph: edge_index (idx_s, idx_t), the number of edges per
structure, the length D_st and the direction V_st of each edge, id_swap and the
triplets (id3_ba, id3_ca, id3_ragged_idx); MatterGen also the cell offsets.
The layer tests run torchms on the same edges in the same order, with the same
vectors: per-edge features compare row by row, and the geometry is the same to
the bit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from helpers.matching import permutation
from helpers.references import Reference
from torchms.data.batching import Structures
from torchms.graph.graph import Graph
from torchms.graph.transforms import AddBonds, AddTriplets


@dataclass(frozen=True)
class UpstreamGraph:
    """Structures on the graph of the original, and the order of its triplets.

    Attributes:
        structures: The recorded crystals with the graph of the original, its
            triplets and its bonds, as torchms builds them from its edges.
        triplet_order: Index of the triplet of the original for each triplet of
            torchms: a per-triplet tensor of the original, indexed by it, is in
            the order of structures.triplets.
        id_swap: Reverse of each edge, as the original lists it.
    """

    structures: Structures
    triplet_order: torch.Tensor
    id_swap: torch.Tensor


def on_upstream_graph(reference: Reference, network: str = "network") -> UpstreamGraph:
    """Structures with the recorded graph of the original GemNet-T.

    The original lists an edge c→a under the atom a, its target idx_t, which is
    the sender of the torchms edge; the receiver is idx_s. The direction V_st
    points from c to a, against the vector of the torchms edge. The shift of
    each edge is the image of its receiver which the vector reaches: it is
    checked against the positions.

    Args:
        reference: Recorded "graph" (generate_interaction_graph) and inputs of
            the whole network, by default under "network": its lattice, or the
            one saved as "lattice" when the network reads lengths and angles.
        network: Name of the whole network in the reference.

    Raises:
        AssertionError: If the vectors do not join the recorded atoms, or the
            triplets of torchms are not the ones of the original.
    """
    call = reference.call(network)
    # cdvae gives GemNet-T lengths and angles: the script saves the lattice
    lattice = call.inputs.get("lattice", reference.arrays.get("lattice"))
    assert lattice is not None, "no lattice recorded"
    x = call.input("frac_coords")
    structures = Structures(
        lattice=lattice,
        x=x,
        z=call.input("atom_types"),
        num_atoms=call.input("num_atoms"),
    )

    graph_out = reference.call("graph")
    idx_s, idx_t = graph_out.output("0")
    distance, direction = graph_out.output("2"), graph_out.output("3")
    sender, receiver = idx_t, idx_s
    vectors = -direction * distance.unsqueeze(-1)

    edge_batch = structures.batch[sender]
    assert torch.equal(edge_batch, structures.batch[receiver])
    assert bool((edge_batch[1:] >= edge_batch[:-1]).all()), "edges by structure"

    graph = Graph.from_edges(
        structures,
        structures.num_atoms,
        torch.stack([sender, receiver]),
        receiver_images(structures, sender, receiver, vectors),
        edge_batch=edge_batch,
        edge_vector=vectors,
        edge_distance=distance,
    )
    structures = AddBonds()(AddTriplets()(replace(structures, graph=graph)))
    assert structures.triplets is not None

    # a torchms triplet gives its second edge to its first one: (id3_ca, id3_ba)
    upstream = torch.stack([graph_out.output("6"), graph_out.output("5")], dim=1)
    triplet_order = permutation(upstream, structures.triplets.index.T)

    return UpstreamGraph(structures, triplet_order, graph_out.output("4"))


def receiver_images(
    structures: Structures,
    sender: torch.Tensor,
    receiver: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Image of the receiver of each edge which its vector reaches, shape (n_edges, 3).

    Raises:
        AssertionError: If a vector does not join its atoms, to 1e-4 of a cell.
    """
    edge_batch = structures.batch[sender]
    # pylint: disable-next=not-callable
    inverse = torch.linalg.inv(structures.lattice.double())[edge_batch]
    fractional = torch.einsum("ei,eij->ej", vectors.double(), inverse)
    offset = fractional - (structures.x[receiver] - structures.x[sender]).double()
    shift = torch.round(offset)
    assert float((offset - shift).abs().max()) < 1e-4, "vectors join the atoms"

    return shift.long()
