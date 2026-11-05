"""Graphs of the original codes which are not defined, to leave them out of a test."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from torchms.data import Structures
from torchms.graph.sparse import count_by
from torchms.graph.transforms import AddRadiusGraph, KeepNearestNeighbors


def neighbours_at_the_same_distance(
    structures: Structures,
    near: Callable[[Structures], Structures],
    max_neighbors: int,
) -> torch.Tensor:
    """Structures where the choice of the nearest neighbours of an atom is not defined.

    The last neighbour kept and the first one left out are at the same
    distance: in a crystal several are, and the two images of an atom on each
    side of it always are. The original keeps one or the other from the
    rounding of their distances and the order of its sort.

    Args:
        structures: Structures, with or without graph.
        near: The transform of the model which attaches the atoms closer than
            its radius, before the limit of neighbours.
        max_neighbors: The limit of neighbours of the model.

    Returns:
        Mask of these structures, shape (n_struct,).

    Examples:
        In a cubic cell, the 6 nearest images of an atom are at the same
        distance: a limit of 4 neighbours is not defined, one of 6 is.

        >>> cubic = Structures(lattice=torch.eye(3)[None] * 3.0, x=[[0.0] * 3], z=[6])
        >>> near = AddRadiusGraph(4.0, self_images=True)
        >>> neighbours_at_the_same_distance(cubic, near, 4).tolist()
        [True]
        >>> neighbours_at_the_same_distance(cubic, near, 6).tolist()
        [False]
    """
    graph = KeepNearestNeighbors(max_neighbors + 1)(
        near(structures.without_graphs())
    ).graph
    assert graph is not None

    # edges are sorted by atom then by length: the last two of each atom
    degree = count_by(graph.senders, len(structures.z))
    last = F.pad(torch.cumsum(degree, dim=0), (1, 0), value=0)[1:] - 1
    full = degree == max_neighbors + 1
    distance = graph.edges_distances()
    tie = (distance[last[full]] - distance[last[full] - 1]).abs() < 1e-4

    return count_by(structures.batch[full][tie], len(structures.num_atoms)) > 0
