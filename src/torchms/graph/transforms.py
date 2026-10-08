"""Graph construction as transforms of batched crystal structures.

Each transform is a callable from Structures to Structures, to give as the
transform of a dataset, alone or chained with Compose after the augmentations.
What it builds is attached to the structures it returns:

* Structures.graph, a Graph: the edges between the atoms;
* Structures.line_graph, Structures.triplets and Structures.quadruplets,
  EdgeTuples: pairs and chains of edges of this graph.

Add transforms attach a field, Keep transforms remove edges of the graph. The
structures given are not modified.

Examples:
    >>> from torchms.data.augmentation import Compose
    >>> transform = Compose(
    ...     [AddRadiusGraph(4.0), KeepNearestNeighbors(4), AddTriplets(geometry=True)]
    ... )
    >>> structures = transform(Structures.diamond())
    >>> structures.graph.num_edges, structures.triplets.num_tuples
    (tensor([32]), tensor([96]))
    >>> structures.triplets.angle.rad2deg().unique().round(decimals=2)
    tensor([109.4700])
"""

from dataclasses import dataclass, replace

import torch
from jaxtyping import Bool, Float

from torchms.data.batching import Structures
from torchms.graph.atomic_radius import atomic_radius
from torchms.graph.cutoff import cutoff, nearest_neighbors
from torchms.graph.geometry import dihedral_angles, triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.line_graph import edge_pairs, quadruplets, triplets


def attached_graph(structures: Structures, transform: object) -> Graph:
    """Graph attached to the structures.

    Raises:
        ValueError: If the structures have no graph.
    """
    if structures.graph is None:
        raise ValueError(
            f"{type(transform).__name__} needs a graph: add one before, "
            "with AddRadiusGraph or AddAtomicRadiusGraph"
        )

    return structures.graph


def attach_graph(structures: Structures, graph: Graph, geometry: bool) -> Structures:
    """Structures with a new graph, without the tuples of edges of the old one."""
    if geometry and graph.edge_vector is None:
        graph = graph.with_geometry()

    return replace(structures.without_graphs(), graph=graph)


def edges_vectors(graph: Graph) -> Float[torch.Tensor, "n_edges 3"]:
    """Vector of each edge in Å, the ones of the graph if it has them."""
    if graph.edge_vector is not None:
        return graph.edge_vector

    return graph.edges_vectors()


def shorter_edges(
    graph: Graph, radius: float | None
) -> Bool[torch.Tensor, "n_edges"] | None:
    """Mask of the edges not longer than a radius, None without radius."""
    if radius is None:
        return None

    if graph.edge_distance is not None:
        return graph.edge_distance <= radius

    return graph.edges_distances() <= radius


def edge_tuples(
    graph: Graph,
    index: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    angle: torch.Tensor | None,
) -> EdgeTuples:
    """Tuples of edges of a graph from their indices, as returned by triplets."""
    return EdgeTuples(
        index=index,
        num_edges=graph.num_edges,
        num_tuples=ptr[1:] - ptr[:-1],
        batch=batch,
        ptr=ptr,
        angle=angle,
        _validate=True,
    )


@dataclass
class AddRadiusGraph:
    """Attach the graph of the atoms closer than a radius.

    An edge goes from each atom to every atom, of the cell or of its periodic
    images, not farther than the radius. Every edge has its reverse. See
    torchms.graph.cutoff.cutoff.

    Fields:
        Sets structures.graph. Removes structures.line_graph, triplets and
        quadruplets, which were tuples of edges of the graph replaced.

        * graph.edge_index, (2, n_edges): sender and receiver atoms.
        * graph.edge_shift, (n_edges, 3): periodic image of the receiver.
        * graph.num_atoms, graph.num_edges, (n_struct,).
        * graph.edge_batch, (n_edges,), and graph.edge_ptr, (n_struct + 1,).
        * graph.edge_vector, (n_edges, 3), and graph.edge_distance, (n_edges,),
          in Å: only with geometry=True, None otherwise.

    Args:
        radius: Cutoff radius in Å.
        self_images: If True, an atom is connected to its own periodic images
            closer than the radius.
        geometry: If True, the vector and the length of each edge are attached.

    Examples:
        >>> structures = AddRadiusGraph(3.0, geometry=True)(Structures.zincblende())
        >>> structures.graph.edge_index.shape
        torch.Size([2, 32])
        >>> structures.graph.edge_distance.unique().round(decimals=2)
        tensor([2.3400])
        >>> AddRadiusGraph(3.0)(Structures.zincblende()).graph.edge_distance is None
        True
    """

    radius: float
    self_images: bool = False
    geometry: bool = False

    def __call__(self, structures: Structures) -> Structures:
        structures = structures.without_graphs()
        graph = cutoff(structures, self.radius, self.self_images)

        return attach_graph(structures, graph, self.geometry)


@dataclass
class AddAtomicRadiusGraph:
    """Attach the graph of the chemical bonds, from the atomic radii.

    Two atoms are connected when their distance is not larger than scaling
    times the sum of their atomic radii. Every edge has its reverse. See
    torchms.graph.atomic_radius.atomic_radius.

    Fields:
        Sets structures.graph, with the same fields as AddRadiusGraph. Removes
        structures.line_graph, triplets and quadruplets.

    Args:
        scaling: Multiple of the sum of the atomic radii.
        ensure_gap: If positive, the edges after the first gap of this size in
            Å, in the distances of a structure, are dropped.
        geometry: If True, the vector and the length of each edge are attached.

    Examples:
        >>> structures = AddAtomicRadiusGraph(1.1)(Structures.zincblende())
        >>> structures.graph.num_edges
        tensor([32])
    """

    scaling: float = 2.0
    ensure_gap: float = 0.5
    geometry: bool = False

    def __call__(self, structures: Structures) -> Structures:
        structures = structures.without_graphs()
        graph = atomic_radius(structures, self.scaling, self.ensure_gap)

        return attach_graph(structures, graph, self.geometry)


@dataclass
class KeepNearestNeighbors:
    """Keep the k shortest edges starting from each atom.

    The result is not symmetric: an edge can be kept without its reverse. See
    torchms.graph.cutoff.nearest_neighbors.

    Fields:
        Replaces structures.graph: fewer edges, sorted by sender atom then by
        length. The geometry of the edges kept is kept. Removes
        structures.line_graph, triplets and quadruplets, which pointed to the
        edges of the graph replaced: add them after this step.

    Args:
        k: Maximum number of edges per sender atom.

    Examples:
        >>> structures = AddRadiusGraph(4.0)(Structures.diamond())
        >>> structures.graph.num_edges
        tensor([320])
        >>> KeepNearestNeighbors(4)(structures).graph.num_edges
        tensor([32])
    """

    k: int

    def __call__(self, structures: Structures) -> Structures:
        graph = nearest_neighbors(attached_graph(structures, self), self.k)

        return attach_graph(structures, graph, False)


@dataclass
class KeepRadius:
    """Keep the edges not longer than a radius.

    Fields:
        Replaces structures.graph: fewer edges, in the same order. The geometry
        of the edges kept is kept. Removes structures.line_graph, triplets and
        quadruplets, which pointed to the edges of the graph replaced: add
        them after this step.

    Args:
        radius: Maximum length of an edge in Å.

    Examples:
        >>> structures = AddRadiusGraph(4.0)(Structures.diamond())
        >>> KeepRadius(2.0)(structures).graph.num_edges
        tensor([32])
    """

    radius: float

    def __call__(self, structures: Structures) -> Structures:
        graph = attached_graph(structures, self)
        mask = shorter_edges(graph, self.radius)
        assert mask is not None

        return attach_graph(
            structures, graph.select_edges(mask.nonzero().squeeze(1)), False
        )


@dataclass
class AddLineGraph:
    """Attach the line graph: pairs of edges u→v then v→w.

    The nodes of the line graph are the edges of the graph, its edges the pairs
    of edges where the first one arrives at the atom the second one leaves. See
    torchms.graph.line_graph.edge_pairs.

    Fields:
        Sets structures.line_graph, an EdgeTuples. The graph is not changed.

        * line_graph.index, (2, n_pairs): row 0 is the edge u→v, row 1 the
          edge v→w, indices of edges of structures.graph in the batch. Sorted
          by the atom v.
        * line_graph.num_edges, (n_struct,): edges of the graph per structure.
        * line_graph.num_tuples, (n_struct,): pairs per structure.
        * line_graph.batch, (n_pairs,), and line_graph.ptr, (n_struct + 1,).
        * line_graph.angle, (n_pairs,), in radians, in [0, pi]: angle at the
          atom v between u and w. Only with geometry=True, None otherwise.

    Args:
        radius: Only edges not longer than this radius, in Å, are in a pair.
            All the edges by default. The indices still point to the edges of
            the whole graph.
        backtrack: If False, an edge is not followed by its reverse: the pair
            u→v, v→u is not listed.
        geometry: If True, the angle of each pair is attached.

    Examples:
        >>> structures = AddRadiusGraph(2.0)(Structures.diamond())
        >>> structures = AddLineGraph(geometry=True)(structures)
        >>> structures.line_graph.index.shape
        torch.Size([2, 96])
        >>> structures.line_graph.angle.rad2deg().unique().round(decimals=2)
        tensor([109.4700])
    """

    radius: float | None = None
    backtrack: bool = False
    geometry: bool = False

    def __call__(self, structures: Structures) -> Structures:
        graph = attached_graph(structures, self)
        index, batch, ptr = edge_pairs(
            graph, shorter_edges(graph, self.radius), self.backtrack
        )

        angle = None
        if self.geometry:
            # the first edge arrives at v, the second one leaves it
            angle = torch.acos(-triplet_cosines(edges_vectors(graph), index))

        return replace(
            structures, line_graph=edge_tuples(graph, index, batch, ptr, angle)
        )


@dataclass
class AddTriplets:
    """Attach the triplets of atoms: pairs of edges i→j, i→k from the same atom.

    A triplet is the angle at the atom i between two of its neighbours. Both
    orders of the two edges are listed: an atom with d edges has d (d - 1)
    triplets. See torchms.graph.line_graph.triplets.

    Fields:
        Sets structures.triplets, an EdgeTuples. The graph is not changed.

        * triplets.index, (2, n_triplets): row 0 is the edge i→j, row 1 the
          edge i→k, indices of edges of structures.graph in the batch. Sorted
          by the atom i.
        * triplets.num_edges, (n_struct,): edges of the graph per structure.
        * triplets.num_tuples, (n_struct,): triplets per structure.
        * triplets.batch, (n_triplets,), and triplets.ptr, (n_struct + 1,).
        * triplets.angle, (n_triplets,), in radians, in [0, pi]: angle at the
          atom i between j and k. Only with geometry=True, None otherwise.

    Args:
        radius: Only edges not longer than this radius, in Å, are in a triplet.
            All the edges by default. The indices still point to the edges of
            the whole graph.
        geometry: If True, the angle of each triplet is attached.

    Examples:
        Each atom of diamond has 4 neighbours, 4 × 3 triplets.

        >>> structures = AddRadiusGraph(2.0)(Structures.diamond())
        >>> structures = AddTriplets(geometry=True)(structures)
        >>> structures.triplets.index.shape, structures.triplets.ptr
        (torch.Size([2, 96]), tensor([ 0, 96]))
        >>> structures.triplets.angle.rad2deg().unique().round(decimals=2)
        tensor([109.4700])
    """

    radius: float | None = None
    geometry: bool = False

    def __call__(self, structures: Structures) -> Structures:
        graph = attached_graph(structures, self)
        index, batch, ptr = triplets(graph, shorter_edges(graph, self.radius))

        angle = None
        if self.geometry:
            angle = torch.acos(triplet_cosines(edges_vectors(graph), index))

        return replace(
            structures, triplets=edge_tuples(graph, index, batch, ptr, angle)
        )


@dataclass
class AddQuadruplets:
    """Attach the quadruplets of atoms: chains of three edges u→v, v→w, w→x.

    A quadruplet is the dihedral angle around the bond v-w. A chain never goes
    back along an edge. Their number grows with the cube of the number of
    neighbours: use a short radius. See torchms.graph.line_graph.quadruplets.

    Fields:
        Sets structures.quadruplets, an EdgeTuples. The graph is not changed.

        * quadruplets.index, (3, n_quadruplets): rows are the edges u→v, v→w
          and w→x, indices of edges of structures.graph in the batch.
        * quadruplets.num_edges, (n_struct,): edges of the graph per structure.
        * quadruplets.num_tuples, (n_struct,): quadruplets per structure.
        * quadruplets.batch, (n_quadruplets,), and quadruplets.ptr,
          (n_struct + 1,).
        * quadruplets.angle, (n_quadruplets,), in radians, in [-pi, pi]:
          dihedral angle between the planes (u, v, w) and (v, w, x), 0 when u
          and x are on the same side (cis), pi on opposite sides (trans). Its
          sign changes in the mirror image. Only with geometry=True, None
          otherwise.

    Args:
        radius: Only edges not longer than this radius, in Å, are in a chain.
            All the edges by default. The indices still point to the edges of
            the whole graph.
        geometry: If True, the dihedral angle of each quadruplet is attached.

    Examples:
        In diamond, each of the 4 neighbours of an atom has 3 other neighbours,
        which have 3 other neighbours: 8 × 4 × 3 × 3 chains, staggered.

        >>> structures = AddRadiusGraph(2.0)(Structures.diamond())
        >>> structures = AddQuadruplets(geometry=True)(structures)
        >>> structures.quadruplets.index.shape
        torch.Size([3, 288])
        >>> structures.quadruplets.angle.abs().rad2deg().unique().round()
        tensor([ 60., 180.])
    """

    radius: float | None = None
    geometry: bool = False

    def __call__(self, structures: Structures) -> Structures:
        graph = attached_graph(structures, self)
        index, batch, ptr = quadruplets(graph, shorter_edges(graph, self.radius))

        angle = None
        if self.geometry:
            angle = dihedral_angles(edges_vectors(graph), index)

        return replace(
            structures, quadruplets=edge_tuples(graph, index, batch, ptr, angle)
        )


__all__ = [
    "AddAtomicRadiusGraph",
    "AddLineGraph",
    "AddQuadruplets",
    "AddRadiusGraph",
    "AddTriplets",
    "KeepNearestNeighbors",
    "KeepRadius",
]
