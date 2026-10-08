"""Graph dataclass for batched crystal structure graphs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from torchms.data.batching import Structures, TensorLike
from torchms.graph.geometry import cartesian
from torchms.graph.sparse import Reduce, count_by, scatter


class InvalidGraph(Exception):
    """Exception raised when a graph has invalid data."""


class InvalidStructures(Exception):
    """Exception raised when references on structures is not a Structures object"""


def cat_optional(
    tensors: Sequence[torch.Tensor | None], field: str
) -> torch.Tensor | None:
    """Concatenate an optional field of the elements of a batch.

    Returns:
        The concatenation, or None if no element has the field.

    Raises:
        ValueError: If only some of the elements have the field.
    """
    present = [tensor for tensor in tensors if tensor is not None]

    if len(present) == 0:
        return None

    if len(present) != len(tensors):
        raise ValueError(
            f'cannot collate: only some of the elements have the field "{field}"'
        )

    return torch.cat(present, dim=0)


@dataclass(init=False)
class Graph:
    """Container for batched crystal structure graphs.

    Mirrors the batching convention of Structures: all per-edge data is stored
    in flat concatenated tensors, with edge_ptr and edge_batch for indexing.

    An edge goes from its sender, an atom of the cell, to its receiver, an atom
    of the cell or of one of its periodic images. Atoms are numbered in the
    whole batch, as the rows of Structures.x. Edges are ordered by structure.

    Attributes:
        edge_index: Atoms of each edge, shape (2, n_edges), int64. Row 0 is the
            sender, row 1 the receiver. Indices of atoms in the whole batch:
            in a batch, the atoms of a structure are shifted by the number of
            atoms of the structures before it (Structures.ptr).
        edge_shift: Periodic image of the receiver of each edge, shape
            (n_edges, 3), int64, in lattice vectors. The edge goes from
            x[sender] to x[receiver] + edge_shift, in fractional coordinates.
            Zero for an edge inside the cell.
        num_atoms: Number of atoms per structure, shape (n_struct,), int64.
        num_edges: Number of edges per structure, shape (n_struct,), int64.
        edge_batch: Structure index of each edge, in ascending order, shape
            (n_edges,), int64.
        edge_ptr: Index of the first edge of each structure, with the number of
            edges at the end, shape (n_struct + 1,), int64. The edges of the
            structure i are edge_ptr[i]:edge_ptr[i + 1].
        edge_vector: Cartesian vector of each edge in Å, from its sender to its
            receiver, shape (n_edges, 3), float. None unless computed, see
            with_geometry.
        edge_distance: Length of each edge in Å, the norm of edge_vector, shape
            (n_edges,), float. None unless computed, see with_geometry.
        structures: Structures of the graph, needed for the geometry of the
            edges. When the graph is attached to structures
            (Structures.graph), they share the tensors of these structures
            and have nothing attached.

    Examples:
        The 4 bonds of each of the 8 atoms of diamond.

        >>> from torchms.graph.cutoff import cutoff
        >>> graph = cutoff(Structures.diamond(), 2.0).with_geometry()
        >>> graph.edge_index.shape, graph.edge_shift.shape
        (torch.Size([2, 32]), torch.Size([32, 3]))
        >>> graph.num_atoms, graph.num_edges, graph.edge_ptr
        (tensor([8]), tensor([32]), tensor([ 0, 32]))
        >>> graph.edge_vector.shape, graph.edge_distance.unique().round(decimals=2)
        (torch.Size([32, 3]), tensor([1.5400]))
    """

    edge_index: Int[torch.Tensor, "2 n_edges"]
    edge_shift: Int[torch.Tensor, "n_edges 3"]
    num_atoms: Int[torch.Tensor, "n_struct"]
    num_edges: Int[torch.Tensor, "n_struct"]
    edge_batch: Int[torch.Tensor, "n_edges"]
    edge_ptr: Int[torch.Tensor, "n_struct+1"]
    structures: Structures | None
    edge_vector: Float[torch.Tensor, "n_edges 3"] | None
    edge_distance: Float[torch.Tensor, "n_edges"] | None
    _validate: bool

    def __init__(
        self,
        edge_index: TensorLike,
        edge_shift: TensorLike,
        num_atoms: TensorLike,
        num_edges: TensorLike,
        edge_batch: TensorLike | None = None,
        edge_ptr: TensorLike | None = None,
        structures: Structures | None = None,
        edge_vector: torch.Tensor | None = None,
        edge_distance: torch.Tensor | None = None,
        _validate: bool = False,
    ):
        self.edge_index = self.as_tensor("edge_index", edge_index)
        self.edge_shift = self.as_tensor("edge_shift", edge_shift)
        self.num_atoms = self.as_tensor("num_atoms", num_atoms)
        self.num_edges = self.as_tensor("num_edges", num_edges)

        # Missing fields are derived from num_edges, they are checked with
        # the given ones in check_tensors.
        if edge_batch is None:
            counts = self.num_edges.flatten().long().clamp(min=0)
            self.edge_batch = torch.arange(
                len(counts), dtype=torch.int64, device=counts.device
            ).repeat_interleave(counts)
        else:
            self.edge_batch = self.as_tensor("edge_batch", edge_batch)

        if edge_ptr is None:
            counts = self.num_edges.flatten().long().clamp(min=0)
            self.edge_ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)
        else:
            self.edge_ptr = self.as_tensor("edge_ptr", edge_ptr)

        self.structures = structures
        self.edge_vector = edge_vector
        self.edge_distance = edge_distance

        self._validate = _validate
        if self._validate:
            return

        self.check_tensors()

        self._validate = True

    @classmethod
    def as_tensor(cls, field: str, values: TensorLike) -> torch.Tensor:
        """Convert the values of a field to a tensor, tensors are left unchanged.

        Raises:
            InvalidGraph: If values cannot be converted to a tensor.
        """
        if isinstance(values, torch.Tensor):
            return values

        try:
            return torch.tensor(values, dtype=torch.int64)
        except (TypeError, ValueError) as e:
            raise InvalidGraph(f"invalid field {field}: {e}") from e

    def check_tensors(self):
        """Validate graph data and cast fields to their dtype.

        Raises:
            InvalidGraph: If edge or structure data is invalid.
        """

        if (self.structures is not None) and (
            not isinstance(self.structures, Structures)
        ):
            raise InvalidStructures(
                "reference to structures must be a Structures object"
            )

        device = self.edge_index.device

        n_struct = len(self.num_atoms)

        if n_struct < 1:
            raise InvalidGraph("must contain at least one structure")

        n_edges = int(self.num_edges.sum())

        self.to_tensor(
            "edge_index", dtype=torch.int64, shape=(2, n_edges), device=device
        )
        self.to_tensor(
            "edge_shift", dtype=torch.int64, shape=(n_edges, 3), device=device
        )
        self.to_tensor("num_atoms", dtype=torch.int64, shape=(n_struct,), device=device)
        self.to_tensor("num_edges", dtype=torch.int64, shape=(n_struct,), device=device)

        if (self.num_atoms <= 0).any():
            raise InvalidGraph("num_atoms must be positive for every structure")

        self.to_tensor("edge_batch", dtype=torch.int64, shape=(n_edges,), device=device)
        if n_edges > 0 and (self.edge_batch[:-1] > self.edge_batch[1:]).any():
            raise InvalidGraph("edges must be ordered by structure in edge_batch")

        self.to_tensor(
            "edge_ptr", dtype=torch.int64, shape=(n_struct + 1,), device=device
        )

        for field, shape in (
            ("edge_vector", (n_edges, 3)),
            ("edge_distance", (n_edges,)),
        ):
            tensor = getattr(self, field)

            if tensor is None:
                continue

            if tensor.shape != shape:
                raise InvalidGraph(
                    f'invalid shape {tensor.shape} for field "{field}", must be {shape}'
                )

            if tensor.device != device:
                raise InvalidGraph(
                    f'inconsistent device for field "{field}", '
                    f"{device} != {tensor.device}"
                )

    def device(self) -> torch.device:
        return self.edge_index.device

    def to(self, device: torch.device | str) -> Graph:
        """Copy of the graph with all its tensors, and its structures, on a device.

        Args:
            device: Device of the copy, "cuda" for instance.

        Returns:
            Graph on the device. Tensors already on it are shared with the
            original ones.

        Examples:
            >>> from torchms.graph.cutoff import cutoff
            >>> graph = cutoff(Structures.diamond(), 2.0).to("cpu")
            >>> graph.device()
            device(type='cpu')
        """
        structures = None if self.structures is None else self.structures.to(device)

        return self.moved(device, structures)

    def moved(self, device: torch.device | str, structures: Structures | None) -> Graph:
        """Copy of the graph on a device, with structures already on this device.

        Used by Structures.to, which moves the structures once and shares them
        with the graph. See to.
        """
        tensors = {
            name: value.to(device)
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }

        return replace(self, structures=structures, **tensors)

    def with_geometry(self) -> Graph:
        """Copy of the graph with the vector and the length of its edges.

        Returns:
            Graph with edge_vector, shape (n_edges, 3), and edge_distance, shape
            (n_edges,), in Å, computed from its structures.

        Examples:
            >>> from torchms.graph.cutoff import cutoff
            >>> graph = cutoff(Structures.zincblende(), 3.0)
            >>> graph.edge_distance is None
            True
            >>> graph.with_geometry().edge_distance.unique().round(decimals=2)
            tensor([2.3400])
        """
        vectors = self.edges_vectors()

        return replace(self, edge_vector=vectors, edge_distance=vectors.norm(dim=-1))

    def select_edges(self, edges: Int[torch.Tensor, "n_selected"]) -> Graph:
        """Graph with some of the edges, on the same atoms.

        Args:
            edges: Indices of the edges to keep, in their order in the result.
                They must stay ordered by structure. Shape (n_selected,).

        Returns:
            Graph with these edges, and their geometry if the graph has it.

        Examples:
            >>> from torchms.graph.cutoff import cutoff
            >>> graph = cutoff(Structures.diamond(), 2.0)
            >>> graph.select_edges(torch.tensor([0, 1, 5])).num_edges
            tensor([3])
        """
        edge_batch = self.edge_batch[edges]

        return Graph.from_edges(
            self.structures,
            self.num_atoms,
            self.edge_index[:, edges],
            self.edge_shift[edges],
            edge_batch=edge_batch,
            edge_vector=None if self.edge_vector is None else self.edge_vector[edges],
            edge_distance=(
                None if self.edge_distance is None else self.edge_distance[edges]
            ),
        )

    @classmethod
    def from_edges(
        cls,
        structures: Structures | None,
        num_atoms: Int[torch.Tensor, "n_struct"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
        edge_shift: Int[torch.Tensor, "n_edges 3"],
        *,
        edge_batch: Int[torch.Tensor, "n_edges"],
        edge_vector: Float[torch.Tensor, "n_edges 3"] | None = None,
        edge_distance: Float[torch.Tensor, "n_edges"] | None = None,
    ) -> Graph:
        """Graph from edges ordered by structure, with the structure of each edge.

        For the functions which build graphs: the edges are counted from
        edge_batch and nothing is checked, which on a GPU would make the CPU
        wait for it several times.

        Args:
            structures: Structures of the graph.
            num_atoms: Number of atoms per structure, shape (n_struct,).
            edge_index: Sender and receiver of each edge, shape (2, n_edges).
            edge_shift: Periodic image of the receiver, shape (n_edges, 3).
            edge_batch: Structure index of each edge, in ascending order, shape
                (n_edges,).
            edge_vector: Cartesian vector of each edge, if computed.
            edge_distance: Length of each edge, if computed.

        Returns:
            The graph, not validated.
        """
        num_edges = count_by(edge_batch, len(num_atoms))

        return cls(
            structures=structures,
            edge_index=edge_index,
            edge_shift=edge_shift,
            num_atoms=num_atoms,
            num_edges=num_edges,
            edge_batch=edge_batch,
            edge_ptr=F.pad(torch.cumsum(num_edges, dim=0), (1, 0), value=0),
            edge_vector=edge_vector,
            edge_distance=edge_distance,
            _validate=True,
        )

    def edges_vectors(self) -> Float[torch.Tensor, "n_edges 3"]:
        """Cartesian vector of each edge in Å, from its sender to its receiver."""
        assert self.structures is not None

        edges_frac = (
            self.structures.x[self.edge_index[1]]
            + self.edge_shift
            - self.structures.x[self.edge_index[0]]
        )
        return cartesian(edges_frac, self.structures.lattice, self.edge_batch)

    def edges_distances(self) -> Float[torch.Tensor, "n_edges"]:
        """Length of each edge in Å."""
        return self.edges_vectors().norm(dim=-1)

    def aggregate_edges_to_nodes(
        self,
        src: Float[torch.Tensor, "n_edges ..."],
        reduce: Reduce = "sum",
    ) -> Float[torch.Tensor, "n_atoms ..."]:
        """Aggregate per-edge features to receiver nodes (message passing).

        Args:
            src: Per-edge features, shape (n_edges, *).
            reduce: Reduction passed to scatter — 'sum', 'mean', 'amax', etc.

        Returns:
            Per-node aggregated features, shape (n_atoms, *).
        """
        return scatter(src, self.edge_index[1], int(self.num_atoms.sum()), reduce)

    def aggregate_nodes_to_structures(
        self,
        src: Float[torch.Tensor, "n_atoms ..."],
        reduce: Reduce = "sum",
    ) -> Float[torch.Tensor, "n_struct ..."]:
        """Aggregate per-node features to structures (global pooling).

        Args:
            src: Per-node features, shape (n_atoms, *).
            reduce: Reduction passed to scatter — 'sum', 'mean', 'amax', etc.

        Returns:
            Per-structure aggregated features, shape (n_struct, *).
        """
        node_batch = torch.arange(
            len(self.num_atoms), device=self.device()
        ).repeat_interleave(self.num_atoms)
        return scatter(src, node_batch, len(self.num_atoms), reduce)

    def aggregate_edges_to_structures(
        self,
        src: Float[torch.Tensor, "n_edges ..."],
        reduce: Reduce = "sum",
    ) -> Float[torch.Tensor, "n_struct ..."]:
        """Aggregate per-edge features to structures (global pooling).

        Args:
            src: Per-edge features, shape (n_edges, *).
            reduce: Reduction passed to scatter — 'sum', 'mean', 'amax', etc.

        Returns:
            Per-structure aggregated features, shape (n_struct, *).
        """
        return scatter(src, self.edge_batch, len(self.num_edges), reduce)

    def to_tensor(
        self,
        field: str,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Check a field and cast it to the specified dtype.

        Raises:
            InvalidGraph: If field value is invalid or incompatible.
        """
        tensor: torch.Tensor = getattr(self, field)

        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)

        if tensor.shape != shape:
            raise InvalidGraph(
                f'invalid shape {tensor.shape} for field "{field}", must be {shape}'
            )

        if tensor.device != device:
            raise InvalidGraph(
                f'inconsistent device for field "{field}", {device} != {tensor.device}'
            )

        setattr(self, field, tensor)

    @classmethod
    def collate(
        cls, graphs: list[Graph], structures: Structures | None = None
    ) -> Graph:
        """Combine multiple graphs into a single batched graph.

        The atoms of a graph are shifted by the number of atoms of the graphs
        before it, so that edge_index points to the atoms of the batch. The
        other fields are concatenated, edge_batch and edge_ptr are rebuilt.

        Args:
            graphs: List of Graph objects to batch together, each one with one
                or several structures.
            structures: Structures of the result. By default, the structures of
                the graphs are collated if all of them have structures.

        Returns:
            A single Graph with concatenated tensors and offset edge_index. It
            has the geometry of the edges if all the graphs have it.

        Raises:
            ValueError: If graphs list is empty, or if only some of the graphs
                have the geometry of their edges.

        Examples:
            >>> from torchms.graph.cutoff import cutoff
            >>> first = cutoff(Structures.diamond(), 2.0)
            >>> second = cutoff(Structures.zincblende(), 3.0)
            >>> batched = Graph.collate([first, second])
            >>> batched.num_edges, batched.edge_ptr
            (tensor([32, 32]), tensor([ 0, 32, 64]))
            >>> int(second.edge_index.min()), int(batched.edge_index[:, 32:].min())
            (0, 8)
        """
        if len(graphs) == 0:
            raise ValueError("graphs list cannot be empty")

        # atom offset per graph: cumulative total atoms across previous graphs
        total_atoms = torch.stack([g.num_atoms.sum() for g in graphs])
        atom_offsets = F.pad(torch.cumsum(total_atoms, dim=0), (1, 0), value=0)[:-1]

        edge_index = torch.cat(
            [g.edge_index + off for g, off in zip(graphs, atom_offsets)], dim=1
        )
        edge_shift = torch.cat([g.edge_shift for g in graphs], dim=0)
        num_atoms = torch.cat([g.num_atoms for g in graphs], dim=0)
        num_edges = torch.cat([g.num_edges for g in graphs], dim=0)
        edge_ptr = F.pad(torch.cumsum(num_edges, dim=0), (1, 0), value=0)
        edge_batch = torch.arange(
            len(num_atoms), device=edge_index.device
        ).repeat_interleave(num_edges)

        if structures is None:
            references = [g.structures for g in graphs if g.structures is not None]
            if len(references) == len(graphs):
                structures = Structures.collate(
                    [reference.without_graphs() for reference in references]
                )

        return cls(
            edge_index=edge_index,
            edge_shift=edge_shift,
            num_atoms=num_atoms,
            num_edges=num_edges,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            structures=structures,
            edge_vector=cat_optional([g.edge_vector for g in graphs], "edge_vector"),
            edge_distance=cat_optional(
                [g.edge_distance for g in graphs], "edge_distance"
            ),
            _validate=True,
        )

    @classmethod
    def separate(
        cls, graph: Graph, structures: list[Structures] | None = None
    ) -> list[Graph]:
        """Split a batched graph into individual single-structure graphs.

        Args:
            graph: A batched Graph object.
            structures: Structures of each graph of the result. By default, the
                structures of the graph are separated, if it has structures.

        Returns:
            List of individual Graph objects, whose edge_index starts at 0.
        """
        n_structs = len(graph.num_atoms)
        atom_ptr = F.pad(torch.cumsum(graph.num_atoms, dim=0), (1, 0), value=0)
        result = []

        if structures is None and graph.structures is not None:
            structures = Structures.separate(graph.structures.without_graphs())

        for i in range(n_structs):
            e_start = int(graph.edge_ptr[i])
            e_end = int(graph.edge_ptr[i + 1])
            n_edges_i = e_end - e_start

            result.append(
                cls(
                    edge_index=graph.edge_index[:, e_start:e_end] - atom_ptr[i],
                    edge_shift=graph.edge_shift[e_start:e_end],
                    num_atoms=graph.num_atoms[i : i + 1],
                    num_edges=graph.num_edges[i : i + 1],
                    edge_batch=torch.zeros(
                        n_edges_i, dtype=torch.int64, device=graph.device()
                    ),
                    edge_ptr=torch.tensor(
                        [0, n_edges_i], dtype=torch.int64, device=graph.device()
                    ),
                    structures=None if structures is None else structures[i],
                    edge_vector=(
                        None
                        if graph.edge_vector is None
                        else graph.edge_vector[e_start:e_end]
                    ),
                    edge_distance=(
                        None
                        if graph.edge_distance is None
                        else graph.edge_distance[e_start:e_end]
                    ),
                    _validate=True,
                )
            )

        return result


@dataclass(init=False)
class EdgeTuples:
    """Container for batched tuples of edges of a graph: pairs or chains.

    The line graph, the triplets and the quadruplets of a graph are tuples of
    its edges. They follow the batching convention of Structures and Graph:
    flat concatenated tensors, with ptr and batch for indexing. Tuples are
    ordered by structure.

    Attributes:
        index: Edges of each tuple, shape (size, n_tuples), int64, with one row
            per edge of a tuple: size is 2 for pairs of edges, 3 for chains of
            three edges. Indices of edges in the whole batch, as the columns
            of Graph.edge_index: in a batch, the edges of a structure are
            shifted by the number of edges of the structures before it
            (Graph.edge_ptr).
        num_edges: Number of edges of the graph per structure, shape
            (n_struct,), int64. Same as Graph.num_edges: it is the size of
            what index points to, used to shift the indices in a batch.
        num_tuples: Number of tuples per structure, shape (n_struct,), int64.
        batch: Structure index of each tuple, in ascending order, shape
            (n_tuples,), int64.
        ptr: Index of the first tuple of each structure, with the number of
            tuples at the end, shape (n_struct + 1,), int64. The tuples of the
            structure i are ptr[i]:ptr[i + 1].
        angle: Angle of each tuple in radians, shape (n_tuples,), float. For a
            pair of edges, the angle between them at the atom they share, in
            [0, pi]. For a chain of three edges, its dihedral angle, in
            [-pi, pi]. None unless computed.

    Examples:
        Two pairs among three edges of one structure, then a batch of two.

        >>> pairs = EdgeTuples(
        ...     index=[[0, 0], [1, 2]], num_edges=[3], num_tuples=[2]
        ... )
        >>> pairs.batch, pairs.ptr
        (tensor([0, 0]), tensor([0, 2]))
        >>> batched = EdgeTuples.collate([pairs, pairs])
        >>> batched.index
        tensor([[0, 0, 3, 3],
                [1, 2, 4, 5]])
        >>> batched.num_tuples, batched.ptr
        (tensor([2, 2]), tensor([0, 2, 4]))
    """

    index: Int[torch.Tensor, "size n_tuples"]
    num_edges: Int[torch.Tensor, "n_struct"]
    num_tuples: Int[torch.Tensor, "n_struct"]
    batch: Int[torch.Tensor, "n_tuples"]
    ptr: Int[torch.Tensor, "n_struct+1"]
    angle: Float[torch.Tensor, "n_tuples"] | None
    _validate: bool

    def __init__(
        self,
        index: TensorLike,
        num_edges: TensorLike,
        num_tuples: TensorLike,
        *,
        batch: TensorLike | None = None,
        ptr: TensorLike | None = None,
        angle: torch.Tensor | None = None,
        _validate: bool = False,
    ):
        self.index = Graph.as_tensor("index", index)
        self.num_edges = Graph.as_tensor("num_edges", num_edges)
        self.num_tuples = Graph.as_tensor("num_tuples", num_tuples)

        # Missing fields are derived from num_tuples, they are checked with
        # the given ones in check_tensors.
        if batch is None:
            counts = self.num_tuples.flatten().long().clamp(min=0)
            self.batch = torch.arange(
                len(counts), dtype=torch.int64, device=counts.device
            ).repeat_interleave(counts)
        else:
            self.batch = Graph.as_tensor("batch", batch)

        if ptr is None:
            counts = self.num_tuples.flatten().long().clamp(min=0)
            self.ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)
        else:
            self.ptr = Graph.as_tensor("ptr", ptr)

        self.angle = angle

        self._validate = _validate
        if self._validate:
            return

        self.check_tensors()

        self._validate = True

    def check_tensors(self):
        """Validate the tuples.

        Raises:
            InvalidGraph: If a field has an invalid shape, dtype or device, or
                if an index is not an edge of its structure.
        """
        device = self.index.device
        n_struct = len(self.num_edges)
        n_tuples = int(self.num_tuples.sum())

        shapes: dict[str, tuple[int, ...]] = {
            "num_edges": (n_struct,),
            "num_tuples": (n_struct,),
            "batch": (n_tuples,),
            "ptr": (n_struct + 1,),
        }

        if self.index.dim() != 2 or self.index.shape[1] != n_tuples:
            raise InvalidGraph(
                f'invalid shape {self.index.shape} for field "index", '
                f"must be (size, {n_tuples})"
            )

        for field in ("index", "num_edges", "num_tuples", "batch", "ptr"):
            tensor: torch.Tensor = getattr(self, field)

            if tensor.dtype != torch.int64:
                raise InvalidGraph(f'field "{field}" must be int64, not {tensor.dtype}')

            if field in shapes and tensor.shape != shapes[field]:
                raise InvalidGraph(
                    f'invalid shape {tensor.shape} for field "{field}", '
                    f"must be {shapes[field]}"
                )

            if tensor.device != device:
                raise InvalidGraph(
                    f'inconsistent device for field "{field}", '
                    f"{device} != {tensor.device}"
                )

        if n_tuples > 0 and (self.batch[:-1] > self.batch[1:]).any():
            raise InvalidGraph("tuples must be ordered by structure in batch")

        # each edge of a tuple is an edge of the structure of the tuple
        edge_ptr = F.pad(torch.cumsum(self.num_edges, dim=0), (1, 0), value=0)
        first, last = edge_ptr[self.batch], edge_ptr[self.batch + 1]
        if ((self.index < first) | (self.index >= last)).any():
            raise InvalidGraph("a tuple has an edge which is not in its structure")

        if self.angle is not None:
            if self.angle.shape != (n_tuples,):
                raise InvalidGraph(
                    f'invalid shape {self.angle.shape} for field "angle", '
                    f"must be {(n_tuples,)}"
                )

            if self.angle.device != device:
                raise InvalidGraph(
                    'inconsistent device for field "angle", '
                    f"{device} != {self.angle.device}"
                )

    def device(self) -> torch.device:
        """Device on which the tensors are placed."""
        return self.index.device

    def to(self, device: torch.device | str) -> EdgeTuples:
        """Copy of the tuples with all their tensors on a device.

        Args:
            device: Device of the copy, "cuda" for instance.

        Returns:
            Tuples on the device. Tensors already on it are shared with the
            original ones.
        """
        tensors = {
            name: value.to(device)
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }

        return replace(self, **tensors)

    @classmethod
    def collate(cls, tuples: list[EdgeTuples]) -> EdgeTuples:
        """Combine tuples of several graphs into the tuples of their batch.

        The edges of an element are shifted by the number of edges of the
        elements before it, so that index points to the edges of the batch of
        the graphs (Graph.collate). The other fields are concatenated, batch
        and ptr are rebuilt.

        Args:
            tuples: List of EdgeTuples to batch together, each one with one or
                several structures, with tuples of the same size.

        Returns:
            A single EdgeTuples. It has the angles if all the elements have them.

        Raises:
            ValueError: If the list is empty, if the tuples do not have the same
                size, or if only some of the elements have their angles.
        """
        if len(tuples) == 0:
            raise ValueError("tuples list cannot be empty")

        if len({t.index.shape[0] for t in tuples}) != 1:
            raise ValueError("cannot collate tuples of different sizes")

        # edge offset per element: cumulative total edges across previous elements
        total_edges = torch.stack([t.num_edges.sum() for t in tuples])
        edge_offsets = F.pad(torch.cumsum(total_edges, dim=0), (1, 0), value=0)[:-1]

        index = torch.cat(
            [t.index + off for t, off in zip(tuples, edge_offsets)], dim=1
        )
        num_edges = torch.cat([t.num_edges for t in tuples], dim=0)
        num_tuples = torch.cat([t.num_tuples for t in tuples], dim=0)
        ptr = F.pad(torch.cumsum(num_tuples, dim=0), (1, 0), value=0)
        batch = torch.arange(len(num_tuples), device=index.device).repeat_interleave(
            num_tuples
        )

        return cls(
            index=index,
            num_edges=num_edges,
            num_tuples=num_tuples,
            batch=batch,
            ptr=ptr,
            angle=cat_optional([t.angle for t in tuples], "angle"),
            _validate=True,
        )

    @classmethod
    def separate(cls, tuples: EdgeTuples) -> list[EdgeTuples]:
        """Split batched tuples into the tuples of each structure.

        Args:
            tuples: Batched EdgeTuples.

        Returns:
            List of EdgeTuples of one structure, whose index starts at 0: it
            points to the edges of the separated graph (Graph.separate).
        """
        device = tuples.device()
        edge_ptr = F.pad(torch.cumsum(tuples.num_edges, dim=0), (1, 0), value=0)
        result = []

        for i in range(len(tuples.num_edges)):
            start = int(tuples.ptr[i])
            end = int(tuples.ptr[i + 1])

            result.append(
                cls(
                    index=tuples.index[:, start:end] - edge_ptr[i],
                    num_edges=tuples.num_edges[i : i + 1],
                    num_tuples=tuples.num_tuples[i : i + 1],
                    batch=torch.zeros(end - start, dtype=torch.int64, device=device),
                    ptr=torch.tensor(
                        [0, end - start], dtype=torch.int64, device=device
                    ),
                    angle=None if tuples.angle is None else tuples.angle[start:end],
                    _validate=True,
                )
            )

        return result
