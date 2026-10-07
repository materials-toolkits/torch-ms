"""Graph dataclass for batched crystal structure graphs."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from torchms.data.batching import Structures, TensorLike
from torchms.graph.geometry import edges_distances
from torchms.graph.sparse import Reduce, scatter


class InvalidGraph(Exception):
    """Exception raised when a graph has invalid data."""


class InvalidStructures(Exception):
    """Exception raised when references on structures is not a Structures object"""


@dataclass(init=False)
class Graph:
    """Container for batched crystal structure graphs.

    Mirrors the batching convention of Structures: all per-edge data is stored
    in flat concatenated tensors, with edge_ptr and edge_batch for indexing.

    Attributes:
        edge_index: Sender/receiver global atom indices, shape (2, n_edges).
        edge_shift: Integer PBC image offsets per edge, shape (n_edges, 3).
        num_atoms: Number of atoms per structure, shape (n_struct,).
        num_edges: Number of edges per structure, shape (n_struct,).
        edge_batch: Structure index per edge, shape (n_edges,).
        edge_ptr: CSR index pointers, shape (n_struct+1,).
    """

    edge_index: Int[torch.Tensor, "2 n_edges"]
    edge_shift: Int[torch.Tensor, "n_edges 3"]
    num_atoms: Int[torch.Tensor, "n_struct"]
    num_edges: Int[torch.Tensor, "n_struct"]
    edge_batch: Int[torch.Tensor, "n_edges"]
    edge_ptr: Int[torch.Tensor, "n_struct+1"]
    structures: Structures | None
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
        _validate: bool = False,
    ):
        self.edge_index = self.as_tensor("edge_index", edge_index)
        self.edge_shift = self.as_tensor("edge_shift", edge_shift)
        self.num_atoms = self.as_tensor("num_atoms", num_atoms)
        self.num_edges = self.as_tensor("num_edges", num_edges)

        # Missing fields are derived from num_edges, they are checked with
        # the given ones in check_tensors.
        counts = self.num_edges.flatten().long().clamp(min=0)

        if edge_batch is None:
            self.edge_batch = torch.arange(
                len(counts), dtype=torch.int64, device=counts.device
            ).repeat_interleave(counts)
        else:
            self.edge_batch = self.as_tensor("edge_batch", edge_batch)

        if edge_ptr is None:
            self.edge_ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)
        else:
            self.edge_ptr = self.as_tensor("edge_ptr", edge_ptr)

        self.structures = structures

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

    def device(self) -> torch.device:
        return self.edge_index.device

    def edges_vectors(self) -> Float[torch.Tensor, "n_edges 3"]:
        """Cartesian vector of each edge in Å, from its sender to its receiver."""
        assert self.structures is not None

        edges_frac = (
            self.structures.x[self.edge_index[1]]
            + self.edge_shift
            - self.structures.x[self.edge_index[0]]
        )
        # rows of the lattice are the lattice vectors: cartesian = fractional @ lattice
        return torch.bmm(
            edges_frac.unsqueeze(1), self.structures.lattice[self.edge_batch]
        ).squeeze(1)

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
    def collate(cls, graphs: list[Graph]) -> Graph:
        """Combine multiple graphs into a single batched graph.

        Args:
            graphs: List of Graph objects to batch together.

        Returns:
            A single Graph with concatenated tensors and offset edge_index.

        Raises:
            ValueError: If graphs list is empty.
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

        return cls(
            edge_index=edge_index,
            edge_shift=edge_shift,
            num_atoms=num_atoms,
            num_edges=num_edges,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            _validate=True,
        )

    @classmethod
    def separate(cls, graph: Graph) -> list[Graph]:
        """Split a batched graph into individual single-structure graphs.

        Args:
            graph: A batched Graph object.

        Returns:
            List of individual Graph objects.
        """
        n_structs = len(graph.num_atoms)
        atom_ptr = F.pad(torch.cumsum(graph.num_atoms, dim=0), (1, 0), value=0)
        result = []

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
                    _validate=True,
                )
            )

        return result
