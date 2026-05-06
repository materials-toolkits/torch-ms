"""Graph dataclass for batched crystal structure graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pycrystalgen.graph.scatter import scatter_mean, scatter_sum


class InvalidGraph(Exception):
    """Exception raised when a graph has invalid data."""


@dataclass
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
    num_atoms:  Int[torch.Tensor, "n_struct"]
    num_edges:  Int[torch.Tensor, "n_struct"]
    edge_batch: Int[torch.Tensor, "n_edges"] | None = None
    edge_ptr:   Int[torch.Tensor, "n_struct+1"] | None = None
    _validate:  bool = False

    def __post_init__(self):
        if self._validate:
            return
        self.check_tensors()
        self._validate = True

    def check_tensors(self):
        """Validate graph data and fill derived fields.

        Raises:
            InvalidGraph: If edge or structure data is invalid.
        """
        device = torch.device("cpu")
        if isinstance(self.edge_index, torch.Tensor):
            device = self.edge_index.device

        n_struct = len(self.num_atoms)

        if n_struct < 1:
            raise InvalidGraph("must contain at least one structure")

        # convert num_edges early to compute n_edges
        if not isinstance(self.num_edges, torch.Tensor):
            try:
                self.num_edges = torch.tensor(self.num_edges, dtype=torch.int64, device=device)
            except TypeError as e:
                raise InvalidGraph(f"invalid field num_edges: {e}") from e

        n_edges = int(self.num_edges.sum())

        self.to_tensor("edge_index", dtype=torch.int64, shape=(2, n_edges), device=device)
        self.to_tensor("edge_shift", dtype=torch.int64, shape=(n_edges, 3), device=device)
        self.to_tensor("num_atoms", dtype=torch.int64, shape=(n_struct,), device=device)
        self.to_tensor("num_edges", dtype=torch.int64, shape=(n_struct,), device=device)

        if (self.num_atoms <= 0).any():
            raise InvalidGraph("num_atoms must be positive for every structure")

        self.to_tensor(
            "edge_batch",
            dtype=torch.int64,
            shape=(n_edges,),
            device=device,
            default=lambda g: torch.arange(n_struct, dtype=torch.int64, device=device)
            .repeat_interleave(g.num_edges),
        )
        if n_edges > 0 and (self.edge_batch[:-1] > self.edge_batch[1:]).any():
            raise InvalidGraph("edges must be ordered by structure in edge_batch")

        self.to_tensor(
            "edge_ptr",
            dtype=torch.int64,
            shape=(n_struct + 1,),
            device=device,
            default=lambda g: F.pad(torch.cumsum(g.num_edges, dim=0), (1, 0), value=0),
        )

    def device(self) -> torch.device:
        return self.edge_index.device

    def aggregate_edges_to_nodes(
        self,
        src: Float[torch.Tensor, "n_edges ..."],
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Float[torch.Tensor, "n_atoms ..."]:
        """Aggregate per-edge features to receiver nodes (message passing).

        Args:
            src: Per-edge features, shape (n_edges, *).
            reduce: Aggregation function — 'sum' or 'mean'.

        Returns:
            Per-node aggregated features, shape (n_atoms, *).
        """
        n_atoms = int(self.num_atoms.sum())
        if reduce == "sum":
            return scatter_sum(src, self.edge_index[1], n_atoms)
        if reduce == "mean":
            return scatter_mean(src, self.edge_index[1], n_atoms)
        raise ValueError(f"unknown reduce {reduce!r}, expected 'sum' or 'mean'")

    def aggregate_nodes_to_structures(
        self,
        src: Float[torch.Tensor, "n_atoms ..."],
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Float[torch.Tensor, "n_struct ..."]:
        """Aggregate per-node features to structures (global pooling).

        Args:
            src: Per-node features, shape (n_atoms, *).
            reduce: Aggregation function — 'sum' or 'mean'.

        Returns:
            Per-structure aggregated features, shape (n_struct, *).
        """
        node_batch = torch.arange(
            len(self.num_atoms), device=self.device()
        ).repeat_interleave(self.num_atoms)
        n_struct = len(self.num_atoms)
        if reduce == "sum":
            return scatter_sum(src, node_batch, n_struct)
        if reduce == "mean":
            return scatter_mean(src, node_batch, n_struct)
        raise ValueError(f"unknown reduce {reduce!r}, expected 'sum' or 'mean'")

    def aggregate_edges_to_structures(
        self,
        src: Float[torch.Tensor, "n_edges ..."],
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Float[torch.Tensor, "n_struct ..."]:
        """Aggregate per-edge features to structures (global pooling).

        Args:
            src: Per-edge features, shape (n_edges, *).
            reduce: Aggregation function — 'sum' or 'mean'.

        Returns:
            Per-structure aggregated features, shape (n_struct, *).
        """
        n_struct = len(self.num_edges)
        if reduce == "sum":
            return scatter_sum(src, self.edge_batch, n_struct)
        if reduce == "mean":
            return scatter_mean(src, self.edge_batch, n_struct)
        raise ValueError(f"unknown reduce {reduce!r}, expected 'sum' or 'mean'")

    def to_tensor(
        self,
        field: str,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
        default: Callable[[Graph], torch.Tensor] | None = None,
    ) -> None:
        values = getattr(self, field)

        if values is None:
            if default is None:
                raise InvalidGraph(f'no default value for "{field}"')
            tensor = default(self)
        elif not isinstance(values, torch.Tensor):
            try:
                tensor = torch.tensor(values, dtype=dtype)
            except TypeError as e:
                raise InvalidGraph(f"invalid field {field}: {e}") from e
        else:
            tensor = values

        if tensor.dtype != dtype:
            tensor = tensor.type(dtype)

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
        num_atoms  = torch.cat([g.num_atoms for g in graphs], dim=0)
        num_edges  = torch.cat([g.num_edges for g in graphs], dim=0)
        edge_ptr   = F.pad(torch.cumsum(num_edges, dim=0), (1, 0), value=0)
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

            result.append(cls(
                edge_index=graph.edge_index[:, e_start:e_end] - atom_ptr[i],
                edge_shift=graph.edge_shift[e_start:e_end],
                num_atoms=graph.num_atoms[i : i + 1],
                num_edges=graph.num_edges[i : i + 1],
                edge_batch=torch.zeros(n_edges_i, dtype=torch.int64, device=graph.device()),
                edge_ptr=torch.tensor([0, n_edges_i], dtype=torch.int64, device=graph.device()),
                _validate=True,
            ))

        return result
