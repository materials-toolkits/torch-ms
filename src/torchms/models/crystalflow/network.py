# Derived from CrystalFlow (https://github.com/ixsluo/CrystalFlow), Copyright (c)
# 2024 Xiaoshan Luo, MIT license, itself derived from DiffCSP, Copyright (c) 2023
# Rui Jiao, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the networks of CrystalFlow at the tag
# v1.0.0-alpha.1 (diffcsp/pl_modules/cspnet.py, type_module.py and
# conditioning.py), for inference only, with the only encodings of the
# published weights: the table of the elements and the gaussians of a
# property. The layers are the ones of torchms.models.layers.cspnet. Names of
# modules and parameters follow CrystalFlow to load its weights.
"""The networks of CrystalFlow: its CSPNet, its table of elements, its conditions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn

from torchms.graph.graph import Graph
from torchms.models.crystalflow.config import GenerationConfig, NetworkConfig
from torchms.models.layers.basis import (
    miller_indices,
    periodic_fourier,
    reciprocal_fourier,
)
from torchms.models.layers.cspnet import CSPLayer


class GaussianExpansion(nn.Module):  # pylint: disable=abstract-method
    """Gaussians of a property, one standard deviation apart, as in CrystalFlow.

    Attributes:
        widths: Width of each gaussian: the distance between two centers.
        offsets: Center of each gaussian.
    """

    widths: torch.Tensor
    offsets: torch.Tensor

    def __init__(self, start: float, stop: float, num_basis: int):
        super().__init__()
        offsets = torch.linspace(start, stop, num_basis)
        self.register_buffer(
            "widths", (offsets[1] - offsets[0]) * torch.ones_like(offsets)
        )
        self.register_buffer("offsets", offsets)

    def expand(
        self, value: Float[torch.Tensor, "n_struct 1"]
    ) -> Float[torch.Tensor, "n_struct num_basis"]:
        """The gaussians of the standardized property of each structure."""
        coefficient = -0.5 / torch.pow(self.widths, 2)[None, :]

        return torch.exp(coefficient * torch.pow(value - self.offsets[None, :], 2))


class ScalarEmbedding(nn.Module):  # pylint: disable=abstract-method
    """Embedding of a property: its gaussians, without a network after them."""

    expansion_net: GaussianExpansion

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.expansion_net = GaussianExpansion(
            network.condition_start, network.condition_stop, network.condition_basis
        )


class MultiEmbedding(nn.Module):
    """Embedding of the properties of a model, one after the other."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.conditions = network.conditions
        self.sub_emb_dict = nn.ModuleDict(
            {name: ScalarEmbedding(network) for name in network.conditions}
        )

    def forward(
        self, standardized: Float[torch.Tensor, "n_struct n_conditions"]
    ) -> Float[torch.Tensor, "n_struct condition_dim"]:
        """Embedding of the standardized properties of each structure, in order."""
        expansions = []
        for i, name in enumerate(self.conditions):
            embedding = self.sub_emb_dict[name]
            assert isinstance(embedding, ScalarEmbedding)
            expansions.append(
                embedding.expansion_net.expand(standardized[:, i : i + 1])
            )

        return torch.cat(expansions, dim=1)


class TypeTableModule(nn.Module):  # pylint: disable=abstract-method
    """Elements as one-hot vectors of their row and of their column in a table.

    Attributes:
        reordered_map: Atomic number of each cell of the table, 0 for none.
        reordered_indices: Row and column of each atomic number, from 1.
        mask: 1 for the cells of the table with an element.
    """

    reordered_map: torch.Tensor
    reordered_indices: torch.Tensor
    mask: torch.Tensor

    def __init__(self, generation: GenerationConfig):
        super().__init__()
        self.num_rows, self.num_columns = generation.num_rows, generation.num_columns
        indices = torch.tensor(generation.table)
        reordered_map = torch.zeros(self.num_rows, self.num_columns, dtype=torch.int64)
        reordered_map[indices[:, 0], indices[:, 1]] = torch.arange(1, len(indices) + 1)

        self.register_buffer("reordered_map", reordered_map)
        self.register_buffer("reordered_indices", indices)
        self.register_buffer("mask", (reordered_map > 0).float())

    def encode(
        self, z: Int[torch.Tensor, "n_atoms"]
    ) -> Float[torch.Tensor, "n_atoms type_dim"]:
        """One-hot vectors of the row then of the column of each atomic number."""
        cells = self.reordered_indices[z - 1]
        # pylint: disable=not-callable
        return torch.cat(
            [
                F.one_hot(cells[:, 0], self.num_rows),
                F.one_hot(cells[:, 1], self.num_columns),
            ],
            dim=1,
        ).float()

    def decode(
        self, vectors: Float[torch.Tensor, "n_atoms type_dim"]
    ) -> Int[torch.Tensor, "n_atoms"]:
        """Atomic numbers: the largest row, then its largest column with an element."""
        rows = vectors[:, : self.num_rows].argmax(dim=-1)
        columns = (
            F.softmax(vectors[:, self.num_rows :], dim=1) * self.mask[rows]
        ).argmax(dim=-1)

        return self.reordered_map[rows, columns]


class CSPNet(nn.Module):
    """CSPNet of CrystalFlow: a velocity of the cell, of each atom and of its element.

    The layers are the ones of DiffCSP; an edge reads the plane waves of the
    reciprocal lattice, the Fourier basis of the difference of the
    coordinates, then the numbers of the cell, in this order. With conditions,
    an embedding of them is added to the features of the atoms before each
    layer.

    Args:
        config: Hyperparameters of the network.
    """

    millers: torch.Tensor

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.node_embedding: nn.Module
        if config.soft_elements:
            self.node_embedding = nn.Linear(config.type_dim, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(config.num_elements, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + config.time_dim, hidden_dim)
        for i in range(config.num_layers):
            self.add_module(f"csp_layer_{i}", CSPLayer(hidden_dim, config.edge_dim))
            if config.conditions:
                self.add_module(
                    f"cemb_mixin_{i}", nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
                self.add_module(
                    f"cemb_adapter_{i}",
                    nn.Sequential(
                        nn.Linear(config.condition_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                    ),
                )
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, config.lattice_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if config.predict_elements:
            self.type_out = nn.Linear(hidden_dim, config.type_dim)

        self.register_buffer(
            "millers", miller_indices(config.num_millers), persistent=False
        )

    def forward(  # pylint: disable=too-many-positional-arguments,too-many-locals
        self,
        time: Float[torch.Tensor, "n_struct time_dim"],
        element: Int[torch.Tensor, "n_atoms"] | Float[torch.Tensor, "n_atoms type_dim"],
        graph: Graph,
        x: Float[torch.Tensor, "n_atoms 3"],
        lattice: Float[torch.Tensor, "n_struct 6"],
        matrix: Float[torch.Tensor, "n_struct 3 3"],
        condition: Float[torch.Tensor, "n_struct condition_dim"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "n_struct 6"],
        Float[torch.Tensor, "n_atoms 3"],
        Float[torch.Tensor, "n_atoms type_dim"] | None,
    ]:
        """Velocity of the numbers of the cell, of the coordinates and of the elements.

        Args:
            time: Embedding of the time of each structure.
            element: Index of the element of each atom, from 0 for hydrogen,
                or with soft_elements its vector.
            graph: Complete graph of the atoms, with an edge from each atom to
                itself, see AddCompleteGraph.
            x: Fractional coordinates of the atoms.
            lattice: The six numbers of each cell.
            matrix: Lattice vectors of each cell, as rows: exp(S) of the
                numbers.
            condition: Embedding of the conditions of each structure, None
                for the velocity without them.

        Returns:
            Velocity of the numbers of each cell, of the coordinates of each
            atom, and with predict_elements of the vector of its element.
        """
        config = self.config
        # from the sender of an edge to its receiver, modulo the cell
        difference = (graph.unpool_receivers(x) - graph.unpool_senders(x)) % 1.0
        edge_matrix = graph.unpool_senders(
            graph.unpool_structures(matrix.flatten(start_dim=1))
        ).view(-1, 3, 3)
        edge_features = torch.cat(
            [
                reciprocal_fourier(difference, edge_matrix, self.millers),
                periodic_fourier(difference, config.num_freqs),
                graph.unpool_senders(graph.unpool_structures(lattice)),
            ],
            dim=1,
        )

        h = self.node_embedding(element)
        h = self.atom_latent_emb(torch.cat([h, graph.unpool_structures(time)], dim=1))
        for i in range(config.num_layers):
            if condition is not None:
                adapter = self.get_submodule(f"cemb_adapter_{i}")
                mixin = self.get_submodule(f"cemb_mixin_{i}")
                h = h + graph.unpool_structures(mixin(adapter(condition)))
            h = self.get_submodule(f"csp_layer_{i}")(graph, h, edge_features)
        h = self.final_layer_norm(h)

        pooled = graph.aggregate_nodes_to_structures(h, "mean")
        elements = self.type_out(h) if config.predict_elements else None

        return self.lattice_out(pooled), self.coord_out(h), elements
