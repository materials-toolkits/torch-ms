# Derived from DiffCSP (https://github.com/jiaor17/DiffCSP), Copyright (c) 2023
# Rui Jiao, MIT license (see models/diffcsp/LICENSE).
#
# Changes: this file is a rewrite of the CSPNet of DiffCSP
# (diffcsp/pl_modules/cspnet.py), for inference only. The complete graph is
# read from torchms Structures instead of being built by the network with a
# loop over the structures, and the basis of the differences of the
# coordinates is computed once instead of in every layer. Only the variant of
# the published weights is kept: sines and cosines of the differences, inner
# products of the lattice vectors, a normalization in every layer and a
# complete graph. The graph of the nearest neighbours and the prediction of a
# scalar are removed.
# Names of modules and parameters follow DiffCSP to load its weights.
"""CSPNet, a network on the complete graph of fractional coordinates.

Features live on the atoms. A layer builds a message for each pair of atoms of
a structure from their features, from the inner products of the lattice
vectors and from a Fourier basis of the difference of their fractional
coordinates, and gives each atom the mean of its messages. The network reads
no distance and no angle: its inputs do not change with a rotation of the
structure or with a translation of its atoms, and the basis has the period of
the cell. It gives a vector of each atom, in fractional coordinates, and a 3 x 3
matrix of each structure, which follows the rotations of the lattice. On
demand it reads the element of an atom as a vector of weights of the elements,
and gives such a vector for each atom: a diffusion of the elements.

The network also reads a vector of each structure, given to all its atoms: the
time of a diffusion.

CSPNet is published in "Crystal Structure Prediction by Joint Equivariant
Diffusion", Jiao, Huang, Lin, Han, Chen, Lu and Liu (arXiv:2309.04475). It is
rewritten here from the code of its authors, see torchms.models.diffcsp.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.graph.graph import Graph
from torchms.models.layers.basis import periodic_fourier


@dataclass(frozen=True)
class CSPNetConfig:
    """Hyperparameters of a CSPNet.

    Args:
        latent_dim: Size of the vector of a structure given to its atoms.
        num_elements: Rows of the embedding of the elements, from hydrogen.
        hidden_dim: Features of the atoms and of the messages.
        num_layers: Number of message passing layers.
        num_freqs: Frequencies of the Fourier basis of the differences of the
            fractional coordinates.
        soft_elements: If True, the element of an atom is given as a vector of
            num_elements weights, read by a linear layer, in place of an index
            read in a table.
        predict_elements: If True, the network also gives a vector of
            num_elements values for each atom.
    """

    latent_dim: int
    num_elements: int
    hidden_dim: int
    num_layers: int
    num_freqs: int
    soft_elements: bool
    predict_elements: bool

    @property
    def edge_dim(self) -> int:
        """Size of what a layer reads on a pair of atoms, besides their features.

        The 9 inner products of the lattice vectors, then a sine and a cosine
        of each frequency for each of the 3 coordinates.
        """
        return 9 + 2 * 3 * self.num_freqs


class CSPLayer(nn.Module):
    """Message passing layer of a CSPNet.

    Args:
        hidden_dim: Features of the atoms and of the messages.
        edge_dim: Size of what the layer reads on a pair of atoms.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        graph: Graph,
        node_features: Float[torch.Tensor, "n_atoms d"],
        edge_features: Float[torch.Tensor, "n_edges edge_dim"],
    ) -> Float[torch.Tensor, "n_atoms d"]:
        """Features of the atoms after the layer.

        Args:
            graph: Complete graph of the atoms.
            node_features: Features of the atoms.
            edge_features: Inner products of the lattice vectors and basis of
                the difference of the coordinates of each edge.
        """
        normalized = self.layer_norm(node_features)

        messages = self.edge_mlp(
            torch.cat(
                [
                    graph.unpool_senders(normalized),
                    graph.unpool_receivers(normalized),
                    edge_features,
                ],
                dim=1,
            )
        )
        # an atom gets the mean of the messages of the edges it sends
        received = graph.aggregate_edges_to_nodes(messages, "mean", to="sender")

        return node_features + self.node_mlp(torch.cat([normalized, received], dim=1))


class CSPNet(nn.Module):
    """CSPNet: a vector of each atom and a matrix of each structure.

    Args:
        config: Hyperparameters of the network.

    Examples:
        >>> from torchms.data.batching import Structures
        >>> from torchms.graph.transforms import AddCompleteGraph
        >>> config = CSPNetConfig(
        ...     latent_dim=8,
        ...     num_elements=100,
        ...     hidden_dim=16,
        ...     num_layers=2,
        ...     num_freqs=4,
        ...     soft_elements=False,
        ...     predict_elements=False,
        ... )
        >>> model = CSPNet(config).eval()
        >>> structures = Structures.collate([Structures.diamond(), Structures.water()])
        >>> structures = AddCompleteGraph(self_loops=True)(structures)
        >>> with torch.no_grad():
        ...     lattice_out, coord_out, type_out = model(
        ...         torch.zeros(2, 8),
        ...         structures.z - 1,
        ...         structures.graph,
        ...         structures.x,
        ...         structures.lattice,
        ...     )
        >>> lattice_out.shape, coord_out.shape, type_out
        (torch.Size([2, 3, 3]), torch.Size([11, 3]), None)
    """

    def __init__(self, config: CSPNetConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.node_embedding: nn.Module
        if config.soft_elements:
            self.node_embedding = nn.Linear(config.num_elements, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(config.num_elements, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + config.latent_dim, hidden_dim)

        for i in range(config.num_layers):
            self.add_module(f"csp_layer_{i}", CSPLayer(hidden_dim, config.edge_dim))

        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if config.predict_elements:
            self.type_out = nn.Linear(hidden_dim, config.num_elements)

    def forward(
        self,
        latent: Float[torch.Tensor, "n_struct latent_dim"],
        element: (
            Int[torch.Tensor, "n_atoms"] | Float[torch.Tensor, "n_atoms n_elements"]
        ),
        graph: Graph,
        x: Float[torch.Tensor, "n_atoms 3"],
        lattice: Float[torch.Tensor, "n_struct 3 3"],
    ) -> tuple[
        Float[torch.Tensor, "n_struct 3 3"],
        Float[torch.Tensor, "n_atoms 3"],
        Float[torch.Tensor, "n_atoms n_elements"] | None,
    ]:
        """Matrix of each structure, vector and elements of each atom.

        The coordinates and the lattices are given apart from the graph: a
        diffusion gives noisy ones, on a graph which does not change.

        Args:
            latent: Vector of each structure.
            element: Index of the element of each atom, from 0 for hydrogen,
                shape (n_atoms,). With soft_elements, weight of each element
                in each atom, from hydrogen, shape (n_atoms, num_elements).
            graph: Complete graph of the atoms, with an edge from each atom to
                itself, see AddCompleteGraph.
            x: Fractional coordinates of the atoms, inside or outside the cell.
            lattice: Lattice vectors of the structures, as rows.

        Returns:
            Matrix of each structure, a 3 x 3 matrix times its lattice, shape
            (n_struct, 3, 3), vector of each atom, in fractional coordinates,
            shape (n_atoms, 3), and with predict_elements a value of each
            element for each atom, from hydrogen, shape (n_atoms,
            num_elements), None otherwise.
        """
        # from the sender of an edge to its receiver, modulo the cell
        difference = (graph.unpool_receivers(x) - graph.unpool_senders(x)) % 1.0
        products = (lattice @ lattice.transpose(-1, -2)).flatten(start_dim=1)
        edge_features = torch.cat(
            [
                graph.unpool_senders(graph.unpool_structures(products)),
                periodic_fourier(difference, self.config.num_freqs),
            ],
            dim=1,
        )

        h = self.node_embedding(element)
        h = self.atom_latent_emb(torch.cat([h, graph.unpool_structures(latent)], dim=1))

        for i in range(self.config.num_layers):
            h = self.get_submodule(f"csp_layer_{i}")(graph, h, edge_features)

        h = self.final_layer_norm(h)

        coord_out = self.coord_out(h)

        pooled = graph.aggregate_nodes_to_structures(h, "mean")
        lattice_out = self.lattice_out(pooled).view(-1, 3, 3) @ lattice

        type_out = self.type_out(h) if self.config.predict_elements else None

        return lattice_out, coord_out, type_out
