# Derived from MatterSim (https://github.com/microsoft/mattersim), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the energy prediction of the M3GNet model
# of MatterSim 1.2.5 (src/mattersim/forcefield/m3gnet/m3gnet.py and modules/),
# for inference only. Forces, stress, the Graphormer variant, training,
# fine-tuning, the moving average of the weights, the applications (relaxation,
# molecular dynamics, phonons), the calculator of ASE and the compiled models
# are removed. The graph is built by torchms.graph instead of pymatgen, the
# roots and the normalizations of the spherical Bessel functions are computed
# instead of being written in a table, the smooth Bessel basis is
# torchms.models.layers.basis.SmoothBesselBasis, and batches are torchms
# Structures. Names of modules and parameters follow MatterSim to load its
# pretrained weights.
"""Energy prediction of the pretrained M3GNet potential of MatterSim.

The network has features on atoms and on bonds, the pairs of atoms closer than
5 Å. A bond starts from a smooth Bessel basis of its length. In each of the
three blocks, a bond is first updated from the angles it forms with the other
bonds of its atom shorter than 4 Å, in a basis of spherical Bessel functions
of the length of the other bond and of spherical harmonics of the angle,
weighted by the atom they share. Bonds and atoms are then updated from each
other. The energy of an atom is scaled and shifted by its element, and the
energy of a structure is their sum.

Weights are the ones of the authors, see config.yaml.
"""

from __future__ import annotations

import os
from typing import Mapping

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn

from torchms.data.augmentation.transforms import Compose, Transform
from torchms.data.batching import Structures
from torchms.graph.geometry import triplet_cosines
from torchms.graph.graph import EdgeTuples, Graph
from torchms.graph.transforms import (
    AddRadiusGraph,
    AddTriplets,
    attached_graph,
    attached_tuples,
    edges_vectors,
)
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
)
from torchms.models.estimator.weights import download
from torchms.models.layers.basis import (
    SmoothBesselBasis,
    legendre,
    polynomial_cutoff,
    spherical_bessel,
    spherical_bessel_roots,
)
from torchms.models.mattersim.config import Config, NetworkConfig

# what spherical_bessel_terms writes out: degrees 0 to 3
MAX_BESSEL_DEGREE = 3


def spherical_bessel_terms(
    degree: int, roots: Float[torch.Tensor, "max_n"]
) -> list[tuple[int, str, Float[torch.Tensor, "max_n"]]]:
    """Closed form of N j_l(z x) as a sum of c x^p sin(z x) or cos(z x), over x^(l+1).

    N is the normalization 1 / |j_(l+1)(z)| of the function of root z. The
    terms are the ones of the table of MatterSim, in its order.

    Returns:
        For each term, the power p, "sin" or "cos", and the coefficient of
        each root.

    Examples:
        >>> roots = torch.tensor([3.1416])
        >>> [(p, kind) for p, kind, _ in spherical_bessel_terms(1, roots)]
        [(1, 'cos'), (0, 'sin')]
    """
    if degree > MAX_BESSEL_DEGREE:
        raise ValueError(f"degrees up to {MAX_BESSEL_DEGREE} are written out")

    norm = 1.0 / spherical_bessel(degree + 1, roots)[degree + 1].abs()
    z = roots

    return {
        # N / z is 1 for j_0, whose roots are multiples of pi: written as 1
        0: [(0, "sin", torch.ones_like(z))],
        1: [(1, "cos", -norm / z), (0, "sin", norm / z**2)],
        2: [
            (2, "sin", -norm / z),
            (1, "cos", -3 * norm / z**2),
            (0, "sin", 3 * norm / z**3),
        ],
        3: [
            (3, "cos", norm / z),
            (2, "sin", -6 * norm / z**2),
            (1, "cos", -15 * norm / z**3),
            (0, "sin", 15 * norm / z**4),
        ],
    }[degree]


class LinearLayer(nn.Module):
    """A linear layer, then an activation: none, a sigmoid or swish."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: str | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        """Output of the layer."""
        x = self.linear(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "swish":
            return x * torch.sigmoid(x)
        return x


class MLP(nn.Module):
    """Layers one after the other."""

    def __init__(
        self, in_dim: int, out_dims: list[int], activation: str | None, use_bias: bool
    ):
        super().__init__()
        sizes = [in_dim, *out_dims]
        self.mlp = nn.Sequential(
            *(
                LinearLayer(size_in, size_out, use_bias, activation)
                for size_in, size_out in zip(sizes[:-1], sizes[1:])
            )
        )

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        """Output of the last layer."""
        return self.mlp(x)


class GatedMLP(nn.Module):
    """Product of a network g and of a gate sigma, whose last layer is a sigmoid.

    Args:
        in_dim: Input features.
        out_dims: Features after each layer.
        activations: Activation of each layer of g; the gate has the same
            ones, but a sigmoid last.
        use_bias: Whether the layers have a bias.
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: list[int],
        activations: list[str | None],
        use_bias: bool = True,
    ):
        super().__init__()
        sizes = [in_dim, *out_dims]
        gates = [*activations[:-1], "sigmoid"]
        self.g = nn.Sequential(
            *(
                LinearLayer(size_in, size_out, use_bias, activation)
                for size_in, size_out, activation in zip(
                    sizes[:-1], sizes[1:], activations
                )
            )
        )
        self.sigma = nn.Sequential(
            *(
                LinearLayer(size_in, size_out, use_bias, activation)
                for size_in, size_out, activation in zip(sizes[:-1], sizes[1:], gates)
            )
        )

    def forward(self, x: Float[torch.Tensor, "n f"]) -> Float[torch.Tensor, "n d"]:
        """The network times its gate."""
        return self.g(x) * self.sigma(x)


class SphericalBasisLayer(nn.Module):
    """Basis of an angle: spherical Bessel functions of the length of a bond
    times spherical harmonics of the angle.

    The functions of degree l are written in closed form, as MatterSim does,
    over the length divided by the radius, and multiplied by sqrt(2 / r_max^3):
    the closed forms round otherwise than a recurrence at short lengths.

    Args:
        network: Hyperparameters of the network.
        radius: Radius of the graph in Å.
    """

    factor: torch.Tensor
    roots: torch.Tensor
    coefficients: torch.Tensor

    def __init__(self, network: NetworkConfig, radius: float):
        super().__init__()
        self.max_n, self.max_l = network.max_n, network.max_l
        self.radius = radius

        roots = torch.tensor(
            spherical_bessel_roots(self.max_l, self.max_n), dtype=torch.float64
        )
        terms = [
            spherical_bessel_terms(degree, roots[degree])
            for degree in range(self.max_l)
        ]
        self.powers = [[power for power, _, _ in degree] for degree in terms]
        self.kinds = [[kind for _, kind, _ in degree] for degree in terms]
        # coefficient of each term of each degree, padded with zeros, by root
        coefficients = torch.zeros(self.max_l, MAX_BESSEL_DEGREE + 1, self.max_n)
        for degree, degree_terms in enumerate(terms):
            for i, (_, _, values) in enumerate(degree_terms):
                coefficients[degree, i] = values.float()

        self.register_buffer("factor", torch.sqrt(torch.tensor(2.0 / radius**3)))
        self.register_buffer("roots", roots.float(), persistent=False)
        self.register_buffer("coefficients", coefficients, persistent=False)

    def radial(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        """The spherical Bessel functions, by degree then by root."""
        r = distance / self.radius
        functions = []
        for degree in range(self.max_l):
            for root in range(self.max_n):
                angle = self.roots[degree, root] * r
                total = torch.zeros_like(r)
                for i, (power, kind) in enumerate(
                    zip(self.powers[degree], self.kinds[degree])
                ):
                    trigonometric = (
                        torch.sin(angle) if kind == "sin" else torch.cos(angle)
                    )
                    term = self.coefficients[degree, i, root] * trigonometric
                    total = total + (term * r**power if power else term)
                functions.append(total / r ** (degree + 1))

        return torch.stack(functions, dim=-1) * self.factor

    def forward(
        self, distance: Float[torch.Tensor, "n"], angle: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        """Basis of the angles of a bond of a length with another bond."""
        # the harmonic of degree l multiplies the max_n functions of degree l
        harmonics = legendre(self.max_l, torch.cos(angle))

        return self.radial(distance) * harmonics.repeat_interleave(self.max_n, dim=1)


class ThreeDInteraction(nn.Module):
    """Update of the bonds from the angles they form with the other bonds."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        self.atom_mlp = LinearLayer(
            network.units, network.basis_dim, activation="sigmoid"
        )
        self.edge_gate_mlp = GatedMLP(
            network.basis_dim, [network.units], ["swish"], use_bias=False
        )

    def forward(  # pylint: disable=too-many-positional-arguments
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        triplet_basis: Float[torch.Tensor, "n_triplets b"],
        graph: Graph,
        angles: EdgeTuples,
    ) -> Float[torch.Tensor, "n_edges d"]:
        """Bonds after the update.

        Args:
            atoms: Features of the atoms.
            bonds: Features of the bonds.
            triplet_basis: Basis of each angle, weighted by the cutoffs of its
                two bonds.
            graph: Bonds of the atoms.
            angles: Pairs of bonds i→j, i→k of the atoms.
        """
        # MatterSim weights an angle i→j, i→k by its atom i, which matgl
        # weights by the atom k
        center = graph.unpool_senders(self.atom_mlp(atoms))
        messages = triplet_basis * angles.unpool_edges(center, position=1)
        received = angles.aggregate_to_edges(messages, position=0)

        return bonds + self.edge_gate_mlp(received)


class MainBlock(nn.Module):
    """Block of message passing: angles, then bonds, then atoms."""

    def __init__(self, network: NetworkConfig):
        super().__init__()
        units, max_n = network.units, network.max_n
        self.gated_mlp_atom = GatedMLP(3 * units, [units, units], ["swish", "swish"])
        self.edge_layer_atom = LinearLayer(max_n, units, bias=False, activation="swish")
        self.gated_mlp_edge = GatedMLP(3 * units, [units, units], ["swish", "swish"])
        self.edge_layer_edge = LinearLayer(max_n, units, bias=False)
        self.three_body = ThreeDInteraction(network)

    def forward(  # pylint: disable=too-many-positional-arguments
        self,
        atoms: Float[torch.Tensor, "n_atoms d"],
        bonds: Float[torch.Tensor, "n_edges d"],
        basis: Float[torch.Tensor, "n_edges max_n"],
        triplet_basis: Float[torch.Tensor, "n_triplets b"],
        graph: Graph,
        angles: EdgeTuples,
    ) -> tuple[Float[torch.Tensor, "n_atoms d"], Float[torch.Tensor, "n_edges d"]]:
        """Atoms and bonds after the block.

        Args:
            atoms: Features of the atoms.
            bonds: Features of the bonds.
            basis: Radial basis of each bond.
            triplet_basis: Weighted basis of each angle.
            graph: Bonds of the atoms.
            angles: Pairs of bonds of the atoms.
        """
        bonds = self.three_body(atoms, bonds, triplet_basis, graph, angles)

        senders, receivers = graph.unpool_senders(atoms), graph.unpool_receivers(atoms)
        bonds = bonds + self.gated_mlp_edge(
            torch.cat([senders, receivers, bonds], dim=1)
        ) * self.edge_layer_edge(basis)

        messages = self.gated_mlp_atom(
            torch.cat([senders, receivers, bonds], dim=1)
        ) * self.edge_layer_atom(basis)

        return atoms + graph.aggregate_edges_to_nodes(messages, to="sender"), bonds


class AtomScaling(nn.Module):  # pylint: disable=abstract-method
    """Scale and shift of the energy of an atom by its element.

    Attributes:
        scale: Scale of each atomic number, from 0.
        shift: Shift of each atomic number, in eV.
    """

    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self, max_z: int):
        super().__init__()
        self.register_buffer("scale", torch.ones(max_z + 1))
        self.register_buffer("shift", torch.zeros(max_z + 1))


class MatterSim(PretrainedModel[Config]):
    """Total energy of structures predicted by the M3GNet potential of MatterSim.

    Examples:
        >>> model = MatterSim()
        >>> structures = model.transform()(
        ...     Structures.collate([Structures.diamond(), Structures.zincblende()])
        ... )
        >>> with torch.no_grad():
        ...     energy = model(structures)
        >>> energy.shape, energy.dtype
        (torch.Size([2]), torch.float32)
    """

    CONFIG = Config
    GRAPH_FIELDS = ("graph", "triplets")

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model
        radius = self.config.graph.radius

        self.rbf = SmoothBesselBasis(radius, network.max_n)
        self.sbf = SphericalBasisLayer(network, radius)
        self.edge_encoder = MLP(network.max_n, [network.units], "swish", use_bias=False)
        self.graph_conv = nn.ModuleList(
            MainBlock(network) for _ in range(network.num_blocks)
        )
        self.final = GatedMLP(
            network.units, [network.units, network.units, 1], ["swish", "swish", None]
        )
        self.atom_embedding = MLP(
            network.max_z + 1, [network.units], None, use_bias=False
        )
        self.normalizer = AtomScaling(network.max_z)

    # weights

    @classmethod
    def for_weights(cls, name: str | None = None) -> MatterSim:
        """Model of the size of some weights, before they are loaded.

        Args:
            name: Name of the weights, see Config.for_weights. The default
                ones if None.
        """
        config = Config.load()

        return cls(config if name is None else config.for_weights(name))

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download weights listed in config.yaml and load them.

        The checkpoint holds tensors only, and the arguments of the model.
        """
        (path,) = download("mattersim", name, cache)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise InvalidWeights(f"{path} is not a checkpoint of MatterSim")

        self.load_pretrained(
            checkpoint["model"],
            checkpoint.get("model_args"),
            checkpoint.get("model_name"),
        )

    def load_pretrained(
        self,
        state_dict: Mapping[str, torch.Tensor],
        model_args: Mapping[str, object] | None = None,
        model_name: str | None = None,
    ) -> None:
        """Load the weights of an M3GNet model of MatterSim.

        Args:
            state_dict: Parameters and buffers of the model.
            model_args: Arguments of the model, checked if given.
            model_name: Architecture of the weights, checked if given.

        Raises:
            InvalidWeights: If parameters are missing or unknown, or if the
                model is of another architecture or was built with other
                arguments.
        """
        upstream = self.config.upstream
        if model_name is not None and model_name != upstream.model_name:
            raise InvalidWeights(
                f"weights of a model {model_name}, not {upstream.model_name}"
            )
        if model_args is not None:
            different = different_arguments(model_args, upstream.model_args)
            if different:
                raise InvalidWeights(
                    f"weights of a model with other arguments: {different}"
                )

        self.load_parameters(state_dict)

    @classmethod
    def transform(cls, config: Config | None = None) -> Transform:
        """Transform attaching the bonds and the angles of the model.

        Atoms closer than the radius, 5 Å, are bonded, periodic images of the
        atom itself included. Angles are the pairs of bonds of at most the
        three-body radius, 4 Å, starting from the same atom.

        Args:
            config: Configuration of the model the graph is for. The one of
                config.yaml by default.
        """
        graph = (config or Config.load()).graph

        return Compose(
            [
                AddRadiusGraph(graph.radius, self_images=True),
                AddTriplets(radius=graph.three_body_radius),
            ]
        )

    # network

    def check_elements(self, z: Int[torch.Tensor, "n_atoms"]) -> None:
        """Refuse atoms the embedding does not have.

        Raises:
            ValueError: If an atomic number is not between 1 and max_z.
        """
        outside = (z < 1) | (z > self.config.model.max_z)
        if outside.any():
            unknown = z[outside].unique().tolist()
            raise ValueError(f"elements {unknown} are not supported by the model")

    def atom_energies(self, structures: Structures) -> Float[torch.Tensor, "n_atoms"]:
        """Energy of each atom in eV, scaled and shifted by its element.

        Raises:
            ValueError: If an element is unknown, or the structures do not
                have the graph and the triplets of the model.
        """
        self.check_graphs(structures)
        self.check_elements(structures.z)
        graph = attached_graph(structures, self)
        angles = attached_tuples(structures, "triplets", self)
        network = self.config.model

        vectors = edges_vectors(graph)
        distance = vectors.norm(dim=-1)
        margin = network.cosine_margin
        cosine = triplet_cosines(vectors, angles).clamp(-1 + margin, 1 - margin)

        # an angle is weighted by the smooth cutoffs of its two bonds
        envelope = polynomial_cutoff(
            distance, self.config.graph.three_body_radius, p=network.cutoff_power
        )
        weight = angles.unpool_edges(envelope, position=0) * angles.unpool_edges(
            envelope, position=1
        )
        # the basis reads the angle, from the cosine, as MatterSim
        triplet_basis = self.sbf(
            angles.unpool_edges(distance, position=1), torch.acos(cosine)
        ) * weight.unsqueeze(-1)

        # pylint: disable-next=not-callable
        one_hot = F.one_hot(structures.z, network.max_z + 1).to(distance.dtype)
        atoms = self.atom_embedding(one_hot)
        basis = self.rbf(distance)
        bonds = self.edge_encoder(basis)

        for block in self.graph_conv:
            atoms, bonds = block(atoms, bonds, basis, triplet_basis, graph, angles)

        energy = self.final(atoms).squeeze(-1)
        scaling = self.normalizer

        return scaling.scale[structures.z] * energy + scaling.shift[structures.z]

    def forward(self, structures: Structures) -> Float[torch.Tensor, "n_struct"]:
        """Predict the total energy of each structure in eV, in single precision.

        Raises:
            ValueError: If a structure has an element the model does not know.
        """
        energies = self.atom_energies(structures)

        return attached_graph(structures, self).aggregate_nodes_to_structures(energies)
