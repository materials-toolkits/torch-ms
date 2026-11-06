"""Layers shared by the models.

* basis: bases of the distances, of the angles and of the directions, and
  smooth cutoffs;
* cspnet: CSPNet, a network on the complete graph of a structure, which reads
  fractional coordinates;
* dimenet: DimeNet++, a network on the bonds of a graph and their angles;
* gemnet: GemNet-T, a network on the edges of a symmetric graph and their
  triplets;
* line_graph_network: LineGraphNetwork, a network of the energy which lists
  the triplets and the quadruplets of atoms and uses their angles, as ALIGNN,
  M3GNet, CHGNet or DPA-3 do;
* equivariant_network: EquivariantNetwork, a network of the energy which only
  uses pairs of atoms and takes products of sums of spherical harmonics, as
  MACE does.

The two networks of the energy were written for this library: they have no
pretrained weights and no model of the library is built on them.

Each network takes its hyperparameters as a frozen dataclass, the Config of its
module: dimenet.Config, gemnet.Config, cspnet.Config, line_graph_network.Config
and equivariant_network.Config. A pretrained model builds the ones of its
layers from its config.yaml. The networks read the graph attached to the
structures.
"""

from torchms.models.layers.basis import (
    AngleBasis,
    BesselBasis,
    GaussianBasis,
    SmoothBesselBasis,
    SphericalBesselBasis,
    gaunt_coefficients,
    legendre,
    miller_indices,
    periodic_fourier,
    polynomial_cutoff,
    reciprocal_fourier,
    sinusoidal_embedding,
    sinusoidal_frequencies,
    sinusoids,
    spherical_harmonics,
)
from torchms.models.layers.cspnet import CSPNet
from torchms.models.layers.dimenet import DimeNetPlusPlus
from torchms.models.layers.equivariant_network import EquivariantNetwork
from torchms.models.layers.gemnet import GemNetT
from torchms.models.layers.line_graph_network import LineGraphNetwork

__all__ = [
    "AngleBasis",
    "BesselBasis",
    "CSPNet",
    "DimeNetPlusPlus",
    "EquivariantNetwork",
    "GaussianBasis",
    "GemNetT",
    "LineGraphNetwork",
    "SmoothBesselBasis",
    "SphericalBesselBasis",
    "gaunt_coefficients",
    "legendre",
    "miller_indices",
    "periodic_fourier",
    "polynomial_cutoff",
    "reciprocal_fourier",
    "sinusoidal_embedding",
    "sinusoidal_frequencies",
    "sinusoids",
    "spherical_harmonics",
]
