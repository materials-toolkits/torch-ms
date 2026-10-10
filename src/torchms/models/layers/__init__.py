"""Layers shared by the models.

* basis: bases of the distances, of the angles and of the directions, and
  smooth cutoffs;
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

DimeNet++ and GemNet-T take their hyperparameters as arguments, a model gives
them the values of its config.yaml, and they read the graph attached to the
structures.
"""

from torchms.models.layers.basis import (
    AngleBasis,
    BesselBasis,
    GaussianBasis,
    SphericalBesselBasis,
    gaunt_coefficients,
    legendre,
    polynomial_cutoff,
    spherical_harmonics,
)
from torchms.models.layers.dimenet import DimeNetConfig, DimeNetPlusPlus
from torchms.models.layers.equivariant_network import EquivariantNetwork
from torchms.models.layers.gemnet import GemNetConfig, GemNetT
from torchms.models.layers.line_graph_network import LineGraphNetwork

__all__ = [
    "AngleBasis",
    "BesselBasis",
    "DimeNetConfig",
    "DimeNetPlusPlus",
    "EquivariantNetwork",
    "GaussianBasis",
    "GemNetConfig",
    "GemNetT",
    "LineGraphNetwork",
    "SphericalBesselBasis",
    "gaunt_coefficients",
    "legendre",
    "polynomial_cutoff",
    "spherical_harmonics",
]
