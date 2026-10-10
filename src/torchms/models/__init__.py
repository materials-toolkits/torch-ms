"""Trainable models of the energy of crystal structures.

Two families of models with terms on more than two atoms:

* LineGraphNetwork lists the triplets and the quadruplets of atoms and uses
  their angles, as ALIGNN, M3GNet, CHGNet or DPA-3.
* EquivariantNetwork only uses pairs of atoms and takes products of sums of
  spherical harmonics, as MACE.

Both predict an energy which does not depend on the description of the crystal
and which grows with the size of its cell.

Models of other projects rewritten with the graphs of torchms, with their
pretrained weights, have a directory each:

* CdvaeModel, in torchms.models.cdvae, generates crystal structures.
"""

from torchms.models.basis import AngleBasis, gaunt_coefficients
from torchms.models.cdvae import CdvaeModel
from torchms.models.equivariant_network import EquivariantNetwork
from torchms.models.line_graph_network import LineGraphNetwork

__all__ = [
    "AngleBasis",
    "CdvaeModel",
    "EquivariantNetwork",
    "LineGraphNetwork",
    "gaunt_coefficients",
]
