"""Models of crystal structures and of molecules.

Supervised models predict a property of a structure: OrbV3, MACE, M3GNet and
CHGNet its total energy, ALIGNN its formation energy.

Generative models generate structures: CDVAE, DiffCSPGeneration and
MatterGen, MatterGenChemicalSystem for given elements, MatterGenSpaceGroup,
MatterGenBandGap, MatterGenMagneticDensity and MatterGenBulkModulus towards a
space group, a band gap, a magnetic density or a bulk modulus,
MatterGenChemicalSystemEnergyAboveHull for given elements near the hull,
MatterGenMagneticDensityHHIScore towards a magnetic density with elements of
diverse supply, and DiffCSP for a given composition.

All of them are pretrained models of other projects, rewritten with the graphs
of torchms. Each one has a directory with its network (model.py), its
hyperparameters and the files of its weights (config.yaml), and the license of
its project.

torchms.models.layers has what these models share and what a new model can be
built from: bases of distances and of angles, DimeNet++, GemNet-T, CSPNet, and two
networks of the energy with terms on more than two atoms. torchms.models.estimator
has the estimator of the energy above the hull, and what loads pretrained
weights.
"""

from torchms.models.alignn import ALIGNN
from torchms.models.cdvae import CDVAE
from torchms.models.chgnet import CHGNet
from torchms.models.diffcsp import DiffCSP, DiffCSPGeneration
from torchms.models.m3gnet import M3GNet
from torchms.models.mace import MACE
from torchms.models.mattergen import (
    MatterGen,
    MatterGenBandGap,
    MatterGenBulkModulus,
    MatterGenChemicalSystem,
    MatterGenChemicalSystemEnergyAboveHull,
    MatterGenMagneticDensity,
    MatterGenMagneticDensityHHIScore,
    MatterGenSpaceGroup,
)
from torchms.models.orb import OrbV3

__all__ = [
    "ALIGNN",
    "CDVAE",
    "CHGNet",
    "DiffCSP",
    "DiffCSPGeneration",
    "M3GNet",
    "MACE",
    "MatterGen",
    "MatterGenBandGap",
    "MatterGenBulkModulus",
    "MatterGenChemicalSystem",
    "MatterGenChemicalSystemEnergyAboveHull",
    "MatterGenMagneticDensity",
    "MatterGenMagneticDensityHHIScore",
    "MatterGenSpaceGroup",
    "OrbV3",
]
