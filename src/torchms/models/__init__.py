"""Models of crystal structures and of molecules.

Supervised models predict a property of a structure: OrbEnergyModel,
MaceEnergyModel, M3gnetEnergyModel and ChgnetEnergyModel its total energy,
AlignnModel its formation energy.

Generative models generate structures: CdvaeModel.

All of them are pretrained models of other projects, rewritten with the graphs
of torchms. Each one has a directory with its network (model.py), its
hyperparameters and the files of its weights (config.yaml), and the license of
its project.

torchms.models.layers has what these models share and what a new model can be
built from: bases of distances and of angles, DimeNet++, GemNet-T, and two
networks of the energy with terms on more than two atoms. torchms.models.estimator
has the estimator of the energy above the hull, and what loads pretrained
weights.
"""

from torchms.models.alignn import AlignnModel
from torchms.models.cdvae import CdvaeModel
from torchms.models.chgnet import ChgnetEnergyModel
from torchms.models.m3gnet import M3gnetEnergyModel
from torchms.models.mace import MaceEnergyModel
from torchms.models.orb import OrbEnergyModel

__all__ = [
    "AlignnModel",
    "CdvaeModel",
    "ChgnetEnergyModel",
    "M3gnetEnergyModel",
    "MaceEnergyModel",
    "OrbEnergyModel",
]
