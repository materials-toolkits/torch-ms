"""MatterSim, a universal interatomic potential, with pretrained weights.

Derived from MatterSim of Microsoft Corporation (MIT license, see LICENSE in
this directory): model.py has the energy prediction of its M3GNet model. Its
pretrained weights are listed in config.yaml, with their hyperparameters,
which config.py reads into a Config.
"""

from torchms.models.mattersim.config import Config
from torchms.models.mattersim.model import InvalidWeights, MatterSim

__all__ = ["Config", "InvalidWeights", "MatterSim"]
