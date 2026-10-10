"""Energy prediction of the pretrained MACE models.

Derived from mace of the ACEsuit developers (MIT license, see LICENSE in this
directory): model.py is the network and reader.py reads its pretrained weights
without mace. They are listed in config.yaml, with its hyperparameters.
"""

from torchms.models.mace.model import InvalidWeights, MACE

__all__ = [
    "InvalidWeights",
    "MACE",
]
