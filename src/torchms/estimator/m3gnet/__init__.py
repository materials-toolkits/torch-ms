"""Energy prediction of the pretrained M3GNet model.

Derived from matgl of the Materials Virtual Lab (BSD 3-Clause license, see
LICENSE in this directory): model.py is the network. Its pretrained weights are
listed in config.yaml, with its hyperparameters.
"""

from torchms.estimator.m3gnet.model import InvalidWeights, M3gnetEnergyModel

__all__ = [
    "InvalidWeights",
    "M3gnetEnergyModel",
]
