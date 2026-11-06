"""ADiT, the All-atom Diffusion Transformer, with pretrained weights.

Derived from All-atom Diffusion Transformers of Meta Platforms, Inc. (CC BY-NC
4.0, see LICENSE in this directory: no commercial use): model.py has its
autoencoder of crystals and molecules and its sampling by flow matching,
dit.py its diffusion transformer. Their pretrained weights are listed in
config.yaml, with their hyperparameters, which config.py reads into a Config.
"""

from torchms.models.adit.config import Config
from torchms.models.adit.dit import DiT
from torchms.models.adit.model import ADiT, Decoded, InvalidWeights, Posterior

__all__ = ["ADiT", "Config", "Decoded", "DiT", "InvalidWeights", "Posterior"]
