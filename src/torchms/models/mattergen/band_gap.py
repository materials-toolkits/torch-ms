# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the band gap from DFT, for
# inference only, see model.py and properties.py.
"""MatterGen fine-tuned on the band gap from DFT."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenBandGap(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the band gap from DFT, in eV.

    The band gap is standardized by the statistics of the training set, 0.798
    and 1.408 eV in the weights, then embedded by sines and cosines as the
    time, see properties.py. It is followed, not imposed.

    Examples:
        >>> guidance = MatterGenBandGap().guidance([3.0, 0.0])
        >>> guidance.values, guidance.scale
        ((tensor([3., 0.]),), 2.0)

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([3.0] * 8))`
        generates 8 crystals meant to have a band gap of 3 eV.
    """

    CONDITIONS = ("dft_band_gap",)
