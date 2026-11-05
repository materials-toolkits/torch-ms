# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the magnetic density from DFT,
# for inference only, see model.py and properties.py.
"""MatterGen fine-tuned on the magnetic density from DFT."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenMagneticDensity(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the magnetic density from DFT.

    The magnetic density, in the unit of the training set of MatterGen, whose
    code names none (its README asks for 0.15), is standardized by the
    statistics of this set, 0.0042 and 0.0158 in the weights, then embedded by
    sines and cosines, see properties.py. It is followed, not imposed.

    Examples:
        >>> MatterGenMagneticDensity().guidance([0.15, 0.0]).values
        (tensor([0.1500, 0.0000]),)

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([0.15] * 8))`
        generates 8 crystals meant to have a magnetic density of 0.15.
    """

    CONDITIONS = ("dft_mag_density",)
