# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the magnetic density from DFT
# and the HHI score together, for inference only, see model.py and
# properties.py.
"""MatterGen fine-tuned on the magnetic density from DFT and the HHI score."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenMagneticDensityHHIScore(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the magnetic density and the HHI score.

    The network has an adapter per property before each block, and adds their
    outputs. The magnetic density is the one of MatterGenMagneticDensity. The
    HHI score is the Herfindahl-Hirschman index of the production of the
    elements of a crystal, from 0 to 10000, low for elements produced in many
    countries: standardized by the statistics of the training set, 3561 and
    1548 in the weights, then embedded by sines and cosines. Both are
    followed, not imposed, and either can be left without condition: None in
    guidance.

    Examples:
        >>> model = MatterGenMagneticDensityHHIScore()
        >>> model.guidance([0.15, 0.2], [1000.0, 1500.0]).values
        (tensor([0.1500, 0.2000]), tensor([1000., 1500.]))
        >>> model.guidance([0.15], None).values[1] is None
        True

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([0.15] * 8, [1000.0] * 8))`
        generates 8 crystals meant to have a magnetic density of 0.15 and
        elements of diverse supply.
    """

    CONDITIONS = ("dft_mag_density", "hhi_score")
