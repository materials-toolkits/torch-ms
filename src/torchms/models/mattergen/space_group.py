# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on a space group, for inference
# only, see model.py and properties.py.
"""MatterGen fine-tuned on a space group."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenSpaceGroup(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on a space group, from 1 to 230.

    The network is the one of MatterGen with an adapter before each block,
    which reads a learned vector of the space group of the crystal. Nothing
    else depends on it: the space group is not imposed on the crystals
    generated, the network is only trained to give crystals of this group. The
    sampling mixes the scores with and without condition, see Guidance.

    Examples:
        >>> guidance = MatterGenSpaceGroup().guidance([225, 194])
        >>> guidance.values, guidance.scale
        ((tensor([225, 194]),), 2.0)

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([225] * 8))`
        generates 8 crystals meant to be of the space group Fm-3m.
    """

    CONDITIONS = ("space_group",)
