# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the bulk modulus predicted by a
# machine-learned potential, for inference only, see model.py and
# properties.py.
"""MatterGen fine-tuned on the bulk modulus predicted by a machine-learned potential."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenBulkModulus(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the bulk modulus, in GPa.

    The bulk modulus of the training set was predicted by a machine-learned
    potential, not computed by DFT. Unlike the other numbers, it is
    standardized after its decimal logarithm, by the statistics of the
    logarithms of the training set, 1.708 (51 GPa) and 0.364 in the weights,
    then embedded by sines and cosines, see properties.py. It must be
    positive. It is followed, not imposed.

    Examples:
        >>> MatterGenBulkModulus().guidance([400.0, 50.0]).values
        (tensor([400.,  50.]),)
        >>> MatterGenBulkModulus().guidance([0.0])
        Traceback (most recent call last):
        ...
        ValueError: values of ml_bulk_modulus must be positive: [0.0]

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([400.0] * 8))`
        generates 8 crystals meant to have a bulk modulus of 400 GPa.
    """

    CONDITIONS = ("ml_bulk_modulus",)
