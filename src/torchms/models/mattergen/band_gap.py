# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the band gap from DFT, for
# inference only, see scalar.py.
"""MatterGen fine-tuned on the band gap from DFT."""

from __future__ import annotations

from torchms.models.mattergen.scalar import ScalarPropertyMatterGen


class MatterGenBandGap(ScalarPropertyMatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the band gap from DFT, in eV.

    See ScalarPropertyMatterGen: the band gap is standardized by the statistics
    of the training set, 0.798 and 1.408 eV in the weights, then embedded by
    sines and cosines. It is followed, not imposed.

    Examples:
        The network on two crystals, the first one conditioned on a gap of
        3 eV:

        >>> import torch
        >>> from torchms.data import Structures
        >>> model = MatterGenBandGap().eval()
        >>> structures = Structures.collate(
        ...     [Structures.zincblende(), Structures.diamond()]
        ... )
        >>> structures = model.transform()(structures)
        >>> guidance = model.guidance([3.0, 0.0])
        >>> guidance.values, guidance.scale
        (tensor([3., 0.]), 2.0)
        >>> with torch.no_grad():
        ...     denoised = model(
        ...         structures,
        ...         torch.tensor([0.1, 0.9]),
        ...         guidance.values,
        ...         torch.tensor([True, False]),
        ...     )
        >>> denoised.lattice.shape
        torch.Size([2, 3, 3])

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance([3.0] * 8))`
        generates 8 crystals meant to have a band gap of 3 eV.
    """

    CONDITION = "dft_band_gap"
