# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on a chemical system, for
# inference only, see model.py and properties.py.
"""MatterGen fine-tuned on a chemical system, the elements a crystal can have."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenChemicalSystem(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on a chemical system, the elements a crystal can have.

    The network is the one of MatterGen with an adapter before each block,
    which reads the embedding of the chemical system of the crystal. For the
    crystals conditioned on it, the scores of the other elements are lowered
    as the ones not allowed: the crystals generated have no other element. The
    sampling mixes the scores with and without condition, see Guidance.

    Examples:
        The network on two crystals, the first one conditioned on Zn-S:

        >>> import torch
        >>> from torchms.data import Structures
        >>> model = MatterGenChemicalSystem().eval()
        >>> structures = Structures.collate(
        ...     [Structures.zincblende(), Structures.diamond()]
        ... )
        >>> structures = model.transform()(structures)
        >>> guidance = model.guidance(["Zn-S", "C"])
        >>> with torch.no_grad():
        ...     denoised = model(
        ...         structures,
        ...         torch.tensor([0.1, 0.9]),
        ...         guidance.values,
        ...         torch.tensor([True, False]),
        ...     )
        >>> kept = denoised.logits[0] > -1e9
        >>> (kept.nonzero().squeeze(1) + 1).tolist()
        [16, 30]

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance(["Li-O"] * 8))`
        generates 8 crystals of lithium and oxygen.
    """

    CONDITIONS = ("chemical_system",)
