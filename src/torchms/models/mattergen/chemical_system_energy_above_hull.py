# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on a chemical system and on the
# energy above the hull together, for inference only, see model.py and
# properties.py.
"""MatterGen fine-tuned on a chemical system and on the energy above the hull."""

from __future__ import annotations

from torchms.models.mattergen.model import MatterGen


class MatterGenChemicalSystemEnergyAboveHull(
    MatterGen
):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on a chemical system and the energy above the hull.

    The network has an adapter per property before each block, and adds their
    outputs. The chemical system restricts the elements of a conditioned
    crystal, as for MatterGenChemicalSystem; the energy above the hull, in
    eV/atom, is standardized by the statistics of the training set, 0.046 and
    0.030 eV/atom in the weights, then embedded by sines and cosines, and is
    followed, not imposed. Either can be left without condition: None in
    guidance.

    Examples:
        >>> model = MatterGenChemicalSystemEnergyAboveHull()
        >>> guidance = model.guidance(["Li-O", "Fe-O"], [0.0, 0.05])
        >>> guidance.values[0].nonzero().tolist(), guidance.values[1]
        ([[0, 3], [0, 8], [1, 8], [1, 26]], tensor([0.0000, 0.0500]))
        >>> model.guidance(None, [0.0, 0.0]).values[0] is None
        True

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance(["Li-O"] * 8, [0.0] * 8))`
        generates 8 crystals of lithium and oxygen meant to be on the hull.
    """

    CONDITIONS = ("chemical_system", "energy_above_hull")
