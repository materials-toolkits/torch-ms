# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on a chemical system, for
# inference only: its adapter (mattergen/adapter.py,
# mattergen/common/gemnet/gemnet_ctrl.py) is in torchms.models.layers.gemnet,
# the rest in model.py; here, the conversion of the chemical systems of
# mattergen/property_embeddings.py. Names of modules and parameters follow
# MatterGen to load its weights.
"""MatterGen fine-tuned on a chemical system, the elements a crystal can have."""

from __future__ import annotations

import torch

from torchms.data.constants import symbol_to_number
from torchms.models.mattergen.diffusion import Guidance
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
        ...         guidance.chemical_systems,
        ...         torch.tensor([True, False]),
        ...     )
        >>> kept = denoised.logits[0] > -1e9
        >>> (kept.nonzero().squeeze(1) + 1).tolist()
        [16, 30]

        With pretrained weights,
        `model.sample(num_atoms, guidance=model.guidance(["Li-O"] * 8))`
        generates 8 crystals of lithium and oxygen.
    """

    CONDITIONAL = True

    def guidance(
        self, chemical_systems: list[str], scale: float | None = None
    ) -> Guidance:
        """Guidance of the sampling towards chemical systems.

        Args:
            chemical_systems: Chemical system of each crystal, its elements
                joined by dashes: "Li-O".
            scale: Factor of the guidance, the one of the configuration by
                default, 2 as in the README of MatterGen.

        Raises:
            ValueError: If an element is not known or comes after fermium.

        Examples:
            >>> guidance = MatterGenChemicalSystem().guidance(["Li-O", "Fe"])
            >>> guidance.chemical_systems.nonzero().tolist(), guidance.scale
            ([[0, 3], [0, 8], [1, 26]], 2.0)
        """
        num_elements = self.config.model.num_elements
        pairs = [
            (row, symbol)
            for row, system in enumerate(chemical_systems)
            for symbol in system.split("-")
        ]
        numbers = []
        for _, symbol in pairs:
            try:
                number = symbol_to_number(symbol)
            except KeyError:
                number = 0
            if not 1 <= number <= num_elements:
                raise ValueError(f"element {symbol} is not supported by the model")
            numbers.append(number)

        device = self.time_frequencies.device
        systems = torch.zeros((len(chemical_systems), num_elements + 1), device=device)
        rows = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=device)
        systems[rows, torch.tensor(numbers, dtype=torch.long, device=device)] = 1.0

        if scale is None:
            scale = self.config.chemical_system.guidance_scale

        return Guidance(systems, scale)
