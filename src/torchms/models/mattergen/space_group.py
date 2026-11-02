# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on a space group, for inference
# only: its adapter (mattergen/adapter.py, mattergen/common/gemnet/gemnet_ctrl.py)
# is in torchms.models.layers.gemnet, the rest in model.py; here, the embedding
# of a space group of mattergen/property_embeddings.py. Names of modules and
# parameters follow MatterGen to load its weights.
"""MatterGen fine-tuned on a space group."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import nn

from torchms.models.mattergen.config import Config
from torchms.models.mattergen.diffusion import Guidance
from torchms.models.mattergen.model import MatterGen

# space groups of the international tables, from 1
NUM_SPACE_GROUPS = 230


class MatterGenSpaceGroup(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on a space group.

    The network is the one of MatterGen with an adapter before each block,
    which reads a learned vector of the space group of the crystal. Nothing
    else depends on it: the space group is not imposed on the crystals
    generated, the network is only trained to give crystals of this group. The
    sampling mixes the scores with and without condition, see Guidance.

    Examples:
        The network on two crystals, the first one conditioned on the space
        group 216 of zincblende:

        >>> from torchms.data import Structures
        >>> model = MatterGenSpaceGroup().eval()
        >>> structures = Structures.collate(
        ...     [Structures.zincblende(), Structures.diamond()]
        ... )
        >>> structures = model.transform()(structures)
        >>> guidance = model.guidance([216, 227])
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
        `model.sample(num_atoms, guidance=model.guidance([225] * 8))`
        generates 8 crystals meant to be of the space group Fm-3m.
    """

    CONDITION = "space_group"

    def __init__(self, config: Config | None = None):
        super().__init__(config)

        # a learned vector per space group, from 1
        self.space_group_embedding = nn.Embedding(
            NUM_SPACE_GROUPS, self.config.model.hidden_dim
        )

    def embed_condition(
        self, values: Int[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct hidden_dim"]:
        return self.space_group_embedding(values.long() - 1)

    def guidance(
        self,
        space_groups: list[int] | Int[torch.Tensor, "n_struct"],
        scale: float | None = None,
    ) -> Guidance:
        """Guidance of the sampling towards space groups.

        Args:
            space_groups: Number of the space group of each crystal, from 1 to
                230, in the international tables.
            scale: Factor of the guidance, the one of the configuration by
                default, 2 as for the chemical systems.

        Raises:
            ValueError: If a number is not one of a space group.

        Examples:
            >>> guidance = MatterGenSpaceGroup().guidance([225, 194])
            >>> guidance.values, guidance.scale
            (tensor([225, 194]), 2.0)
        """
        values = torch.as_tensor(
            space_groups, dtype=torch.long, device=self.time_frequencies.device
        )
        outside = (values < 1) | (values > NUM_SPACE_GROUPS)
        if outside.any():
            raise ValueError(f"no space group {values[outside].tolist()}")

        if scale is None:
            scale = self.config.space_group.guidance_scale

        return Guidance(values, scale)
