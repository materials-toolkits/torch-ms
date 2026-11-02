# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: what the models of MatterGen fine-tuned on a number share, for
# inference only: the standardization of the number (StandardScalerTorch of
# mattergen/common/utils/data_utils.py) and its sinusoidal embedding
# (NoiseLevelEncoding of mattergen/diffusion/model_utils.py). Their adapter is
# in torchms.models.layers.gemnet, the rest in model.py. Names of modules and
# parameters follow MatterGen to load its weights.
"""What the models of MatterGen fine-tuned on a number share."""

from __future__ import annotations

import torch
from jaxtyping import Float

from torchms.models.layers.basis import sinusoidal_frequencies, sinusoids
from torchms.models.mattergen.config import Config
from torchms.models.mattergen.diffusion import Guidance
from torchms.models.mattergen.model import MatterGen


class ScalarPropertyMatterGen(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on a number per crystal: a band gap, a magnetic density.

    The network is the one of MatterGen with an adapter before each block,
    which reads the number of the crystal: standardized by the mean and the
    deviation of the training set, then embedded by sines and cosines as the
    time. The number is not imposed: the network is trained to give crystals
    with it, and the guidance pushes towards it, see Guidance.

    Attributes:
        property_frequencies: Frequencies of the embedding of the number, from
            the weights: the ones of the time.
        property_mean: Mean of the numbers of the training set.
        property_std: Their deviation.
    """

    property_frequencies: torch.Tensor
    property_mean: torch.Tensor
    property_std: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model

        # the weights have their own frequencies, the ones of the time, and
        # the statistics of their training set
        self.register_buffer(
            "property_frequencies",
            sinusoidal_frequencies(
                network.hidden_dim, network.time_max_period, endpoint=False
            ),
        )
        self.register_buffer("property_mean", torch.zeros(1))
        self.register_buffer("property_std", torch.ones(1))

    def embed_condition(
        self, values: Float[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct hidden_dim"]:
        scaled = (values - self.property_mean) / self.property_std

        return sinusoids(
            scaled.unsqueeze(1) * self.property_frequencies, interleave=True
        )

    def guidance(
        self,
        values: list[float] | Float[torch.Tensor, "n_struct"],
        scale: float | None = None,
    ) -> Guidance:
        """Guidance of the sampling towards a number per crystal.

        Args:
            values: Number of each crystal, in the unit of the training set
                of MatterGen.
            scale: Factor of the guidance, the one of the configuration by
                default, 2 as in the README of MatterGen.

        Raises:
            ValueError: If a value is not a finite number.
        """
        assert self.CONDITION is not None
        numbers = torch.as_tensor(
            values, dtype=torch.float32, device=self.time_frequencies.device
        )
        if not numbers.isfinite().all():
            raise ValueError(f"values must be finite numbers: {numbers.tolist()}")

        if scale is None:
            scale = self.config.condition(self.CONDITION).guidance_scale

        return Guidance(numbers, scale)
