# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the model of MatterGen fine-tuned on the band gap from DFT, for
# inference only: its adapter (mattergen/adapter.py,
# mattergen/common/gemnet/gemnet_ctrl.py) is in torchms.models.layers.gemnet,
# the rest in model.py; here, the standardization of a band gap
# (StandardScalerTorch of mattergen/common/utils/data_utils.py) and its
# sinusoidal embedding (NoiseLevelEncoding of mattergen/diffusion/model_utils.py).
# Names of modules and parameters follow MatterGen to load its weights.
"""MatterGen fine-tuned on the band gap from DFT."""

from __future__ import annotations

import torch
from jaxtyping import Float

from torchms.models.layers.basis import sinusoidal_frequencies, sinusoids
from torchms.models.mattergen.config import Config
from torchms.models.mattergen.diffusion import Guidance
from torchms.models.mattergen.model import MatterGen


class MatterGenBandGap(MatterGen):  # pylint: disable=abstract-method
    """MatterGen fine-tuned on the band gap from DFT, in eV.

    The network is the one of MatterGen with an adapter before each block,
    which reads the band gap of the crystal: standardized by the mean and the
    deviation of the training set, then embedded by sines and cosines as the
    time. The band gap is not imposed: the network is trained to give crystals
    of this gap, and the guidance pushes towards it, see Guidance.

    Attributes:
        band_gap_frequencies: Frequencies of the embedding of the band gap,
            from the weights.
        band_gap_mean: Mean of the band gaps of the training set, in eV.
        band_gap_std: Their deviation, in eV.

    Examples:
        The network on two crystals, the first one conditioned on a gap of
        3 eV:

        >>> from torchms.data import Structures
        >>> model = MatterGenBandGap().eval()
        >>> structures = Structures.collate(
        ...     [Structures.zincblende(), Structures.diamond()]
        ... )
        >>> structures = model.transform()(structures)
        >>> guidance = model.guidance([3.0, 0.0])
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

    band_gap_frequencies: torch.Tensor
    band_gap_mean: torch.Tensor
    band_gap_std: torch.Tensor

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        network = self.config.model

        # the weights have their own frequencies, the ones of the time, and
        # the statistics of their training set
        self.register_buffer(
            "band_gap_frequencies",
            sinusoidal_frequencies(
                network.hidden_dim, network.time_max_period, endpoint=False
            ),
        )
        self.register_buffer("band_gap_mean", torch.zeros(1))
        self.register_buffer("band_gap_std", torch.ones(1))

    def embed_condition(
        self, values: Float[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct hidden_dim"]:
        scaled = (values - self.band_gap_mean) / self.band_gap_std

        return sinusoids(
            scaled.unsqueeze(1) * self.band_gap_frequencies, interleave=True
        )

    def guidance(
        self,
        band_gaps: list[float] | Float[torch.Tensor, "n_struct"],
        scale: float | None = None,
    ) -> Guidance:
        """Guidance of the sampling towards band gaps.

        Args:
            band_gaps: Band gap of each crystal, in eV.
            scale: Factor of the guidance, the one of the configuration by
                default, 2 as in the README of MatterGen.

        Raises:
            ValueError: If a band gap is not a finite number.

        Examples:
            >>> guidance = MatterGenBandGap().guidance([0.0, 3.5])
            >>> guidance.values, guidance.scale
            (tensor([0.0000, 3.5000]), 2.0)
        """
        values = torch.as_tensor(
            band_gaps, dtype=torch.float32, device=self.time_frequencies.device
        )
        if not values.isfinite().all():
            raise ValueError(f"band gaps must be finite numbers: {values.tolist()}")

        if scale is None:
            scale = self.config.dft_band_gap.guidance_scale

        return Guidance(values, scale)
