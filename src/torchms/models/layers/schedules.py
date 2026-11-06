# Derived from DiffCSP (https://github.com/jiaor17/DiffCSP), Copyright (c) 2023
# Rui Jiao, MIT license (see LICENSE in this directory).
#
# Changes: this file is a rewrite of the two schedules of noise of DiffCSP
# (diffcsp/pl_modules/diff_utils.py), with the cosine schedule only, and of
# the updates of a step of its sampling (diffcsp/pl_modules/diffusion.py), for
# inference. The normalization of the scores of the coordinates, which DiffCSP
# estimates from random draws, is read from the weights. Names of buffers
# follow DiffCSP to load its weights.
"""The schedules of noise of DiffCSP, and the updates of its predictor-corrector.

Two kinds of values get noise. A lattice, or anything else in a vector space,
gets normal noise with a cosine schedule of its variance (BetaScheduler).
Fractional coordinates get normal noise wrapped around the cell, of a level
which grows geometrically (SigmaScheduler). The time 0 is the value without
noise; a sampling goes from the last time down to 1.

DiffCSP and DiffCSP++ (torchms.models.diffcsp, torchms.models.diffcsp_pp) use
both schedules with the same updates: a step corrects the coordinates by a
Langevin move at the level of noise of its time, then predicts the values of
the time before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn


@dataclass(frozen=True)
class Config:
    """Hyperparameters of the two schedules.

    Args:
        timesteps: Steps of the diffusion.
        beta_offset: Offset s of the cosine schedule of the normal noise
            (arXiv:2102.09672).
        beta_min: The betas of this schedule are not smaller.
        beta_max: The betas of this schedule are not larger.
        sigma_begin: Noise of the fractional coordinates at the first time,
            in cells.
        sigma_end: Noise of the fractional coordinates at the last time.
    """

    timesteps: int
    beta_offset: float
    beta_min: float
    beta_max: float
    sigma_begin: float
    sigma_end: float


class BetaScheduler(nn.Module):  # pylint: disable=abstract-method
    """Normal noise at each time, with a cosine schedule.

    The time 0 is the value without noise: its values stand for no noise.

    Attributes:
        betas: Variance of the noise added at each time, shape (timesteps + 1,).
        alphas: 1 - betas.
        alphas_cumprod: Product of the alphas up to each time: the part of the
            variance of a noisy value which is not noise.
        sigmas: Deviation of the noise a step of the sampling adds at each time.

    Examples:
        >>> config = Config(1000, 0.008, 1e-4, 0.9999, 0.005, 0.5)
        >>> schedule = BetaScheduler(config)
        >>> schedule.alphas_cumprod.shape, float(schedule.sigmas[1])
        (torch.Size([1001]), 0.0)
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sigmas: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        timesteps, offset = config.timesteps, config.beta_offset
        steps = torch.linspace(0, timesteps, timesteps + 1)
        cumprod = (
            torch.cos((steps / timesteps + offset) / (1 + offset) * math.pi * 0.5) ** 2
        )
        cumprod = cumprod / cumprod[0]
        betas = torch.clip(
            1 - cumprod[1:] / cumprod[:-1], config.beta_min, config.beta_max
        )

        betas = F.pad(betas, (1, 0), value=0.0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        variances = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas = F.pad(torch.sqrt(variances), (1, 0), value=0.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)

    def predict(
        self,
        values: Float[torch.Tensor, "..."],
        found: Float[torch.Tensor, "..."],
        time: int,
        noise: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Values of the time before, from noisy values and the noise found in them.

        Args:
            values: Noisy values at a time.
            found: Noise the network finds in them.
            time: Time of the values, from timesteps down to 1.
            noise: Standard normal noise.
        """
        alpha = self.alphas[time]
        alpha_cumprod = self.alphas_cumprod[time]

        return (1.0 / torch.sqrt(alpha)) * (
            values - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * found
        ) + self.sigmas[time] * noise


class SigmaScheduler(nn.Module):  # pylint: disable=abstract-method
    """Noise of the fractional coordinates at each time.

    Attributes:
        sigmas: Level of the noise at each time, in cells, growing
            geometrically, with 0 at the time 0. Shape (timesteps + 1,).
        sigmas_norm: Mean of the squared score of the noise wrapped around the
            cell, at each level. A network gives the score divided by its
            square root. DiffCSP estimates it from random draws when a model is
            created: it is read from the weights, and is 1 before.

    Examples:
        >>> config = Config(1000, 0.008, 1e-4, 0.9999, 0.005, 0.5)
        >>> schedule = SigmaScheduler(config)
        >>> [round(float(schedule.sigmas[t]), 4) for t in (0, 1, 1000)]
        [0.0, 0.005, 0.5]
    """

    sigmas: torch.Tensor
    sigmas_norm: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.sigma_begin = config.sigma_begin
        sigmas = torch.exp(
            torch.linspace(
                math.log(config.sigma_begin),
                math.log(config.sigma_end),
                config.timesteps,
                dtype=torch.float64,
            )
        ).float()

        self.register_buffer("sigmas", F.pad(sigmas, (1, 0), value=0.0))
        self.register_buffer("sigmas_norm", torch.ones(config.timesteps + 1))

    def score(
        self, normalized: Float[torch.Tensor, "n_atoms 3"], time: int
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Score of the coordinates, from the one a network gives at a time.

        Args:
            normalized: Score over the square root of sigmas_norm at the time.
            time: Time of the coordinates.
        """
        return normalized * torch.sqrt(self.sigmas_norm[time])

    def langevin_step(self, time: int, step_size: float) -> torch.Tensor:
        """Step of the corrector at a time, as DiffCSP: as the square of the level.

        Args:
            time: Time of the coordinates.
            step_size: Step at the lowest level of noise.
        """
        return step_size * (self.sigmas[time] / self.sigma_begin) ** 2

    def normalized_langevin_step(self, time: int, step_size: float) -> torch.Tensor:
        """Step of the corrector at a time, as DiffCSP++: over the mean squared score.

        Args:
            time: Time of the coordinates.
            step_size: Step at the lowest level of noise, times the mean
                squared score there.
        """
        return step_size / (self.sigmas_norm[time] * self.sigma_begin**2)

    def correct(
        self,
        x: Float[torch.Tensor, "n_atoms 3"],
        score: Float[torch.Tensor, "n_atoms 3"],
        step: torch.Tensor,
        noise: Float[torch.Tensor, "n_atoms 3"],
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Langevin move of the coordinates.

        Args:
            x: Fractional coordinates at a time.
            score: Their score, see score.
            step: Step of the move, see langevin_step.
            noise: Standard normal noise.

        Returns:
            Corrected coordinates, not wrapped in the cell.
        """
        return x - step * score + torch.sqrt(2 * step) * noise

    def predict(
        self,
        x: Float[torch.Tensor, "n_atoms 3"],
        score: Float[torch.Tensor, "n_atoms 3"],
        time: int,
        noise: Float[torch.Tensor, "n_atoms 3"],
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Coordinates of the time before, from the ones of a time and their score.

        Args:
            x: Fractional coordinates at this time.
            score: Their score, see score.
            time: Time of the coordinates.
            noise: Standard normal noise.

        Returns:
            Coordinates at the time before, not wrapped in the cell.
        """
        sigma = self.sigmas[time]
        previous_sigma = self.sigmas[time - 1]

        step = sigma**2 - previous_sigma**2
        deviation = torch.sqrt(previous_sigma**2 * step / sigma**2)

        return x - step * score + deviation * noise
