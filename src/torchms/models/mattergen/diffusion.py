# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license, whose predictor and corrector are adapted
# from score_sde_pytorch under the Apache License 2.0 (see LICENSE in this
# directory).
#
# Changes: the outputs of the denoiser, the noise of a step and the condition
# of the sampling as dataclasses, and the arithmetic of a step and of the
# guidance of mattergen/diffusion/sampling and
# mattergen/common/diffusion/corruption.py as functions. See model.py.
"""What a step of the sampling of MatterGen reads and computes, without the network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float


@dataclass(frozen=True)
class Denoised:
    """What the network gives for noisy crystals.

    Args:
        x: Score of the fractional coordinates of each atom, times their
            deviation at the time of its crystal, shape (n_atoms, 3).
        lattice: Score of the lattice of each crystal, times its deviation,
            shape (n_struct, 3, 3).
        logits: Scores of the element each atom had at the time 0, from
            hydrogen, then the mask, shape (n_atoms, num_classes). The ones
            of the elements which are not allowed are lowered.
    """

    x: Float[torch.Tensor, "n_atoms 3"]
    lattice: Float[torch.Tensor, "n_struct 3 3"]
    logits: Float[torch.Tensor, "n_atoms num_classes"]


@dataclass(frozen=True)
class Guidance:
    """Condition of the sampling of a conditioned model, see MatterGenChemicalSystem.

    The scores are the ones without condition plus scale times their
    difference with the ones with the condition: 0 is the model without
    condition, 1 the model with it, more than 1 follows the condition more.

    Args:
        chemical_systems: Elements each crystal can have, 1 in the column of
            their atomic number, shape (n_struct, num_elements + 1): the
            column 0 is not read. See MatterGenChemicalSystem.guidance.
        scale: Factor of the guidance, gamma of MatterGen.
    """

    chemical_systems: Float[torch.Tensor, "n_struct num_classes"]
    scale: float


@dataclass(frozen=True)
class StepNoise:
    """Random draws of one step of the sampling.

    Args:
        corrector_x: Standard normal noise of the correction of the
            coordinates, shape (n_atoms, 3).
        corrector_lattice: Standard normal noise of the correction of the
            lattice, shape (n_struct, 3, 3), made symmetric by the step.
        predictor_x: Standard normal noise of the prediction of the
            coordinates, shape (n_atoms, 3).
        predictor_lattice: Standard normal noise of the prediction of the
            lattice, one value per crystal, shape (n_struct,): MatterGen draws
            one value and gives it to the nine elements of the lattice.
        elements: Gumbel noise of the prediction of the elements, shape
            (n_atoms, num_classes). If None, each atom takes its most probable
            element, as MatterGen returns after its last step.
    """

    corrector_x: Float[torch.Tensor, "n_atoms 3"]
    corrector_lattice: Float[torch.Tensor, "n_struct 3 3"]
    predictor_x: Float[torch.Tensor, "n_atoms 3"]
    predictor_lattice: Float[torch.Tensor, "n_struct"]
    elements: Float[torch.Tensor, "n_atoms num_classes"] | None


def symmetric_noise(
    noise: Float[torch.Tensor, "n 3 3"] | Float[torch.Tensor, "n 1 1"],
) -> Float[torch.Tensor, "n 3 3"]:
    """Symmetric matrices of standard normal noise, from independent draws.

    An element out of the diagonal is the sum of two draws over sqrt(2). Given
    one draw per matrix, shape (n, 1, 1), every element of the diagonal is
    this draw and every other one sqrt(2) times it, as in the predictor of
    MatterGen.

    Examples:
        >>> symmetric_noise(torch.ones(1, 1, 1)).round(decimals=4)
        tensor([[[1.0000, 1.4142, 1.4142],
                 [1.4142, 1.0000, 1.4142],
                 [1.4142, 1.4142, 1.0000]]])
    """
    eye = torch.eye(3, device=noise.device)[None]

    return (1 / (2**0.5)) * (1 - eye) * (noise + noise.transpose(1, 2)) + eye * noise


def ancestral_coefficients(  # pylint: disable=too-many-positional-arguments
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    alpha_s: torch.Tensor,
    sigma_s: torch.Tensor,
    at_zero: torch.Tensor,
    min_alpha_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coefficients of an ancestral step from the time t to the time s before.

    The values at the time t are alpha_t times the ones without noise plus a
    noise of deviation sigma_t. The values at the time s are sampled from the
    ones at the time t and their score:
    x_coeff x + score_coeff score + std noise.

    Args:
        alpha_t: Mean coefficient at the time t.
        sigma_t: Deviation at the time t.
        alpha_s: Mean coefficient at the time s.
        sigma_s: Deviation at the time s.
        at_zero: Where s is the time 0 or before: no noise is added.
        min_alpha_ratio: Lower bound of alpha_t / alpha_s.

    Returns:
        x_coeff, score_coeff and std.
    """
    sigma_s = torch.where(at_zero, torch.zeros_like(sigma_s), sigma_s)
    sigma2_t_given_s = sigma_t**2 - sigma_s**2 * alpha_t**2 / alpha_s**2
    std = torch.sqrt(sigma2_t_given_s) * sigma_s / sigma_t

    ratio = alpha_t / alpha_s
    if torch.any(ratio < min_alpha_ratio):
        ratio = torch.clip(ratio, min_alpha_ratio, 1)

    std = torch.where(at_zero, torch.zeros_like(std), std)

    return 1.0 / ratio, sigma2_t_given_s / ratio, std


def guided_scores(
    unconditional: Denoised, conditional: Denoised, scale: float
) -> Denoised:
    """Scores without condition, plus scale times their difference with the conditional.

    For the elements, the scores are logarithms of probabilities: the mixed
    ones are the ones of p(x)^(1 - scale) p(x | condition)^scale.
    """
    return Denoised(
        x=torch.lerp(unconditional.x, conditional.x, scale),
        lattice=torch.lerp(unconditional.lattice, conditional.lattice, scale),
        logits=torch.lerp(unconditional.logits, conditional.logits, scale),
    )
