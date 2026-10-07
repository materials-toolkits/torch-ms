"""Bases of the distances, of the angles and of the directions of the models."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from pycrystalgen.estimator.orb.model import spherical_harmonics

NUM_HARMONICS = 16

# degree l of each of the 16 spherical harmonics of spherical_harmonics
DEGREES = tuple(degree for degree in range(4) for _ in range(2 * degree + 1))


def mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    """Network with one hidden layer, a normalization and SiLU activations."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_size),
        nn.LayerNorm(output_size),
        nn.SiLU(),
    )


class AngleBasis(nn.Module):
    """Fourier basis of an angle: cos(k angle) for k < num_basis.

    With sine=True, sin(k angle) for 0 < k < num_basis is added. The sign of a
    dihedral angle changes in a mirror: its sines distinguish a crystal from
    its mirror image, its cosines do not.

    Examples:
        >>> basis = AngleBasis(3)
        >>> basis(torch.tensor([0.0, math.pi])).round()  # 1, cos, cos of twice
        tensor([[ 1.,  1.,  1.],
                [ 1., -1.,  1.]])
        >>> AngleBasis(3, sine=True).num_features
        5
    """

    frequencies: torch.Tensor

    def __init__(self, num_basis: int, sine: bool = False):
        super().__init__()
        self.sine = sine
        self.num_features = 2 * num_basis - 1 if sine else num_basis
        self.register_buffer(
            "frequencies",
            torch.arange(num_basis, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, angle: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_features"]:
        phase = angle.unsqueeze(-1) * self.frequencies

        if self.sine:
            return torch.cat([torch.cos(phase), torch.sin(phase[:, 1:])], dim=-1)

        return torch.cos(phase)


@functools.lru_cache(maxsize=None)
def _gaunt() -> np.ndarray:
    """Gaunt coefficients in double precision, computed once."""
    # exact quadrature of polynomials of degree 9 on the sphere:
    # Gauss-Legendre in cos(theta), regular in phi
    cosine, weight = np.polynomial.legendre.leggauss(8)
    phi = 2.0 * math.pi * np.arange(16) / 16
    sine = np.sqrt(1.0 - cosine**2)

    directions = np.stack(
        [
            sine[:, None] * np.cos(phi)[None, :],
            sine[:, None] * np.sin(phi)[None, :],
            np.broadcast_to(cosine[:, None], (8, 16)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weights = np.repeat(weight / (2.0 * 16), 16)

    harmonics = spherical_harmonics(torch.from_numpy(directions)).numpy()

    return np.einsum("p,pi,pj,pk->ijk", weights, harmonics, harmonics, harmonics)


def gaunt_coefficients() -> Float[torch.Tensor, "16 16 16"]:
    """Mean over the sphere of the products of three spherical harmonics.

    The harmonics are the ones of spherical_harmonics, of degree l up to 3.
    The product of two functions on the sphere with coefficients a and b has
    the coefficients sum_ij G[i, j, k] a[i] b[j], up to l = 3: G couples two
    sets of harmonics to a third one. G[i, j, k] is zero unless the degrees
    of i, j and k can form a triangle and have an even sum.

    Returns:
        Coefficients, symmetric in their three indices, shape (16, 16, 16).

    Examples:
        >>> gaunt = gaunt_coefficients()
        >>> gaunt.shape
        torch.Size([16, 16, 16])
        >>> torch.allclose(gaunt[0], torch.eye(16), atol=1e-6)  # product with 1
        True
    """
    return torch.from_numpy(_gaunt()).float()
