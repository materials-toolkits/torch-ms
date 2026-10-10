# polynomial_cutoff, spherical_harmonics and BesselBasis are derived from
# orb-models (https://github.com/orbital-materials/orb-models), Copyright 2024
# Orbital Materials, Inc., licensed under the Apache License, Version 2.0 (see
# models/orb/LICENSE). They were moved here from models/orb/model.py.
# Spherical harmonics come from e3nn (https://github.com/e3nn/e3nn, MIT
# License), through orb-models.
#
# spherical_bessel_roots, spherical_bessel, legendre and SphericalBesselBasis
# are derived from matgl (https://github.com/materialsvirtuallab/matgl),
# Copyright (c) 2022, Materials Virtual Lab, BSD 3-Clause license (see
# models/m3gnet/LICENSE). They were moved here from
# models/m3gnet/model.py.
"""Bases of the distances, of the angles and of the directions of the models."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from scipy.optimize import brentq
from scipy.special import spherical_jn
from torch import nn

# Largest degree l of spherical_harmonics, which is written out degree by
# degree, and of gaunt_coefficients. It is what these functions implement, not
# a setting: a model cannot ask for another one.
MAX_DEGREE = 3

NUM_HARMONICS = (MAX_DEGREE + 1) ** 2

# degree l of each of the spherical harmonics of spherical_harmonics
DEGREES = tuple(
    degree for degree in range(MAX_DEGREE + 1) for _ in range(2 * degree + 1)
)


def polynomial_cutoff(
    distance: Float[torch.Tensor, "n"], radius: Float[torch.Tensor, "n"] | float, p: int
) -> Float[torch.Tensor, "n"]:
    """Smooth envelope going from 1 at a distance of 0 to 0 at the radius.

    Its first and second derivatives are zero at the radius. It is zero after.

    Examples:
        >>> distance = torch.tensor([0.0, 3.0, 6.0, 7.0])
        >>> polynomial_cutoff(distance, 6.0, p=4)
        tensor([1.0000, 0.6562, 0.0000, 0.0000])
    """
    ratio = distance / radius
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(ratio, p)
        + p * (p + 2.0) * torch.pow(ratio, p + 1)
        - (p * (p + 1.0) / 2.0) * torch.pow(ratio, p + 2)
    )
    return torch.where(distance < radius, envelope, torch.zeros_like(envelope))


def spherical_harmonics(
    vectors: Float[torch.Tensor, "n 3"],
) -> Float[torch.Tensor, "n 16"]:
    """Real spherical harmonics up to l = 3 of the direction of vectors.

    Convention of e3nn with the "component" normalization: the 2l+1 harmonics
    of a degree l have a squared norm of 2l+1 for any direction. A vector of
    length zero gives the harmonics of the zero vector.

    Examples:
        >>> harmonics = spherical_harmonics(torch.tensor([[0.0, 2.0, 0.0]]))
        >>> harmonics[0, :4]
        tensor([1.0000, 0.0000, 1.7321, 0.0000])
        >>> harmonics[0, 1:4].square().sum(), harmonics[0, 4:9].square().sum()
        (tensor(3.), tensor(5.))
    """
    x, y, z = F.normalize(vectors, dim=-1).unbind(dim=-1)

    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    degree_0 = torch.ones_like(x).unsqueeze(-1)
    degree_1 = torch.stack([x, y, z], dim=-1)
    degree_2 = torch.stack(
        [
            sh_2_0,
            math.sqrt(3.0) * x * y,
            y2 - 0.5 * x2z2,
            math.sqrt(3.0) * y * z,
            sh_2_4,
        ],
        dim=-1,
    )
    degree_3 = torch.stack(
        [
            math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x),
            math.sqrt(5.0) * sh_2_0 * y,
            math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x,
            0.5 * y * (2.0 * y2 - 3.0 * x2z2),
            math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2),
            math.sqrt(5.0) * sh_2_4 * y,
            math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x),
        ],
        dim=-1,
    )

    return torch.cat(
        [
            degree_0,
            math.sqrt(3.0) * degree_1,
            math.sqrt(5.0) * degree_2,
            math.sqrt(7.0) * degree_3,
        ],
        dim=-1,
    )


def sine_bessel(
    distance: Float[torch.Tensor, "n"],
    frequencies: Float[torch.Tensor, "num_bases"],
    prefactor: Float[torch.Tensor, ""] | float,
) -> Float[torch.Tensor, "n num_bases"]:
    """Radial basis prefactor sin(f r) / r of each frequency f, for r > 0.

    With the frequencies n pi / r_max and the prefactor sqrt(2 / r_max), the
    functions are orthonormal for the measure r^2 dr up to r_max.

    Examples:
        >>> frequencies = math.pi / 6.0 * torch.arange(1.0, 3.0)
        >>> sine_bessel(torch.tensor([3.0, 6.0]), frequencies, 1.0).round(decimals=4)
        tensor([[0.3333, -0.0000],
                [-0.0000, 0.0000]])
    """
    distance = distance.unsqueeze(-1)
    return prefactor * torch.sin(frequencies * distance) / distance


class BesselBasis(nn.Module):
    """Radial basis sqrt(2 / r_max) sin(n pi r / r_max) / r, for n = 1..num_bases.

    Its buffers are named as in orb-models: they are in its pretrained weights.
    """

    bessel_weights: torch.Tensor
    r_max: torch.Tensor
    prefactor: torch.Tensor

    def __init__(self, r_max: float, num_bases: int):
        super().__init__()
        self.register_buffer(
            "bessel_weights", math.pi / r_max * torch.arange(1.0, num_bases + 1.0)
        )
        self.register_buffer("r_max", torch.tensor(r_max))
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / r_max)))

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_bases"]:
        return sine_bessel(distance, self.bessel_weights, self.prefactor)


def spherical_bessel_roots(max_l: int, max_n: int) -> np.ndarray:
    """First roots of the spherical Bessel functions j_0 to j_(max_l - 1).

    The roots of j_l are between the ones of j_(l-1).

    Returns:
        Roots with shape (max_l, max_n).

    Examples:
        >>> spherical_bessel_roots(2, 2).round(4)
        array([[3.1416, 6.2832],
               [4.4934, 7.7253]])
    """
    zeros = np.arange(1, max_l + max_n + 1) * math.pi
    roots = [zeros[:max_n]]

    for degree in range(1, max_l):
        zeros = np.array(
            [
                brentq(lambda x, v=degree: spherical_jn(v, x), low, high)
                for low, high in zip(zeros[:-1], zeros[1:])
            ]
        )
        roots.append(zeros[:max_n])

    return np.array(roots)


def spherical_bessel(
    max_l: int, x: Float[torch.Tensor, "..."]
) -> list[Float[torch.Tensor, "..."]]:
    """Spherical Bessel functions j_0 to j_max_l, for x > 0.

    Examples:
        >>> j0, j1 = spherical_bessel(1, torch.tensor([math.pi / 2]))
        >>> j0, j1
        (tensor([0.6366]), tensor([0.4053]))
    """
    functions = [torch.sin(x) / x, torch.sin(x) / x**2 - torch.cos(x) / x]

    for degree in range(1, max_l):
        functions.append((2 * degree + 1) / x * functions[-1] - functions[-2])

    return functions[: max_l + 1]


def legendre(max_l: int, x: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n max_l"]:
    """Spherical harmonics Y_l0 of an angle from its cosine, for l < max_l.

    Examples:
        >>> legendre(3, torch.tensor([1.0, 0.0]))
        tensor([[ 0.2821,  0.4886,  0.6308],
                [ 0.2821,  0.0000, -0.3154]])
    """
    polynomials = [torch.ones_like(x), x]

    for degree in range(1, max_l - 1):
        polynomials.append(
            ((2 * degree + 1) * x * polynomials[-1] - degree * polynomials[-2])
            / (degree + 1)
        )

    return torch.stack(
        [
            math.sqrt((2 * degree + 1) / (4 * math.pi)) * polynomials[degree]
            for degree in range(max_l)
        ],
        dim=-1,
    )


class SphericalBesselBasis(nn.Module):
    """Radial basis j_l(z_ln r / r_max), with z_ln the roots of j_l.

    The functions are normalized on the sphere of radius r_max, and ordered by
    l then by n. Its buffers are named as in matgl: the cutoff is in its
    pretrained weights.

    Args:
        max_l: Number of degrees l, from 0.
        max_n: Number of roots, and of functions, per degree.
        r_max: Radius in Å, distances above it are taken at the radius.

    Examples:
        >>> basis = SphericalBesselBasis(max_l=3, max_n=3, r_max=5.0)
        >>> basis(torch.tensor([1.0, 5.0])).shape
        torch.Size([2, 9])
    """

    cutoff: torch.Tensor
    roots: torch.Tensor
    norm: torch.Tensor

    def __init__(self, max_l: int, max_n: int, r_max: float):
        super().__init__()
        self.max_l = max_l

        roots = torch.tensor(spherical_bessel_roots(max_l, max_n), dtype=torch.float64)
        next_degree = torch.stack(
            [
                spherical_bessel(max_l, roots[degree])[degree + 1]
                for degree in range(max_l)
            ]
        )

        self.register_buffer("cutoff", torch.tensor(r_max))
        self.register_buffer("roots", roots.float(), persistent=False)
        self.register_buffer(
            "norm",
            (math.sqrt(2.0 / r_max**3) / next_degree.abs()).float(),
            persistent=False,
        )

    def forward(
        self, distance: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n max_l*max_n"]:
        scaled = (distance.clamp(max=self.cutoff) / self.cutoff).unsqueeze(-1)

        return torch.cat(
            [
                spherical_bessel(self.max_l, scaled * self.roots[degree])[degree]
                * self.norm[degree]
                for degree in range(self.max_l)
            ],
            dim=1,
        )


class GaussianBasis(nn.Module):
    """Gaussians exp(-gamma (value - center)^2), centered on a regular grid.

    Args:
        start: First center.
        stop: Last center.
        num_basis: Number of centers.
        gamma: Inverse of the squared width of the gaussians.

    Examples:
        >>> basis = GaussianBasis(0.0, 2.0, 3, gamma=1.0)
        >>> basis(torch.tensor([1.0]))
        tensor([[0.3679, 1.0000, 0.3679]])
    """

    centers: torch.Tensor

    def __init__(self, start: float, stop: float, num_basis: int, gamma: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(start, stop, num_basis))
        self.gamma = gamma

    def forward(
        self, value: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n num_basis"]:
        return torch.exp(-self.gamma * (value.unsqueeze(1) - self.centers) ** 2)


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
    # exact quadrature on the sphere of the products of three harmonics,
    # polynomials of degree 3 MAX_DEGREE: Gauss-Legendre in cos(theta), exact
    # up to the degree 2 n - 1, and regular in phi, exact below the degree 2 n
    n = 2 * MAX_DEGREE + 2
    cosine, weight = np.polynomial.legendre.leggauss(n)
    phi = 2.0 * math.pi * np.arange(2 * n) / (2 * n)
    sine = np.sqrt(1.0 - cosine**2)

    directions = np.stack(
        [
            sine[:, None] * np.cos(phi)[None, :],
            sine[:, None] * np.sin(phi)[None, :],
            np.broadcast_to(cosine[:, None], (n, 2 * n)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weights = np.repeat(weight / (2.0 * 2 * n), 2 * n)

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
