"""The bases of torchms.models.layers.basis: values, symmetries, originals.

Their comparison with the original codes, in the pretrained models which use
them, is in the tests of the layers built on them: the gaussians, the cutoff
and the Legendre polynomials of GemNet-T (test_gemnet.py), the spherical Bessel
functions of DimeNet++ (test_dimenet.py), the basis of the coordinates of
CSPNet (test_cspnet.py). Below: the roots of the spherical Bessel functions of
matgl and the embeddings of the time of MatterGen and DiffCSP.
"""

import math

import numpy as np
import torch
from scipy.special import spherical_jn

from helpers.compare import assert_close
from helpers.paths import DATA
from helpers.references import load_reference
from torchms.data.augmentation import random_orthogonal
from torchms.models.layers import AngleBasis, gaunt_coefficients
from torchms.models.layers.basis import (
    DEGREES,
    BesselBasis,
    SphericalBesselBasis,
    legendre,
    periodic_fourier,
    polynomial_cutoff,
    sinusoidal_embedding,
    sinusoidal_frequencies,
    sinusoids,
    spherical_bessel,
    spherical_bessel_roots,
    spherical_harmonics,
)
from torchms.models.m3gnet import Config as M3GNetConfig


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# cutoff and harmonics, of orb-v3 and MACE


def test_polynomial_cutoff():
    distance = torch.linspace(0.0, 8.0, 801, dtype=torch.float64)
    envelope = polynomial_cutoff(distance, 6.0, p=4)

    assert envelope[0] == 1.0
    assert (envelope[distance >= 6.0] == 0.0).all()
    assert ((envelope >= 0.0) & (envelope <= 1.0)).all()
    assert (envelope[1:] <= envelope[:-1] + 1e-12).all()
    # smooth at the radius: value and slope go to zero
    assert envelope[599] < 1e-5

    radius = torch.tensor([2.0, 4.0])
    per_edge = polynomial_cutoff(torch.tensor([3.0, 3.0]), radius, p=6)
    assert per_edge[0] == 0.0 and 0.0 < per_edge[1] < 1.0


def test_spherical_harmonics_norms_and_parity():
    generator = torch.Generator().manual_seed(0)
    vectors = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    harmonics = spherical_harmonics(vectors)

    assert harmonics.shape == (50, 16)
    start = 0
    for degree in range(4):
        block = harmonics[:, start : start + 2 * degree + 1]
        expected = torch.full((50,), 2.0 * degree + 1, dtype=torch.float64)
        assert torch.allclose(block.square().sum(dim=1), expected)
        # parity of the degree
        opposite = spherical_harmonics(-vectors)[:, start : start + 2 * degree + 1]
        assert torch.allclose(opposite, (-1) ** degree * block)
        start += 2 * degree + 1

    # only the direction matters
    assert torch.allclose(spherical_harmonics(3.0 * vectors), harmonics)
    # degree 1 is the direction, in the order x, y, z
    unit = torch.nn.functional.normalize(vectors, dim=1)
    assert torch.allclose(harmonics[:, 1:4], math.sqrt(3.0) * unit)


def test_spherical_harmonics_rotation():
    generator = torch.Generator().manual_seed(1)
    vectors = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )[0]

    harmonics = spherical_harmonics(vectors)
    rotated = spherical_harmonics(vectors @ rotation)

    # a rotation mixes the harmonics of a degree: scalar products are kept
    start = 0
    for degree in range(4):
        block = slice(start, start + 2 * degree + 1)
        assert torch.allclose(
            harmonics[:, block] @ harmonics[:, block].T,
            rotated[:, block] @ rotated[:, block].T,
            atol=1e-10,
        )
        start += 2 * degree + 1


def test_bessel_basis():
    basis = BesselBasis(r_max=6.0, num_bases=8)
    distance = torch.tensor([1.0, 3.0, 6.0])
    values = basis(distance)

    assert values.shape == (3, 8)
    expected = math.sqrt(2.0 / 6.0) * math.sin(2 * math.pi * 3.0 / 6.0) / 3.0
    assert torch.isclose(values[1, 1], torch.tensor(expected), atol=1e-6)
    # every function is zero at the radius
    assert values[2].abs().max() < 1e-6
    assert set(basis.state_dict()) == {"bessel_weights", "r_max", "prefactor"}


# spherical Bessel functions and Legendre polynomials, of matgl


def test_spherical_bessel_roots():
    roots = spherical_bessel_roots(4, 5)

    assert roots.shape == (4, 5)
    for degree in range(4):
        assert np.allclose(spherical_jn(degree, roots[degree]), 0.0, atol=1e-10)
        assert (np.diff(roots[degree]) > 0).all()
    # the first roots, not later ones
    assert np.allclose(roots[:, 0], [math.pi, 4.493409, 5.763459, 6.987932], atol=1e-5)


def test_spherical_bessel():
    x = torch.linspace(0.3, 15.0, 50, dtype=torch.float64)
    functions = spherical_bessel(4, x)

    assert len(functions) == 5
    for degree, values in enumerate(functions):
        assert np.allclose(values.numpy(), spherical_jn(degree, x.numpy()), atol=1e-9)


def test_legendre():
    cosine = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    harmonics = legendre(3, cosine)

    assert harmonics.shape == (9, 3)
    expected = torch.stack(
        [
            torch.full_like(cosine, math.sqrt(1 / (4 * math.pi))),
            math.sqrt(3 / (4 * math.pi)) * cosine,
            math.sqrt(5 / (16 * math.pi)) * (3 * cosine**2 - 1),
        ],
        dim=-1,
    )
    assert torch.allclose(harmonics, expected)


def test_radial_basis_is_orthonormal():
    config = M3GNetConfig.load()
    basis = SphericalBesselBasis(
        config.model.max_l, config.model.max_n, config.graph.radius
    ).double()
    distance = torch.linspace(1e-4, config.graph.radius, 20001, dtype=torch.float64)
    values = basis(distance)

    assert values.shape == (len(distance), config.model.max_l * config.model.max_n)

    # functions of the same degree are orthonormal for the measure r^2 dr
    weight = distance**2 * (distance[1] - distance[0])
    for degree in range(config.model.max_l):
        block = values[
            :, degree * config.model.max_n : (degree + 1) * config.model.max_n
        ]
        gram = block.T @ (block * weight.unsqueeze(-1))
        assert torch.allclose(
            gram, torch.eye(config.model.max_n, dtype=torch.float64), atol=1e-3
        )


def test_bessel_roots_match_matgl():
    with np.load(DATA / "m3gnet_reference.npz") as data:
        expected = data["bessel_roots"]

    assert np.allclose(spherical_bessel_roots(4, 4), expected, atol=1e-5)


# the bases of DiffCSP


def test_periodic_fourier():
    x = torch.rand(7, 3, generator=generator())
    basis = periodic_fourier(x, 5)

    assert basis.shape == (7, 30)
    # sines then cosines, by coordinate then by frequency
    assert torch.allclose(basis[:, 15::5], torch.ones(7, 3))
    assert torch.allclose(basis[:, 6], torch.sin(2 * torch.pi * x[:, 1]), atol=1e-6)
    assert torch.allclose(basis[:, 27], torch.cos(4 * torch.pi * x[:, 2]), atol=1e-6)

    # of period 1
    shift = torch.tensor([[1.0, -2.0, 3.0]])
    assert torch.allclose(periodic_fourier(x + shift, 5), basis, atol=1e-4)

    assert periodic_fourier(x.double(), 5).dtype == torch.float64


def test_sinusoidal_embedding():
    time = torch.tensor([0, 1, 500])
    embedding = sinusoidal_embedding(time, 16, 10000.0)

    assert embedding.shape == (3, 16)
    assert torch.equal(embedding[0], torch.tensor([0.0] * 8 + [1.0] * 8))
    # frequencies from 1 to 1 / max_period
    assert torch.allclose(embedding[1, [0, 7]], torch.tensor([1.0, 1e-4]).sin())
    assert torch.allclose(embedding[:, :8] ** 2 + embedding[:, 8:] ** 2, torch.ones(8))
    assert torch.equal(sinusoidal_embedding(time.float(), 16, 10000.0), embedding)


# the bases of the networks to train


def test_angle_basis():
    angle = torch.linspace(-3.0, 3.0, 13)

    cosines = AngleBasis(4)(angle)
    assert cosines.shape == (13, 4)
    assert torch.allclose(cosines, AngleBasis(4)(-angle), atol=1e-6)

    both = AngleBasis(4, sine=True)(angle)
    assert both.shape == (13, 7)
    assert torch.allclose(both[:, :4], cosines)
    assert torch.allclose(
        both[:, 4:], -AngleBasis(4, sine=True)(-angle)[:, 4:], atol=1e-6
    )


def test_gaunt_coefficients_selection_rules():
    gaunt = gaunt_coefficients().double()

    # symmetric in its three indices
    for permutation in [(1, 0, 2), (0, 2, 1), (2, 1, 0)]:
        assert torch.allclose(gaunt, gaunt.permute(permutation), atol=1e-7)

    # the product with the constant harmonic is the identity
    assert torch.allclose(gaunt[0], torch.eye(16, dtype=torch.float64), atol=1e-6)

    # zero unless the degrees form a triangle and have an even sum
    degree = torch.tensor(DEGREES)
    l1, l2, l3 = degree[:, None, None], degree[None, :, None], degree[None, None, :]
    allowed = ((l1 + l2 + l3) % 2 == 0) & (l3 <= l1 + l2) & (l3 >= (l1 - l2).abs())
    assert gaunt[~allowed].abs().max() < 1e-6
    assert gaunt[allowed].abs().max() > 0.1


def test_gaunt_coefficients_are_equivariant():
    gaunt = gaunt_coefficients().double()
    directions = torch.randn(64, 3, dtype=torch.float64, generator=generator())
    # orthogonal in double precision
    rotation = torch.linalg.qr(
        random_orthogonal(1, torch.device("cpu"), generator())[0].double()
    ).Q

    # a function on the sphere as coefficients of harmonics, and its rotation:
    # the harmonics of rotated directions are a linear map of the harmonics
    harmonics = spherical_harmonics(directions)
    rotated = spherical_harmonics(directions @ rotation.T)
    wigner = torch.linalg.lstsq(
        harmonics, rotated
    ).solution  # rotated = harmonics @ wigner
    assert torch.allclose(harmonics @ wigner, rotated, atol=1e-8)

    a = torch.randn(16, dtype=torch.float64, generator=generator(1))
    b = torch.randn(16, dtype=torch.float64, generator=generator(2))

    product = torch.einsum("ijk,i,j->k", gaunt, a, b)
    product_of_rotated = torch.einsum("ijk,i,j->k", gaunt, a @ wigner, b @ wigner)

    assert torch.allclose(product_of_rotated, product @ wigner, atol=1e-8)


# the embeddings of the time, against what the originals computed


def test_sinusoids_match_mattergen():
    """NoiseLevelEncoding of MatterGen: a sine then a cosine per frequency.

    Its frequencies are computed here; the ones in the weights of MatterGen
    differ from them by one bit (see MatterGen.time_frequencies).
    """
    reference = load_reference("layers/mattergen_gemnet.npz")
    frequencies = sinusoidal_frequencies(512, 10000.0, endpoint=False)

    found = sinusoids(reference["times"].float().unsqueeze(1) * frequencies, True)
    assert_close(found, reference.call("network").input("z"), name="time of MatterGen")


def test_sinusoidal_embedding_matches_diffcsp():
    """The embedding of the time steps of DiffCSP: sines, then cosines."""
    reference = load_reference("layers/diffcsp_cspnet.npz")

    found = sinusoidal_embedding(reference["times"], 256, 10000.0)
    assert_close(found, reference.call("network").input("t"), name="time of DiffCSP")
