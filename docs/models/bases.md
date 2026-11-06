# Bases

`torchms.models.layers.basis` has the functions which turn a distance, an angle or a
direction into the features a network reads, and the smooth cutoffs which make a term
vanish at its radius.

```python
from torchms.models.layers import BesselBasis, polynomial_cutoff

distance = torch.tensor([1.0, 3.0, 6.0])
basis = BesselBasis(r_max=6.0, num_bases=8)

basis(distance).shape                       # torch.Size([3, 8])
polynomial_cutoff(distance, 6.0, p=5)       # 1 at 0, 0 at the radius and after
```

## Distances

| Name | Functions | From | Used by |
|---|---|---|---|
| `BesselBasis` | `sqrt(2 / r_max) sin(n pi r / r_max) / r` | orb-models | orb-v3, `LineGraphNetwork`, `EquivariantNetwork` |
| `sine_bessel` | the same with given frequencies and prefactor | orb-models | MACE |
| `SphericalBesselBasis` | `j_l(z_ln r / r_max)`, with `z_ln` the roots of `j_l`, normalized | matgl | M3GNet, DimeNet++ |
| `SmoothBesselBasis` | orthogonal combinations of `sin(x) / x` at `(n + 1) pi r / r_max` and `(n + 2) pi r / r_max`, zero with two derivatives at `r_max` (arXiv:1907.02374) | MatterSim | MatterSim |
| `reciprocal_fourier` | `cos(G r)` and `sin(G r)` of the reciprocal vectors of Miller indices (`miller_indices`), at fractional coordinates | CrystalFlow | CrystalFlow |
| `GaussianBasis` | gaussians centered on a regular grid | — | ALIGNN, GemNet-T |
| `polynomial_cutoff` | envelope from 1 at a distance of 0 to 0 at the radius, with two derivatives zero there | orb-models | every model |

`spherical_bessel_roots` and `spherical_bessel` give the roots and the values of the spherical
Bessel functions. The roots are found once by bisection, in double precision; the functions by recurrence in
torch, which loses digits in single precision for a high degree and a short distance (see
[CDVAE](cdvae.md#comparison-with-the-original)). [MatterSim](mattersim.md) writes the
functions of degree 0 to 3 in closed form instead, from the same roots.

## Angles and directions

| Name | Functions | From | Used by |
|---|---|---|---|
| `legendre` | spherical harmonics `Y_l0` of an angle, from its cosine | matgl | M3GNet, DimeNet++, GemNet-T |
| `AngleBasis` | Fourier basis `cos(k angle)`, and `sin(k angle)` on demand | — | `LineGraphNetwork` |
| `spherical_harmonics` | the 16 real spherical harmonics of a direction, `l <= 3` | e3nn, through orb-models | orb-v3, MACE, `EquivariantNetwork` |
| `gaunt_coefficients` | couplings of two of these harmonics into a third | — | `EquivariantNetwork` |

The Gaunt coefficients are computed in the library by an exact quadrature on the sphere.

## Fractional coordinates and time

| Name | Functions | From | Used by |
|---|---|---|---|
| `periodic_fourier` | `sin(2 pi k x)` and `cos(2 pi k x)` of each coordinate, of period 1 | DiffCSP | CSPNet |
| `sinusoidal_embedding` | sines and cosines of a time at geometric frequencies | DiffCSP | DiffCSP |
| `sinusoidal_frequencies` | these frequencies, from 1 down to `1 / max_period`, with or without this last one | DiffCSP, MatterGen | MatterGen |
| `sinusoids` | sines and cosines of angles, the sines first or each sine before its cosine | DiffCSP, MatterGen | MatterGen |

`periodic_fourier` is for fractional coordinates and their differences: it does not change
when an atom moves by a whole number of cells.

## Licenses

The functions derived from orb-models (Apache-2.0), from matgl (BSD 3-Clause) and from
DiffCSP (MIT) are listed
in the header of the module, with the license of each in the directory of its model.
