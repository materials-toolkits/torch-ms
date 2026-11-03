# Classifier-free guidance

The models of [MatterGen](../models/mattergen.md) fine-tuned on a property sample crystals
towards a value of it by classifier-free guidance. This page says how torchms does it and how
far a value can be from the ones of the training set.

## Mixing two scores

The fine-tuned network is trained with and without its properties: a crystal without condition
skips the adapters (`gemnet.Condition`). At each step of the sampling, it is called twice on the
same noisy crystals, once without the properties and once with them, and the two scores are
mixed:

```
score = unconditional + scale (conditional - unconditional)
```

for the coordinates, the lattice and the scores of the elements (`guided_scores` in
`models/mattergen/diffusion.py`). These are the scores of p(x)^(1 − scale) p(x | condition)^scale:
0 is the model without condition, 1 the model with it, more than 1 follows the condition more.
No classifier is trained; the network is its own. The factor is 2 by default, the value of the
README of MatterGen, in the `config.yaml` of each property.

| | |
|---|---|
| `model.guidance(*values, scale=None)` | a `Guidance`: one value per crystal for each property of `CONDITIONS`, in their order, and the factor |
| `None` for a property | the sampling is not conditioned on it: `guidance(None, [0.0] * 8)` follows the energy above the hull alone |
| scale 0 or 1 | one call of the network, not two |
| `model.scores(structures, times, guidance)` | the mixed scores |

The two calls make a guided sampling twice as long as an unguided one: 2000 steps of
prediction and correction, 4000 calls of the network. Without a `Guidance`, `sample` samples the
fine-tuned model without condition, which is not the base model `MatterGen`.

## Followed or imposed

The guidance only pushes the scores: a value is **followed, not imposed**. Nothing in the
sampling checks the space group, the band gap, the magnetic density or the energy above the
hull of a crystal.

The chemical system is the exception. In the call with the condition, the scores of the
elements outside the system of a crystal are lowered by 1e10, as the ones of the elements the
model does not allow. The mix keeps them below −1e10 for any factor of 1 or more, so the atoms
can only take the elements of the system.

## The range of a number

A number (band gap, magnetic density, energy above the hull, bulk modulus) is standardized
by the mean and the deviation of the training set, which are in the weights, then embedded
by sines and cosines as the time. The bulk modulus is standardized after its decimal
logarithm, as in MatterGen: its statistics are the ones of the logarithms, and it must be
positive. A value far from the training set gives the adapters inputs they never
saw, and the guidance stops meaning anything.

| Property | Mean | Deviation | Unit |
|---|---|---|---|
| Band gap | 0.798 | 1.408 | eV |
| Magnetic density | 0.0042 | 0.0158 | none given by MatterGen |
| Energy above the hull | 0.046 | 0.030 | eV/atom |
| Bulk modulus, decimal logarithm | 1.708 (51 GPa) | 0.364 (a factor of 2.3) | GPa |

For the energy above the hull, this was measured. `MatterGenChemicalSystemEnergyAboveHull`
sampled 8 crystals of 8 chemical systems at 0, 0.2, 0.5 and 1 eV/atom, with a generator of
seed 0: the same numbers of atoms and the same noise for every value, a factor of 2, about
280 s per value on a GPU of 8 GB. Their energy above the hull was estimated by
`EnergyEstimator`, without relaxation.

| Target (eV/atom) | Deviations from the mean | Median | Mean |
|---|---|---|---|
| 0 | −1.5 | 0.057 | 0.147 |
| 0.2 | 5 | 0.116 | 0.177 |
| 0.5 | 15 | **0.210** | 0.318 |
| 1 | 32 | 0.080 | 0.309 |

| System | 0 | 0.2 | 0.5 | 1 |
|---|---|---|---|---|
| Li-O | Li2O4 0.080 | Li4O2 0.062 | Li4O2 0.274 | Li4O2 0.062 |
| Na-Cl | Na2Cl6 0.033 | Na4Cl4 0.088 | Na5Cl3 0.190 | Na2Cl6 0.051 |
| Ga-N | Ga4N4 0.008 | Ga4N4 0.147 | Ga4N4 0.179 | Ga4N4 0.058 |
| Si-C | Si4C4 −0.002 | Si4C4 0.090 | Si5C3 0.640 | Si6C2 0.250 |
| Ti-O | Ti6O6 0.289 | Ti8O4 0.112 | Ti8O4 0.010 | Ti8O4 0.011 |
| Mg-Al-O | Mg4Al2O8 0.212 | Mg2Al4O8 0.189 | Mg3Al3O8 0.199 | Mg2Al4O8 0.325 |
| Cu-Zn-Sn-S | Cu4Zn2Sn2S8 0.012 | Cu4Zn2Sn3S7 0.121 | Cu3Zn4Sn2S7 0.222 | Cu3Zn3Sn2S8 0.097 |
| Li-Fe-P-O | LiFe3P2O8 0.542 | Li3Fe3PO7 0.607 | Li2Fe4PO7 0.826 | Li2Fe6PO5 1.615 |

Up to 0.5 eV/atom the median grows with the target, 0.057, 0.116, 0.210, and the crystals
move away from their simple compositions (Na5Cl3, Si5C3). At 1 eV/atom it falls back to
0.080: six systems of eight are lower than at 0.5, and the mean stays at 0.3 by one crystal of
Li-Fe-P-O at 1.6. The model follows the energy up to about 0.2 to 0.5 eV/atom, 5 to 15
deviations from its mean, and not beyond. Every atom of the 32 crystals has an element of its
system, whatever the target.

Eight crystals per value, single draws, unrelaxed energies: a direction, not a measure.

## Limits

- The factor is the one of the README of MatterGen for its chemical systems and its magnetic
  densities; it gives none for the space groups and the band gap, which take the same 2.
- A crystal follows the values given to it, each on its own, but the step of the Langevin
  corrector is shared by the batch, as in MatterGen (see [the behaviours kept from MatterGen](../models/mattergen.md#behaviours-kept-from-mattergen)).
- The useful range of the band gap and of the magnetic density was not measured: torchms has
  no model of either. The bulk modulus is followed up to 400 GPa, 2.5 deviations of its
  logarithm from the mean: crystals asked at 400 GPa have a median of 347 GPa, estimated
  without relaxation (see [MatterGen](../models/mattergen.md#conditioned-on-a-bulk-modulus)).
