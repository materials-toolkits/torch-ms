# MatterGen: is a Langevin step per crystal worse than MatterGen's step per batch?

2026-10-15

## Question

The Langevin corrector of MatterGen (and of the torchms port, `MatterGen.langevin_step_size`)
computes one step for the whole batch, from the means over the batch of the norms of the
scores and of the noises. On a recorded run of 4 crystals, the shared step was between 0.01
and 6 times the step each crystal would get alone. Does giving each crystal its own step
("independent") change the quality of what is generated?

## Setup

- **Arms.** `coupled`: `MatterGen` as ported, identical to MatterGen 9242366 (steps
  bit-identical given the same scores and noise, see `docs/models/mattergen.md`).
  `independent`: the same model with `langevin_step_size` computing
  `2 alpha (snr ‖z_c‖ / ‖score_c‖)²` per crystal instead of from the batch means
  (`experiments/mattergen-langevin/sample.py`, class `Independent`). Nothing else differs.
- **Pairing.** For each batch, both arms get the same numbers of atoms (drawn from
  Alex-MP-20, `sample_num_atoms`, seed `10000 + batch`) and the same prior and noise at every
  step (generator seed `batch`). Only the step size of the corrector differs.
- **Size.** 32 batches of 64 crystals per arm (64 is MatterGen's default `batch_size`):
  2048 pairs, 1 to 20 atoms.
- **Metrics** (`experiments/mattergen-langevin/analyse.py`, run locally on an RTX 4060):
  energy above the hull from torchms's `EnergyEstimator.from_lematerial()` (orb-v3 against
  the LeMat-Bulk hull, MP and Alexandria), **on the unrelaxed structures**; shortest
  interatomic distance (periodic, 3 Å graph); volume per atom; uniqueness of compositions.
- Weights: `mattergen-base` (HF `microsoft/mattergen@5244495`, SHA-256 `81668ee1…`).

## Provenance

- Commit `58c11bd` (main) **with a dirty tree**: the MatterGen port and these scripts were
  not committed. The deployed copy was every tracked and untracked non-ignored file except
  `datasets/` (`git ls-files -co --exclude-standard`), deployed with `cluster deploy` from a
  staging directory. `GIT_SHA=58c11bd+mattergen-uncommitted` in the logs.
- Machine: CRIL cluster, partition `h100`, nodes `nodeGcor1`, `nodeGcor2`, `nodeGcor3`,
  1 × NVIDIA H100 NVL (94 GB) per job, 8 CPUs, 32 GB.
- Command of each job (everything after `--`), `FIRST` = 0 with `N` = 2, then
  `FIRST` = 2, 8, 14, 20, 26 with `N` = 6:

  ```
  env GIT_SHA=58c11bd+mattergen-uncommitted bash experiments/mattergen-langevin/run.sh FIRST N
  ```

  which runs `python experiments/mattergen-langevin/sample.py results/langevin_FIRST.npz 64 N FIRST`.
- Jobs 13514570 to 13514575, all COMPLETED. Logs checked: no error, every job printed its
  `done: batches …` line. Logs and results are in `experiments/mattergen-langevin/results/`.

## Timing

182 to 230 s per arm per batch of 64 crystals (1000 steps, 2000 calls of the network) on an
H100, peak 3.7 GiB: about 3 s per crystal. Wall clock: 14 min for the job of 2 batches, 40 to
43 min for the jobs of 6. For comparison, 16 crystals took 243 s on the local RTX 4060.

## Results

| 2048 pairs | coupled (MatterGen) | independent |
|---|---|---|
| Energy above hull, unrelaxed, mean | 0.089 eV/atom | 0.108 eV/atom |
| Median | 0.061 | 0.077 |
| Below 0.1 eV/atom | **71.6%** | **62.5%** |
| Below 0.25 eV/atom | 94.7% | 92.1% |
| Two atoms closer than 0.5 Å | 0.0% | 0.0% |
| Shortest distance, median | 2.68 Å | 2.64 Å |
| Volume per atom, median | 23.8 Å³ | 24.0 Å³ |
| Unique compositions | 99.8% | 99.9% |
| No energy (element outside the hull) | 0 | 0 |

Paired differences, independent minus coupled, with bootstrap 95% intervals (5000
resamplings of pairs):

| | difference | 95% interval |
|---|---|---|
| Energy above hull, mean | +0.0195 eV/atom | [+0.0132, +0.0257] |
| Energy above hull, median of the differences | +0.0095 eV/atom | |
| Fraction below 0.1 eV/atom | −9.1 points | [−11.8, −6.4] |

Discordant pairs: 495 where only the coupled crystal is below 0.1 eV/atom, 308 where only the
independent one is.

By number of atoms:

| Atoms | Pairs | Below 0.1, coupled / independent | Mean difference |
|---|---|---|---|
| 1–4 | 394 | 78.9% / 70.6% | +0.018 |
| 5–8 | 520 | 73.8% / 64.8% | +0.016 |
| 9–12 | 740 | 69.3% / 60.9% | +0.019 |
| 13–20 | 394 | 65.5% / 54.1% | +0.026 |

The two arms end with the same composition in 3.1% of the pairs only: the step size of the
corrector changes which crystal is generated, not only how well it is refined. The elements
are distributed alike: O, S, F, Se, Cl, Pd lead in both; La and Tl among the eight most common
in the coupled arm, Hg and Li in the independent one.

## Qualitative

The first 12 pairs of batch 0 (formula, unrelaxed energy above the hull in eV/atom):

| Atoms | coupled | independent |
|---|---|---|
| 4 | Rh2W2, +0.037 | LuScZn2, +0.009 |
| 14 | Rh2Sc2Te10, +0.165 | K2Sc5Te7, +0.157 |
| 14 | Ag2Mg8Pd4, +0.030 | La2Pd6Tb6, +0.091 |
| 6 | Cd2DyTb3, +0.018 | Dy2Nb2Si2, −0.005 |
| 6 | CuO4Te, +0.047 | CuO4Te, +0.171 |
| 12 | Ba2In3SmTe6, +0.123 | Ho5Te7, +0.010 |
| 4 | MnNi2Pt, +0.099 | Al2MnNi, +0.034 |
| 6 | Al2CuTi3, +0.040 | Au5Pd, +0.004 |
| 12 | Tb4Te8, +0.100 | Pt5Tb2Te5, +0.234 |
| 4 | DyHgInSr, +0.019 | FeIr2Ni, +0.070 |
| 8 | Ni4Pt4, +0.040 | Ir2Nb2Ni4, −0.009 |
| 5 | Fe2Ge2Rh, +0.414 | Fe2Ge2Yb, −0.007 |

Mostly intermetallics and chalcogenides, as MatterGen's paper describes for its base model.
Pair by pair, either arm can win (the third row against the twelfth); the difference is in
the distribution. CuO4Te is the one pair here with the same composition in both arms, 0.12
eV/atom apart: the same composition, a worse structure. No structural failure was seen in
either arm: no atoms closer than 0.5 Å, no collapsed or exploded cells (median volumes 24 Å³
per atom in both).

## Conclusion

**Yes: a step per crystal degrades the samples.** At MatterGen's default batch size of 64,
the independent step gives 9 points fewer crystals within 0.1 eV/atom of the hull (62.5%
against 71.6%, interval [−11.8, −6.4]) and a mean energy above the hull higher by 0.02
eV/atom. The effect is there for every size of crystal, a little larger for the largest ones,
not concentrated on the small crystals as I expected from the recorded run.

So the batch coupling is not only a quirk to reproduce: the batch mean of the norms acts as a
regularization of the step. A plausible mechanism, **not tested here**: alone, a crystal whose
score is small relative to its noise gets a very large step (the 2-atom crystal of the
recorded run would have moved 5 σ instead of 0.5 σ at the last step); the mean over 64
crystals damps these. A consequence for users: **sampling with `batch_size=1` is the
independent arm**, and small batches presumably fall in between. MatterGen's reported numbers
are for its default batches.

## Limits

- **Unrelaxed energies.** MatterGen's stability metric relaxes each crystal (MatterSim) before
  computing its energy; torchms has no relaxation. These numbers are a proxy, biased upward,
  and are not comparable to the paper's. Relaxation could shrink or widen the gap.
- One estimator (orb-v3 with a correction, hull of LeMat-Bulk MP and Alexandria), reliable
  within 0.1 eV/atom but not exactly on the hull.
- One batch size (64) for the coupled arm; no novelty against the training set, no symmetry,
  no charge neutrality.

## Next

- Coupled arm at batch sizes 1, 4, 16, 64, 256, to see how the quality depends on it.
- Relax both arms (e.g. with a relaxer outside torchms) to confirm on relaxed energies.
- Log the per-crystal step sizes to test the mechanism above.
