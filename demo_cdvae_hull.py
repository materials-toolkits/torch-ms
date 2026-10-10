"""Demo: generate crystals with CDVAE and estimate their energy above the hull.

    uv run python demo_cdvae_hull.py                 # 1000 crystals
    uv run python demo_cdvae_hull.py --num 50 --steps-per-level 10   # a quick try

Latent vectors are drawn from a standard normal and decoded by CDVAE into
crystals, by 50 levels of noise of 100 steps each. EnergyEstimator then
estimates the total energy of each crystal with orb-v3 and compares it to the
convex hull of LeMat-Bulk. The crystals are saved with their energy above the
hull in an HDF5 file.

What to keep in mind when reading the numbers:

* the weights of CDVAE were trained on MP-20 by a third party, their quality
  is not known;
* the crystals are not relaxed: their energy is the one of the positions given
  by CDVAE, above the one a relaxation would reach;
* the estimator has an error of about 0.01 eV/atom on known crystals, and was
  not validated on generated ones.

The dynamics takes about 3 minutes per batch on an RTX 4060. The hulls are the
other long step: a crystal with 8 elements or more is compared to a hull in as
many dimensions, which takes about a minute. They are computed at the end, for
all the crystals, on several processes.

It needs huggingface_hub for the hull (uv sync --extra hub).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from torchms.data import HDF5Dataset, Structures
from torchms.data.hull import energy_above_reference
from torchms.models.estimator import EnergyEstimator
from torchms.graph.geometry import lattice_from_parameters, lattice_heights
from torchms.models import CDVAE

# a cell thinner than this, in Å, is not decoded: it has no room for an atom,
# and the number of its images within the radius of the graphs has no limit
MIN_HEIGHT = 1.5

# thresholds of the energy above the hull reported, in eV/atom
THRESHOLDS = (0.0, 0.025, 0.1, 0.25, 0.5)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--num", type=int, default=1000, help="crystals to generate")
    parser.add_argument(
        "--batch-size", type=int, default=250, help="crystals generated together"
    )
    parser.add_argument(
        "--energy-batch-size",
        type=int,
        default=50,
        help="crystals whose energy is estimated together",
    )
    parser.add_argument(
        "--steps-per-level",
        type=int,
        default=100,
        help="steps of dynamics at each of the 50 levels of noise",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="processes computing the hulls",
    )
    parser.add_argument("--output", default="cdvae_samples.hdf5")

    return parser.parse_args()


@torch.no_grad()
def generate(
    model: CDVAE,
    num: int,
    steps_per_level: int,
    generator: torch.Generator,
) -> tuple[Structures, int]:
    """Generate crystals from latent vectors drawn at random.

    Returns:
        Crystals, and the number of latent vectors which were not decoded:
        the ones of a crystal without atom or with a flat cell.
    """
    device = model.sigmas.device
    latent_dim = model.config.model.latent_dim
    latent = torch.randn((num, latent_dim), generator=generator, device=device)

    num_atoms = model.predict_num_atoms(latent).argmax(dim=-1)
    lattice = lattice_from_parameters(*model.predict_lattice(latent, num_atoms))
    valid = (
        (num_atoms > 0)
        & lattice.isfinite().all(dim=(1, 2))
        & (lattice_heights(lattice).amin(dim=1) > MIN_HEIGHT)
    )

    structures = model.generate(
        latent[valid],
        num_atoms=num_atoms[valid],
        steps_per_level=steps_per_level,
        generator=generator,
    )

    return structures, int((~valid).sum())


def report(above: np.ndarray, num_atoms: np.ndarray, rejected: int) -> None:
    """Print the distribution of the energy above the hull of the crystals."""
    known = np.isfinite(above)
    values = above[known]

    print(f"\ncrystals generated       {len(above)}")
    print(f"latent vectors rejected  {rejected} (no atom or flat cell)")
    print(
        f"without energy           {int((~known).sum())} "
        "(element absent from the hull, or two atoms at the same position)"
    )
    print(f"atoms per crystal        {num_atoms.mean():.1f} on average")

    if len(values) == 0:
        return

    print("\nenergy above the hull, in eV/atom, not relaxed:")
    print(f"  median                 {np.median(values):.3f}")
    print(f"  mean                   {values.mean():.3f}")
    low, high = np.percentile(values, [10, 90])
    print(f"  10% and 90%            {low:.3f}, {high:.3f}")
    print(f"  lowest                 {values.min():.3f}")
    for threshold in THRESHOLDS:
        count = int((values <= threshold).sum())
        print(
            f"  at most {threshold:<5}          {count:4d} "
            f"({100 * count / len(above):.1f}% of the crystals)"
        )


def main() -> None:
    args = parse_arguments()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    model = CDVAE.from_pretrained(device=args.device)
    estimator = EnergyEstimator.from_lematerial(device=args.device)
    print(f"device {args.device}")
    print(estimator)

    if estimator.hull is None:
        raise RuntimeError("the estimator has no hull")

    batches: list[Structures] = []
    energy: list[torch.Tensor] = []
    rejected = 0
    start = time.time()

    for first in range(0, args.num, args.batch_size):
        num = min(args.batch_size, args.num - first)
        structures, skipped = generate(model, num, args.steps_per_level, generator)
        rejected += skipped

        # fewer crystals at a time: the graph of the estimator is larger
        for part in structures.split(args.energy_batch_size):
            energy.append(estimator(part).cpu())

        batches.append(structures.to("cpu"))
        done = sum(len(batch.num_atoms) for batch in batches)
        print(
            f"{done + rejected}/{args.num} latent vectors, {done} crystals, "
            f"{time.time() - start:.0f} s"
        )

    crystals = Structures.collate(batches)
    crystals.total_energy = torch.cat(energy).float()

    # all the hulls at once: one process per chemical system
    print(f"energy above the hull, on {args.num_workers} processes")
    above = energy_above_reference(
        estimator.hull,
        crystals.z.numpy(),
        crystals.num_atoms.numpy(),
        torch.cat(energy).numpy(),
        args.num_workers,
        progress=True,
    )
    energy_above_hull = torch.from_numpy(above).float()
    crystals.energy_above_hull = energy_above_hull
    print(f"done in {time.time() - start:.0f} s")

    report(energy_above_hull.numpy(), crystals.num_atoms.numpy(), rejected)

    HDF5Dataset.from_structures(args.output, crystals)
    print(f"\ncrystals and their energy above the hull saved to {args.output}")


if __name__ == "__main__":
    main()
