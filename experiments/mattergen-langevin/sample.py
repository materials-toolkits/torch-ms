"""Paired sampling: MatterGen's coupled Langevin step vs a step per crystal.

usage: langevin_experiment.py OUT BATCH_SIZE NUM_BATCHES [FIRST_SEED]
Both arms get the same numbers of atoms, prior and noise (same generator seed
per batch); only the step size of the corrector differs.
"""
import sys, time
import numpy as np, torch
from torchms.models import MatterGen


class Independent(MatterGen):
    def langevin_step_size(self, score_norm, noise_norm, snr, alpha):
        max_step = torch.tensor(self.config.sampling.max_step_size, device=alpha.device)
        step = torch.minimum((snr * noise_norm / score_norm) ** 2 * 2 * alpha, max_step)
        return torch.where(score_norm == 0, max_step, step)


out, batch_size, num_batches = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
models = {"coupled": MatterGen.from_pretrained(device="cuda"),
          "independent": Independent.from_pretrained(device="cuda")}
result = {}
for b in range(first, first + num_batches):
    num_atoms = models["coupled"].sample_num_atoms(
        batch_size, generator=torch.Generator(device="cuda").manual_seed(10_000 + b))
    for arm, model in models.items():
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.no_grad():
            s = model.sample(num_atoms, generator=torch.Generator(device="cuda").manual_seed(b))
        s = s.to("cpu")
        print(f"batch {b} {arm} {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        for k, v in {"lattice": s.lattice, "x": s.x, "z": s.z, "num_atoms": s.num_atoms}.items():
            result.setdefault(f"{arm}_{k}", []).append(v.numpy())
    np.savez(out, **{k: np.concatenate(v) for k, v in result.items()})
