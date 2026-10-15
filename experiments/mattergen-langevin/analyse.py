"""Compare the two arms of the paired sampling: analyse.py RESULTS.npz [...]"""
import sys
from collections import Counter

import numpy as np
import torch

from torchms.data import Structures
from torchms.graph.transforms import AddRadiusGraph
from torchms.models.estimator import EnergyEstimator

ARMS = ("coupled", "independent")
files = [np.load(f) for f in sys.argv[1:]]
d = {k: np.concatenate([f[k] for f in files]) for k in files[0].files}
estimator = EnergyEstimator.from_lematerial(device="cuda")


def load(arm):
    return Structures(
        lattice=torch.from_numpy(d[f"{arm}_lattice"]),
        x=torch.from_numpy(d[f"{arm}_x"]),
        z=torch.from_numpy(d[f"{arm}_z"]),
        num_atoms=torch.from_numpy(d[f"{arm}_num_atoms"]),
    )


def metrics(s):
    _, above = estimator.predict_energy_above_hull(s)
    graph = AddRadiusGraph(3.0, self_images=True)(s).graph
    nearest = torch.full((len(s.num_atoms),), 3.0).scatter_reduce(
        0, graph.edge_batch, graph.edges_distances(), "amin"
    )
    volume = torch.linalg.det(s.lattice).abs() / s.num_atoms
    return {
        "above": np.asarray(above, dtype=np.float64),
        "nearest": nearest.numpy(),
        "volume": volume.numpy(),
    }


m = {arm: metrics(load(arm)) for arm in ARMS}
n = len(d["coupled_num_atoms"])
num_atoms = d["coupled_num_atoms"]
ptr = np.concatenate([[0], np.cumsum(num_atoms)])
comp = {arm: [tuple(sorted(d[f"{arm}_z"][ptr[i]:ptr[i + 1]])) for i in range(n)] for arm in ARMS}
rng = np.random.default_rng(0)


def bootstrap(values, statistic, reps=5000):
    idx = rng.integers(0, len(values), (reps, len(values)))
    stats = np.array([statistic(values[i]) for i in idx])
    return np.percentile(stats, [2.5, 97.5])


print(f"{n} pairs, {len(files)} files; same composition in both arms: "
      f"{np.mean([a == b for a, b in zip(*comp.values())]):.1%}")
ok = np.isfinite(m["coupled"]["above"]) & np.isfinite(m["independent"]["above"])
print(f"pairs with an energy in both arms: {ok.sum()}")
for arm in ARMS:
    a, near, vol = m[arm]["above"][ok], m[arm]["nearest"], m[arm]["volume"]
    unique = len(set(comp[arm])) / n
    print(f"{arm:12s} E_hull mean {a.mean():.3f} median {np.median(a):.3f} | <0.1 {np.mean(a < 0.1):.1%} "
          f"<0.25 {np.mean(a < 0.25):.1%} | nearest<0.5 {np.mean(near < 0.5):.1%} median {np.median(near):.2f} | "
          f"vol/atom median {np.median(vol):.1f} | unique compositions {unique:.1%}")

a, b = m["coupled"]["above"][ok], m["independent"]["above"][ok]
diff = b - a
lo, hi = bootstrap(diff, np.mean)
print(f"\npaired E_hull, independent - coupled: mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] 95% CI, "
      f"median {np.median(diff):+.4f}")
stable = np.stack([a < 0.1, b < 0.1], 1).astype(float)
lo, hi = bootstrap(stable, lambda s: s[:, 1].mean() - s[:, 0].mean())
print(f"fraction <0.1, independent - coupled: {stable[:, 1].mean() - stable[:, 0].mean():+.1%} [{lo:+.1%}, {hi:+.1%}]")
print(f"discordant pairs: only coupled <0.1 {int(((a < 0.1) & (b >= 0.1)).sum())}, "
      f"only independent <0.1 {int(((b < 0.1) & (a >= 0.1)).sum())}")

print("\nby number of atoms: pairs, fraction <0.1 coupled / independent, mean paired difference")
for lo_n, hi_n in ((1, 4), (5, 8), (9, 12), (13, 20)):
    sel = (num_atoms[ok] >= lo_n) & (num_atoms[ok] <= hi_n)
    if sel.sum():
        print(f"  {lo_n:2d}-{hi_n:2d} atoms: {sel.sum():4d}  {np.mean(a[sel] < 0.1):.1%} / {np.mean(b[sel] < 0.1):.1%}  "
              f"{diff[sel].mean():+.4f}")

for arm in ARMS:
    elements = Counter(d[f"{arm}_z"].tolist())
    print(f"{arm}: most common elements {elements.most_common(8)}")
