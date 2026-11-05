"""Reference outputs of the two models of DiffCSP, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
DiffCSP. DiffCSP is not on PyPI: clone it first, at the commit the rewrite was
written for.

    git clone https://github.com/jiaor17/DiffCSP.git
    git -C DiffCSP checkout 7121d159826efa2ba9500bf299250d96da37f146

    uv run --isolated --no-project --python 3.8 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-1.9.0+cpu.html \
        --index-strategy unsafe-best-match --no-build-package pyshtools \
        --with "torch==1.9.0+cpu" --with torch-scatter==2.0.8 \
        --with torch-sparse==0.6.12 --with torch-geometric==1.7.2 \
        --with pytorch-lightning==1.3.8 --with hydra-core==1.1.0 \
        --with pymatgen==2023.8.10 --with "pyxtal<0.6" \
        --with p-tqdm==1.3.3 --with python-dotenv --with "numpy<1.24" \
        --with scikit-learn --with pandas --with networkx --with sympy \
        --with "torchmetrics<0.8" --with setuptools==59.5.0 --with einops \
        python tests/references/reference_diffcsp.py DiffCSP WEIGHTS

torch, torch-geometric, pytorch-lightning and pymatgen are the versions of the
README of DiffCSP. The other ones are what its modules import, at versions
which still install: pyxtal before 0.6 and a built pyshtools, as later ones
need a newer numpy or a Fortran compiler, and setuptools 59.5, the last one
the tensorboard of torch 1.9 imports.

WEIGHTS is the directory of the two files of weights listed in
src/torchms/models/diffcsp/config.yaml: the cache of torchms after a first
from_pretrained(). The script is run once for each model, and finds which one
it is given:

* "~/.cache/torchms/diffcsp/diffcsp-mp20", of DiffCSP: the result is
  written to tests/data/diffcsp_reference.npz;
* "~/.cache/torchms/diffcsp/diffcsp-gen-mp20", of DiffCSPGeneration: the
  result is written to tests/data/diffcsp_generation_reference.npz.

The structures are the ones of tests/data/orb_reference.npz. The results are
read by tests/models/test_diffcsp.py.

The checkpoints only hold tensors in an OrderedDict: they are read by
torch.load as they are.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# resolved now: importing diffcsp changes the working directory
DATA = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT = "last.ckpt"

# structures of the sampling, which is run whole, and its step at the lowest noise
NUM_DYNAMICS = 6
STEP_LR = 1e-5

# steps of the sampling which are saved, with the noise they add: the first
# ones follow each other
SAVED_TIMES = [1000, 999, 998, 750, 500, 250, 100, 10, 2, 1]

# the same for the generation, with fewer atoms and fewer steps: a step has
# 100 values per atom
GENERATION_DYNAMICS = [0, 1, 4, 5]
GENERATION_STEP_LR = 5e-6
GENERATION_SAVED_TIMES = [1000, 999, 998, 500, 1]


def load_model(weights: Path):
    """Model of DiffCSP built from hparams.yaml, with the weights of the checkpoint."""
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(weights.resolve())):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

    checkpoint = torch.load(weights / CHECKPOINT, map_location="cpu")
    # the checkpoint of the generation has a schedule its model does not have
    state = {
        name: values
        for name, values in checkpoint["state_dict"].items()
        if not name.startswith("type_sigma_scheduler.")
    }
    model.load_state_dict(state)

    return model.eval()


def load_batch():
    """Structures of the tests as a batch of DiffCSP, and their cells."""
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    crystals = [
        Data(
            frac_coords=torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]]),
            atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
            num_atoms=int(data["num_atoms"][i]),
            num_nodes=int(data["num_atoms"][i]),
        )
        for i in range(len(data["num_atoms"]))
    ]

    return Batch.from_data_list(crystals), torch.from_numpy(data["lattice"])


class Recorder:
    """Records what torch draws while a model samples."""

    def __init__(self):
        self.drawn = []
        self.functions = (torch.randn, torch.rand, torch.randn_like)

    def record(self, function):
        def wrapped(*args, **kwargs):
            self.drawn.append(function(*args, **kwargs))
            return self.drawn[-1]

        return wrapped

    def __enter__(self):
        torch.randn, torch.rand, torch.randn_like = map(self.record, self.functions)
        return self.drawn

    def __exit__(self, *error):
        torch.randn, torch.rand, torch.randn_like = self.functions


def prediction(model, batch, lattices):
    """Reference of the model which is given the elements."""
    from torch_geometric.data import Batch

    n_struct = batch.num_graphs
    num_atoms = batch.num_atoms
    result = {}

    def denoise(times, frac_coords, cells):
        return model.decoder(
            model.time_embedding(times),
            batch.atom_types,
            frac_coords,
            cells,
            num_atoms,
            batch.batch,
        )

    # the network on the structures as they are, at times from the first
    # to the last
    times = torch.linspace(1, model.beta_scheduler.timesteps, n_struct).long()
    pred_l, pred_x = denoise(times, batch.frac_coords, lattices)
    result.update(times=times.numpy(), pred_l=pred_l.numpy(), pred_x=pred_x.numpy())

    # and on the structures with the noise of these times, as in the training
    alphas_cumprod = model.beta_scheduler.alphas_cumprod[times]
    sigmas = model.sigma_scheduler.sigmas[times].repeat_interleave(num_atoms)
    noisy_l = torch.sqrt(alphas_cumprod)[:, None, None] * lattices + torch.sqrt(
        1.0 - alphas_cumprod
    )[:, None, None] * torch.randn_like(lattices)
    noisy_x = batch.frac_coords + sigmas[:, None] * torch.randn_like(batch.frac_coords)
    noisy_x = noisy_x % 1.0
    pred_l, pred_x = denoise(times, noisy_x, noisy_l)
    result.update(
        noisy_l=noisy_l.numpy(),
        noisy_x=noisy_x.numpy(),
        noisy_pred_l=pred_l.numpy(),
        noisy_pred_x=pred_x.numpy(),
    )

    # the whole sampling of the first structures, with everything it draws
    first = Batch.from_data_list(batch.to_data_list()[:NUM_DYNAMICS])
    with Recorder() as drawn:
        _, trajectory = model.sample(first, step_lr=STEP_LR)

    # a cell and positions to start from, then at each step but the last one
    # the noise of the corrector, of a cell and of positions, which is not
    # used, and the noise of the predictor
    steps = model.beta_scheduler.timesteps
    assert len(drawn) == 2 + 4 * (steps - 1)
    all_l, all_x = trajectory["all_lattices"], trajectory["all_frac_coords"]
    assert torch.equal(all_l[0], drawn[0]) and torch.equal(all_x[0], drawn[1] % 1.0)

    def noise(time, position):
        if time == 1:
            return torch.zeros_like(drawn[position])

        return drawn[2 + 4 * (steps - time) + position]

    def saved(values):
        return torch.stack(values).numpy()

    result.update(
        dynamics_times=np.array(SAVED_TIMES),
        dynamics_l=saved([all_l[steps - t] for t in SAVED_TIMES]),
        dynamics_x=saved([all_x[steps - t] for t in SAVED_TIMES]),
        dynamics_next_l=saved([all_l[steps - t + 1] for t in SAVED_TIMES]),
        dynamics_next_x=saved([all_x[steps - t + 1] for t in SAVED_TIMES]),
        corrector_noise_x=saved([noise(t, 1) for t in SAVED_TIMES]),
        predictor_noise_l=saved([noise(t, 2) for t in SAVED_TIMES]),
        predictor_noise_x=saved([noise(t, 3) for t in SAVED_TIMES]),
        final_l=all_l[-1].numpy(),
        final_x=all_x[-1].numpy(),
    )

    np.savez_compressed(DATA / "diffcsp_reference.npz", **result)
    print("pred_l", np.abs(result["pred_l"]).max(), np.abs(result["pred_l"]).mean())
    print("pred_x", np.abs(result["pred_x"]).max(), np.abs(result["pred_x"]).mean())
    print("noisy pred_x", np.abs(result["noisy_pred_x"]).max())
    print("final cells", result["final_l"].round(2).tolist())


def generation(model, batch, lattices):
    """Reference of the model which also generates the elements."""
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data

    n_struct = batch.num_graphs
    num_atoms = batch.num_atoms
    num_elements = model.decoder.node_embedding.in_features
    result = {}

    # the network on the structures with the noise of times from the first to
    # the last, as in the training: an element is a vector of 100 values
    times = torch.linspace(1, model.beta_scheduler.timesteps, n_struct).long()
    alphas_cumprod = model.beta_scheduler.alphas_cumprod[times]
    signal, deviation = torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)
    sigmas = model.sigma_scheduler.sigmas[times].repeat_interleave(num_atoms)

    noisy_l = signal[:, None, None] * lattices + deviation[
        :, None, None
    ] * torch.randn_like(lattices)
    noisy_x = batch.frac_coords + sigmas[:, None] * torch.randn_like(batch.frac_coords)
    noisy_x = noisy_x % 1.0
    one_hot = F.one_hot(batch.atom_types - 1, num_classes=num_elements).float()
    noisy_t = signal.repeat_interleave(num_atoms)[
        :, None
    ] * one_hot + deviation.repeat_interleave(num_atoms)[:, None] * torch.randn_like(
        one_hot
    )

    pred_l, pred_x, pred_t = model.decoder(
        model.time_embedding(times), noisy_t, noisy_x, noisy_l, num_atoms, batch.batch
    )
    result.update(
        times=times.numpy(),
        noisy_l=noisy_l.numpy(),
        noisy_x=noisy_x.numpy(),
        noisy_t=noisy_t.numpy(),
        noisy_pred_l=pred_l.numpy(),
        noisy_pred_x=pred_x.numpy(),
        noisy_pred_t=pred_t.numpy(),
    )

    # the whole sampling of a few structures, of which only the numbers of
    # atoms are given, with everything it draws. The vectors of the elements
    # are not returned: they are read where the network gets them, at its
    # first call of each step
    first = Batch.from_data_list(
        [
            Data(num_atoms=torch.LongTensor([int(n)]), num_nodes=int(n))
            for n in num_atoms[GENERATION_DYNAMICS]
        ]
    )
    given = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, arguments: given.append(arguments[1])
    )
    with Recorder() as drawn:
        outputs, trajectory = model.sample(first, step_lr=GENERATION_STEP_LR)
    hook.remove()

    # a cell, positions and elements to start from, then at each step but the
    # last one the noise of the corrector, of a cell, of elements and of
    # positions, of which only the last is used, and the noise of the predictor
    steps = model.beta_scheduler.timesteps
    assert len(drawn) == 3 + 6 * (steps - 1) and len(given) == 2 * steps
    all_l, all_x = trajectory["all_lattices"], trajectory["all_frac_coords"]
    all_t = given[::2] + [outputs["atom_types"]]
    assert torch.equal(all_l[0], drawn[0]) and torch.equal(all_x[0], drawn[1] % 1.0)
    assert torch.equal(all_t[0], drawn[2])

    def noise(time, position):
        if time == 1:
            # a cell, elements or positions, as the start in another order
            return torch.zeros_like(drawn[(0, 2, 1)[position % 3]])

        return drawn[3 + 6 * (steps - time) + position]

    def saved(values):
        return torch.stack(values).numpy()

    saved_times = GENERATION_SAVED_TIMES
    result.update(
        dynamics_structures=np.array(GENERATION_DYNAMICS),
        dynamics_times=np.array(saved_times),
        dynamics_l=saved([all_l[steps - t] for t in saved_times]),
        dynamics_x=saved([all_x[steps - t] for t in saved_times]),
        dynamics_t=saved([all_t[steps - t] for t in saved_times]),
        dynamics_next_l=saved([all_l[steps - t + 1] for t in saved_times]),
        dynamics_next_x=saved([all_x[steps - t + 1] for t in saved_times]),
        dynamics_next_t=saved([all_t[steps - t + 1] for t in saved_times]),
        corrector_noise_x=saved([noise(t, 2) for t in saved_times]),
        predictor_noise_l=saved([noise(t, 3) for t in saved_times]),
        predictor_noise_t=saved([noise(t, 4) for t in saved_times]),
        predictor_noise_x=saved([noise(t, 5) for t in saved_times]),
        final_l=all_l[-1].numpy(),
        final_x=all_x[-1].numpy(),
        final_z=trajectory["atom_types"][-1].numpy(),
    )

    np.savez_compressed(DATA / "diffcsp_generation_reference.npz", **result)
    for name in ("noisy_pred_l", "noisy_pred_x", "noisy_pred_t"):
        print(name, np.abs(result[name]).max(), np.abs(result[name]).mean())
    print("final elements", result["final_z"].tolist())
    print("final cells", result["final_l"].round(2).tolist())


def main():
    diffcsp, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()
    os.environ["PROJECT_ROOT"] = str(diffcsp)
    sys.path.insert(0, str(diffcsp))

    torch.manual_seed(0)
    model = load_model(weights)
    batch, lattices = load_batch()
    print("structures", batch.num_graphs, "atoms", int(batch.num_atoms.sum()))

    with torch.no_grad():
        if model.decoder.pred_type:
            generation(model, batch, lattices)
        else:
            prediction(model, batch, lattices)


if __name__ == "__main__":
    main()
