"""Reference outputs of ADiT, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
ADiT. ADiT is not on PyPI: clone it at the commit the rewrite was written for,
and give its path to the script, which imports its modules of the model and of
the sampling, not its evaluation (pymatgen, rdkit, openbabel, posebusters):
the modules of the metrics are replaced by empty ones before the import.

    git clone https://github.com/facebookresearch/all-atom-diffusion-transformer.git
    git -C all-atom-diffusion-transformer checkout b9ce505f170597a7c8ca50d13ce8e15df21cf8c9

    uv run --isolated --no-project --python 3.10 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html \
        --index-strategy unsafe-best-match \
        --with "torch==2.3.1+cpu" --with torch-geometric==2.6.1 \
        --with torch-scatter==2.1.2 --with lightning==2.4.0 \
        --with torchmetrics==1.4.1 --with omegaconf==2.3.0 \
        --with hydra-core==1.3.2 --with numpy==1.26.4 --with pandas==2.2.3 \
        --with rich --with joblib --with tqdm \
        python tests/references/reference_adit.py all-atom-diffusion-transformer \
        WEIGHTS VAE

torch, torch-geometric, torch-scatter, lightning, torchmetrics, omegaconf,
numpy and pandas are the versions of the requirements.txt of ADiT, for the CPU.

WEIGHTS is the file of weights listed in src/torchms/models/adit/config.yaml,
ldm.ckpt: "~/.cache/torchms/adit/ldm.ckpt" after a first from_pretrained().
VAE is vae.ckpt of the same commit of the same repository on Hugging Face,
https://huggingface.co/chaitjo/all-atom-diffusion-transformer/resolve/49ee17be088855ffed4ab8145f194e7a3c8b8879/vae.ckpt
(SHA-256 b549373e3b473d62021d6bc1a1ea0b5fafb808dfdb06e049c2a4f922943499e5):
ADiT builds its model from it before loading ldm.ckpt, which has the same
autoencoder.

The model is built as src/eval_diffusion.py of ADiT builds it, with the
interpolant and the sampling of configs/diffusion_module/ldm.yaml, then the
state dict of the checkpoint is loaded, as trainer.test does. What it reads of
the datasets is written to a temporary directory: the numbers of atoms of
config.yaml, and empty lists of CIF and SMILES, which only its metrics read.

The structures are the 40 crystals of tests/data/orb_reference.npz and the
molecules of MOLECULES. The result is written to tests/data/adit_reference.npz
and read by tests/models/test_adit.py:

* the posterior of the encoder of each structure, crystals and molecules in
  one batch, and what the decoder predicts from its mean;
* the sampling of NUM_SAMPLES crystals and NUM_SAMPLES molecules by
  sample_and_decode: the numbers of atoms it draws, its initial noise before
  (padded, the padding included) and after centering, the latent vectors and
  the predictions of the steps of SAVED_STEPS, the last one of which is the
  prediction at the time 1, and what the decoder predicts from it.
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

# as configs/diffusion_module/ldm.yaml
NUM_TIMESTEPS = 100
CFG_SCALE = 2.0
MIN_T = 1e-2

NUM_SAMPLES = 6

# steps of the sampling which are saved: the latent vectors before the step,
# the prediction of the step before, the prediction and the latent vectors
# after the step; the last step is the prediction at the time 1
SAVED_STEPS = [0, 1, 2, 50, 97, 98, 99]

# number of atoms by number of atoms, as num_nodes_bincount.pt of MP20_ADiT
# and QM9_ADiT, written in src/torchms/models/adit/config.yaml
NUM_ATOMS_CRYSTALS = [0, 131, 1152, 1129, 8341, 2591, 4522, 1218, 4275, 1817, 5224,
                      703, 5322, 1180, 3575, 807, 3639, 490, 2919, 571, 4670]  # fmt: skip
NUM_ATOMS_MOLECULES = [0, 0, 0, 2, 4, 5, 12, 20, 65, 172, 483, 1053, 2189, 4027,
                       6758, 10216, 13824, 16969, 17442, 18146, 12403, 13166, 4428,
                       6362, 712, 1923, 59, 356, 0, 35]  # fmt: skip

# molecules, positions in Å, not centered
MOLECULES = [
    # water
    ([8, 1, 1], [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]]),
    # methane
    (
        [6, 1, 1, 1, 1],
        [
            [1.0, 1.0, 1.0],
            [1.629, 1.629, 1.629],
            [0.371, 0.371, 1.629],
            [0.371, 1.629, 0.371],
            [1.629, 0.371, 0.371],
        ],
    ),
    # ammonia
    (
        [7, 1, 1, 1],
        [
            [0.0, 0.0, 0.1],
            [0.94, 0.0, -0.27],
            [-0.47, 0.814, -0.27],
            [-0.47, -0.814, -0.27],
        ],
    ),
    # ethanol
    (
        [6, 6, 8, 1, 1, 1, 1, 1, 1],
        [
            [-1.21, -0.23, 0.0],
            [0.12, 0.53, 0.0],
            [1.16, -0.43, 0.0],
            [-1.28, -0.87, 0.88],
            [-1.28, -0.87, -0.88],
            [-2.06, 0.46, 0.0],
            [0.19, 1.17, 0.89],
            [0.19, 1.17, -0.89],
            [1.99, 0.06, 0.0],
        ],
    ),
]


def stub_metrics():
    """Empty modules in place of the ones of the metrics of ADiT and of wandb."""

    class Evaluator:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    for name in ("crystal", "mof", "molecule"):
        for kind in ("Generation", "Reconstruction"):
            module = types.ModuleType(f"src.eval.{name}_{kind.lower()}")
            title = {"crystal": "Crystal", "mof": "MOF", "molecule": "Molecule"}[name]
            setattr(module, f"{title}{kind}Evaluator", Evaluator)
            sys.modules[module.__name__] = module

    sys.modules["wandb"] = types.ModuleType("wandb")


def write_datasets(directory: Path):
    """What the model reads of the datasets when it is built."""
    import pandas as pd

    for name, counts in (
        ("mp_20", NUM_ATOMS_CRYSTALS),
        ("qm9", NUM_ATOMS_MOLECULES),
        ("qmof", NUM_ATOMS_CRYSTALS),
    ):
        (directory / name / "raw").mkdir(parents=True)
        torch.save(torch.tensor(counts), directory / name / "num_nodes_bincount.pt")
    torch.save(torch.ones(231), directory / "mp_20" / "spacegroups_bincount.pt")
    pd.DataFrame({"cif": []}).to_csv(directory / "mp_20" / "raw" / "all.csv")
    torch.save([], directory / "qm9" / "smiles.pt")


def load_model(weights: Path, vae: Path, data_dir: Path):
    """Model of ADiT as src/eval_diffusion.py builds it, with the weights of ldm.ckpt."""
    from omegaconf import OmegaConf

    from src.models.interpolants.flow_matching import FlowMatchingInterpolant
    from src.models.ldm_module import LatentDiffusionLitModule

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    denoiser = checkpoint["hyper_parameters"]["denoiser"]

    model = LatentDiffusionLitModule(
        autoencoder_ckpt=str(vae),
        denoiser=denoiser,
        interpolant=FlowMatchingInterpolant(
            min_t=MIN_T,
            corrupt=True,
            num_timesteps=NUM_TIMESTEPS,
            self_condition=True,
            self_condition_prob=0.5,
        ),
        augmentations=OmegaConf.create({"frac_coords": True, "pos": True}),
        sampling=OmegaConf.create(
            {
                "cfg_scale": CFG_SCALE,
                "num_samples": NUM_SAMPLES,
                "batch_size": NUM_SAMPLES,
                "data_dir": str(data_dir),
                "visualize": False,
                "save_dir": str(data_dir),
                "removeHs": False,
            }
        ),
        conditioning=OmegaConf.create({"dataset_idx": True, "spacegroup": False}),
        optimizer=None,
        scheduler=None,
        scheduler_frequency=1,
        compile=False,
    )
    model.load_state_dict(checkpoint["state_dict"])

    return model.eval()


def load_batch():
    """Crystals of the tests and molecules, as a batch of ADiT."""
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    structures = []
    for i, n in enumerate(data["num_atoms"]):
        frac_coords = torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]])
        cell = torch.from_numpy(data["lattice"][i])
        structures.append(
            Data(
                atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
                frac_coords=frac_coords,
                pos=frac_coords @ cell,
                num_atoms=torch.tensor([int(n)]),
                token_idx=torch.arange(int(n)),
                dataset_idx=torch.tensor([0]),
                num_nodes=int(n),
            )
        )

    for z, positions in MOLECULES:
        structures.append(
            Data(
                atom_types=torch.tensor(z),
                frac_coords=torch.zeros(len(z), 3),
                pos=torch.tensor(positions),
                num_atoms=torch.tensor([len(z)]),
                token_idx=torch.arange(len(z)),
                dataset_idx=torch.tensor([1]),
                num_nodes=len(z),
            )
        )

    return Batch.from_data_list(structures)


def autoencoder(model, batch):
    """Posterior of each atom, and what the decoder predicts from its mean."""
    encoded = model.autoencoder.encode(batch)
    posterior = encoded["posterior"]
    mean, logvar = posterior.mean.clone(), posterior.logvar.clone()
    encoded["x"] = posterior.mode()
    out = model.autoencoder.decode(encoded)

    result = {"enc_mean": mean, "enc_logvar": logvar}
    result.update(
        {
            f"dec_{key}": out[key]
            for key in ("atom_types", "lattices", "frac_coords", "pos")
        }
    )
    return result


def sampling(model, dataset_idx, prefix):
    """The sampling of NUM_SAMPLES structures, with its noise and its steps."""
    drawn = []
    centered = model.interpolant._centered_gaussian

    def recorded(*args, **kwargs):
        noise = torch.randn(*args[:3], device=model.interpolant.device)
        drawn.append(noise)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

    model.interpolant._centered_gaussian = recorded
    try:
        out, batch, samples = model.sample_and_decode(
            num_nodes_bincount=model.num_nodes_bincount[
                {0: "mp20", 1: "qm9"}[dataset_idx]
            ],
            spacegroups_bincount=None,
            batch_size=NUM_SAMPLES,
            cfg_scale=CFG_SCALE,
            dataset_idx=dataset_idx,
        )
    finally:
        model.interpolant._centered_gaussian = centered

    assert len(drawn) == 1
    num_atoms = batch["num_atoms"]
    mask = torch.arange(int(num_atoms.max()))[None] < num_atoms[:, None]

    # the latent vectors of the batch and of its copy without dataset are the
    # same: the first half is kept
    tokens = [x[:NUM_SAMPLES] for x in samples["tokens_traj"]]
    clean = samples["clean_traj"]
    assert len(clean) == NUM_TIMESTEPS and len(tokens) == NUM_TIMESTEPS + 1

    result = {
        "num_atoms": num_atoms,
        "raw_noise": drawn[0],
        "noise": tokens[0][mask],
        "steps": torch.tensor(SAVED_STEPS),
    }
    for step in SAVED_STEPS:
        result[f"step{step}_x"] = tokens[step][mask]
        if step > 0:
            result[f"step{step}_previous"] = clean[step - 1][mask]
        result[f"step{step}_prediction"] = clean[step][mask]
        if step < NUM_TIMESTEPS - 1:
            result[f"step{step}_next"] = tokens[step + 1][mask]
    result.update(
        {
            f"dec_{key}": out[key]
            for key in ("atom_types", "lattices", "frac_coords", "pos")
        }
    )

    return {f"{prefix}_{key}": value for key, value in result.items()}


def main():
    project, weights, vae = (Path(arg).resolve() for arg in sys.argv[1:4])
    sys.path.insert(0, str(project))
    stub_metrics()

    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory() as directory, torch.no_grad():
        write_datasets(Path(directory))
        model = load_model(weights, vae, Path(directory))
        batch = load_batch()

        result = {
            "num_atoms": batch.num_atoms,
            "num_crystals": torch.tensor(len(batch.num_atoms) - len(MOLECULES)),
            "molecules_z": torch.cat([torch.tensor(z) for z, _ in MOLECULES]),
            "molecules_pos": torch.cat([torch.tensor(pos) for _, pos in MOLECULES]),
        }
        result.update(autoencoder(model, batch))
        result.update(sampling(model, 0, "crystals"))
        result.update(sampling(model, 1, "molecules"))

    arrays = {
        name: (
            value.numpy().astype(np.float64)
            if value.is_floating_point()
            else value.numpy()
        )
        for name, value in result.items()
    }
    np.savez_compressed(DATA / "adit_reference.npz", **arrays)
    print({name: value.shape for name, value in arrays.items()})


if __name__ == "__main__":
    main()
