"""Reference outputs of FlowMM, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
FlowMM. FlowMM is not on PyPI: clone it with its submodules (riemannian-fm,
DiffCSP-official, cdvae) at the commit the rewrite was written for, and give
its path to the script.

    git clone https://github.com/facebookresearch/flowmm.git
    git -C flowmm checkout 6a96aec3b6eba89f6fa07436f0c8837979abb285
    git -C flowmm submodule update --init

    uv run --isolated --no-project --python 3.9 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html \
        --index-strategy unsafe-best-match \
        --with "torch==2.1.0+cpu" --with torch-geometric==2.4.0 \
        --with torch-scatter==2.1.2 --with torch-sparse==0.6.18 \
        --with pytorch-lightning==1.8.5.post0 --with torchmetrics==1.2.1 \
        --with hydra-core==1.2.0 --with omegaconf --with geoopt==0.5.0 \
        --with numpy==1.26.4 --with pandas==2.2.0 --with pymatgen==2023.10.11 \
        --with pyxtal==0.6.1 --with p-tqdm==1.3.3 --with networkx \
        --with einops==0.7.0 --with scikit-learn==1.4.0 \
        --with torchdiffeq==0.2.3 --with matplotlib --with python-dotenv==1.0.1 \
        --with click==8.1.7 --with tqdm==4.66.2 --with "setuptools<70" \
        python tests/references/reference_flowmm.py flowmm WEIGHTS

torch, torch-geometric, pytorch-lightning, torchmetrics, hydra-core, geoopt,
numpy, pandas, pymatgen and pyxtal are the versions of the environment.yml of
FlowMM, for the CPU. Its modules of meshes, maps and logging import igl, evtk,
geopandas, shapely and wandb, which the model does not use: they are replaced
by empty modules before the import.

WEIGHTS is the directory of the two files listed in
src/torchms/models/flowmm/config.yaml, "~/.cache/torchms/flowmm" after a
first from_pretrained(): flowmm_mp20.ckpt and config.yaml. FlowMM finds the
configuration of a checkpoint in a directory .hydra next to it: the script
copies both to a temporary directory laid out so, and loads the model with
flowmm.model.eval_utils.load_model, in evaluation mode, which puts the moving
average of the weights in the network.

The structures are the 40 crystals of tests/data/orb_reference.npz. The result
is written to tests/data/flowmm_reference.npz and read by
tests/models/test_flowmm.py:

* the crystals in the representation of FlowMM: analog bits of the
  elements, lattice parameters with the angles in the unconstrained space,
  and the lattice matrix it builds from them;
* the vector field on these crystals at the times of TIMES;
* the sampling of NUM_ATOMS by gen_sample, with 1000 steps: its initial
  state, the state and the vector field of each step of SAVED_STEPS, the
  final state and the crystals read from it;
* the vector field of FlowMM in double precision on the states of these
  steps, the model and the states converted to float64 (FlowMM keeps the
  frequencies of its Fourier basis and the bounds of the angles in float32),
  which tells the rounding of float32 near the time 1.
"""

import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

# times of the vector field on the crystals of the tests, one per crystal. Near
# the time 1 the network multiplies the rounding of float32: these times are
# checked on the sampling, against FlowMM in double precision
TIMES = torch.linspace(0.0, 0.95, 40)

# numbers of atoms of the sampling, one structure of a single atom
NUM_ATOMS = [4, 8, 1, 12, 20, 6]
NUM_STEPS = 1000
SAVED_STEPS = [0, 1, 2, 500, 998, 999]


def import_flowmm(project: Path):
    """Modules of FlowMM, with empty ones for what the model does not use."""
    for path in ("src", "remote/riemannian-fm", "remote/DiffCSP-official"):
        sys.path.insert(0, str(project / path))
    for name in ("igl", "evtk", "geopandas", "shapely", "shapely.geometry", "wandb"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["evtk"].hl = sys.modules["evtk"].vtk = None
    sys.modules["shapely.geometry"].Point = None
    os.environ.setdefault("PROJECT_ROOT", str(project))


def load_model(weights: Path, directory: Path):
    """Model of FlowMM as its scripts of evaluation load it."""
    from flowmm.model.eval_utils import (
        load_model as load,
        register_omega_conf_resolvers,
    )

    (directory / ".hydra").mkdir()
    (directory / "checkpoints").mkdir()
    shutil.copy(weights / "config.yaml", directory / ".hydra" / "config.yaml")
    shutil.copy(
        weights / "flowmm_mp20.ckpt", directory / "checkpoints" / "flowmm_mp20.ckpt"
    )

    register_omega_conf_resolvers()
    cfg, model = load(directory / "checkpoints" / "flowmm_mp20.ckpt")

    return cfg, model.eval()


def georep(model, flat, dims, mask):
    """Elements, coordinates and lattice of a flat representation of FlowMM."""
    a, f, l = model.manifold_getter.flatrep_to_georep(flat, dims, mask)
    return {"bits": a, "x": f, "lattice": l}


def crystals(model):
    """The crystals of the tests, and the vector field on them."""
    from torch_geometric.data import Batch, Data

    from diffcsp.common.data_utils import lattices_to_params_shape

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])
    lattices = torch.from_numpy(data["lattice"])
    lengths, angles = lattices_to_params_shape(lattices)
    batch = Batch.from_data_list(
        [
            Data(
                atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
                frac_coords=torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]]),
                num_atoms=torch.tensor([int(n)]),
                num_nodes=int(n),
            )
            for i, n in enumerate(data["num_atoms"])
        ]
    )

    x, manifold, dims, mask = model.manifold_getter(
        batch.batch,
        batch.atom_types,
        batch.frac_coords,
        lengths,
        angles,
        split_manifold=False,
    )
    result = {"crystals_lengths": lengths, "crystals_angles": angles}
    result.update({f"crystals_{k}": v for k, v in georep(model, x, dims, mask).items()})
    _, _, matrices = model.manifold_getter.flatrep_to_crystal(x, dims, mask)
    result["crystals_matrix"] = matrices

    # one time per crystal: the field of each crystal alone, at its time
    fields = []
    for i in range(len(TIMES)):
        sub = Batch.from_data_list([batch.get_example(i)])
        xi, mi, di, maski = model.manifold_getter(
            sub.batch,
            sub.atom_types,
            sub.frac_coords,
            lengths[i : i + 1],
            angles[i : i + 1],
            split_manifold=False,
        )
        v = model.vecfield(
            num_atoms=sub.num_atoms,
            node2graph=sub.batch,
            dims=di,
            mask_a_or_f=maski,
            t=TIMES[i].reshape(1, 1),
            x=xi,
            manifold=mi,
        )
        fields.append(georep(model, v, di, maski))
    result["crystals_times"] = TIMES
    for key in ("bits", "x", "lattice"):
        result[f"crystals_v_{key}"] = torch.cat([field[key] for field in fields])

    return result


def manifold_double(model, node2graph):
    """The manifolds of a batch of empty structures, as gen_sample builds them."""
    _, manifold, *_ = model.manifold_getter.from_empty_batch(
        node2graph, 3, split_manifold=False
    )
    return manifold


def sampling(model, cfg):
    """The sampling of structures of NUM_ATOMS atoms, with its steps."""
    num_atoms = torch.tensor(NUM_ATOMS)
    node2graph = torch.arange(len(NUM_ATOMS)).repeat_interleave(num_atoms)
    *_, dims, mask = model.manifold_getter.from_empty_batch(
        node2graph, 3, split_manifold=False
    )

    calls = []
    network = model.model
    forward = network.forward

    def recorded(*args, **kwargs):
        v = forward(*args, **kwargs)
        calls.append((kwargs["t"].clone(), kwargs["x"].clone(), v.clone()))
        return v

    network.forward = recorded
    try:
        x1 = model.gen_sample(node2graph, 3, num_steps=NUM_STEPS)
    finally:
        network.forward = forward
    assert len(calls) == NUM_STEPS

    result = {"sample_num_atoms": num_atoms, "sample_steps": torch.tensor(SAVED_STEPS)}
    result.update(
        {f"sample_x0_{k}": v for k, v in georep(model, calls[0][1], dims, mask).items()}
    )
    for step in SAVED_STEPS:
        t, x, v = calls[step]
        result[f"sample_step{step}_t"] = t.reshape(())
        result.update(
            {
                f"sample_step{step}_x_{k}": u
                for k, u in georep(model, x, dims, mask).items()
            }
        )
        result.update(
            {
                f"sample_step{step}_v_{k}": u
                for k, u in georep(model, v, dims, mask).items()
            }
        )
        after = calls[step + 1][1] if step + 1 < NUM_STEPS else x1
        result.update(
            {
                f"sample_step{step}_next_{k}": u
                for k, u in georep(model, after, dims, mask).items()
            }
        )
    result.update(
        {f"sample_x1_{k}": v for k, v in georep(model, x1, dims, mask).items()}
    )

    # the vector field in double precision on the same states
    network.double()
    try:
        for step in SAVED_STEPS:
            t, x, _ = calls[step]
            v = network(
                num_atoms=num_atoms,
                node2graph=node2graph,
                dims=dims,
                mask_a_or_f=mask,
                t=t.double(),
                x=x.double(),
                manifold=manifold_double(model, node2graph),
            )
            result.update(
                {
                    f"sample_step{step}_double_v_{k}": u
                    for k, u in georep(model, v, dims, mask).items()
                }
            )
    finally:
        network.float()

    bits, frac_coords, lattices = model.manifold_getter.flatrep_to_crystal(
        x1, dims, mask
    )
    result["sample_z"] = model.manifold_getter._inverse_atomic_bits(bits)
    result["sample_matrix"] = lattices
    assert cfg.integrate.inference_anneal_slope == 0.0

    return result


def main():
    project, weights = (Path(arg).resolve() for arg in sys.argv[1:3])
    import_flowmm(project)
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as directory, torch.no_grad():
        cfg, model = load_model(weights, Path(directory))
        result = crystals(model)
        result.update(sampling(model, cfg))

    arrays = {
        name: (
            value.numpy().astype(np.float64)
            if value.is_floating_point()
            else value.numpy()
        )
        for name, value in result.items()
    }
    np.savez_compressed(DATA / "flowmm_reference.npz", **arrays)
    print({name: value.shape for name, value in arrays.items()})


if __name__ == "__main__":
    main()
