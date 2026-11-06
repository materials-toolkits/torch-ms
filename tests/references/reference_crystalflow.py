"""Reference outputs of CrystalFlow, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
CrystalFlow. CrystalFlow is not on PyPI: clone it at the tag its checkpoints
were released with, and give its path to the script.

    git clone https://github.com/ixsluo/CrystalFlow.git
    git -C CrystalFlow checkout v1.0.0-alpha.1    # d77d6d0ac870d9bab5245276e9b60e569a6b50d8

    uv run --isolated --no-project --python 3.11 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html \
        --index-strategy unsafe-best-match \
        --with "torch==2.3.1+cpu" --with torch-geometric==2.5.3 \
        --with torch-scatter==2.1.2 --with torch-sparse==0.6.18 \
        --with lightning==2.3.2 --with pytorch-lightning==2.3.2 \
        --with hydra-core==1.3.2 --with omegaconf --with python-dotenv \
        --with p-tqdm --with pymatgen==2024.6.10 --with pyxtal --with smact \
        --with einops --with chemparse --with torchdyn --with "numpy<2" \
        --with scipy --with tqdm --with "setuptools<81" \
        python tests/references/reference_crystalflow.py CrystalFlow WEIGHTS

python, torch, torch-geometric and lightning are the versions of the README of
CrystalFlow, for the CPU; setuptools before 81 still has the pkg_resources
lightning imports.

WEIGHTS is the directory of one of the checkpoints of the release
v1.0.0-alpha.1 (ckpt-v1.0.0.zip), with its hparams.yaml: in
"~/.cache/torchms/crystalflow/<name>" after a first from_pretrained(), or
extracted from the archive. The model is loaded as scripts/eval_utils.py of
CrystalFlow loads it. The script finds which model it is given:

"CSP-mp-20", "CSP-mpts-52" (in CSP-mpts-52.zip of the same release),
"CSP-mp-calypso-60", "DNG-mp-20" or "DNG-mp-20-Ef", see MODELS: the result is
written to tests/data/crystalflow_<model>_reference.npz.

The structures are the 40 crystals of tests/data/orb_reference.npz. The result
is read by tests/models/test_crystalflow.py:

* the polar decomposition of their lattices, k, and the lattice CrystalFlow
  rebuilds from it;
* the vector field of the network on these crystals, at the times of TIMES,
  with the vectors of their elements for the generation, and with and
  without their conditions for a model with conditions;
* the sampling of the first NUM_SAMPLED crystals by CSPFlow.sample with the
  settings of the README (NUM_STEPS steps, the coordinates annealed with the
  slope ANNEAL_SLOPE): its initial lattices and coordinates, the inputs and the
  outputs of the network at the steps of SAVED_STEPS, and the final lattices
  and coordinates;
* the same sampling in double precision, from the same noise: the flow
  multiplies the rounding of float32, which tells it from a difference.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

TIMES = torch.linspace(0.01, 1.0, 40)

NUM_SAMPLED = 6
NUM_STEPS = 100
ANNEAL_SLOPE = 5.0
SAVED_STEPS = [1, 2, 3, 50, 99, 100]


class Recorder:
    """Records what torch draws while a model samples."""

    def __init__(self):
        self.drawn = []
        self.functions = (torch.randn, torch.rand)

    def record(self, function):
        def wrapped(*args, **kwargs):
            self.drawn.append(function(*args, **kwargs))
            return self.drawn[-1]

        return wrapped

    def __enter__(self):
        torch.randn, torch.rand = map(self.record, self.functions)
        return self.drawn

    def __exit__(self, *error):
        torch.randn, torch.rand = self.functions


def load_batch(indices=None):
    """Structures of the tests as a batch of CrystalFlow, and their lattices."""
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])
    indices = range(len(data["num_atoms"])) if indices is None else indices

    crystals = [
        Data(
            frac_coords=torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]]),
            atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
            num_atoms=int(data["num_atoms"][i]),
            num_nodes=int(data["num_atoms"][i]),
        )
        for i in indices
    ]

    return Batch.from_data_list(crystals), torch.from_numpy(
        data["lattice"][list(indices)]
    )


# the models of the checkpoints, by name of their directory: the file of the
# result, whether the model generates the elements, its conditions with the
# values given to the 40 crystals (linearly from the first to the second one),
# whether its coordinates are annealed (scripts/evaluate.py, not
# scripts/generation.py) and the guide factor of its sampling (both passes
# with and without the conditions are recorded when it is not 1)
MODELS = {
    "CSP-mp-20": ("crystalflow_csp_mp20_reference.npz", False, {}, True, None),
    "CSP-mpts-52": ("crystalflow_csp_mpts52_reference.npz", False, {}, True, None),
    "CSP-mp-calypso-60": (
        "crystalflow_csp_calypso60_pressure_reference.npz",
        False,
        {"pressure": (0.0, 200.0)},
        True,
        1.5,
    ),
    "DNG-mp-20": ("crystalflow_dng_mp20_reference.npz", True, {}, False, None),
    "DNG-mp-20-Ef": (
        "crystalflow_dng_mp20_formation_energy_reference.npz",
        True,
        {"formation_energy_per_atom": (-3.0, 0.0)},
        False,
        1.5,
    ),
}


def condition_values(conditions, count):
    """Value of each condition for each structure, in the units of the data."""
    return {
        name: torch.linspace(first, last, count)
        for name, (first, last) in conditions.items()
    }


def standardized(model, cfg, values):
    """The conditions as the batch of CrystalFlow holds them: standardized, (B, 1)."""
    return {
        name: torch.as_tensor(
            model.scalers[list(cfg.data.properties).index(name)].transform(value)
        ).view(-1, 1)
        for name, value in values.items()
    }


def network(model, cfg, batch, lattices, generation, conditions):
    """The polar decomposition of the lattices, and the network on the crystals."""
    from diffcsp.common.data_utils import (
        lattice_polar_build_torch,
        lattice_polar_decompose_torch,
    )

    k = lattice_polar_decompose_torch(lattices)
    matrix = lattice_polar_build_torch(k)
    atom_types = (
        model.type_encoding(batch.atom_types).float()
        if generation
        else batch.atom_types
    )
    inputs = dict(
        t=model.time_embedding(TIMES),
        atom_types=atom_types,
        frac_coords=batch.frac_coords,
        lattices_rep=k,
        num_atoms=batch.num_atoms,
        node2graph=batch.batch,
        lattices_mat=matrix,
    )
    result = {"k": k, "matrix": matrix, "times": TIMES}
    if generation:
        result["elements"] = atom_types

    def save(output, suffix):
        result[f"pred_l{suffix}"], result[f"pred_f{suffix}"] = output[0], output[1]
        if generation:
            result[f"pred_t{suffix}"] = output[2]

    def run(inputs, suffix):
        if conditions:
            values = condition_values(conditions, len(TIMES))
            result["conditions"] = torch.stack(list(values.values()), dim=1)
            scaled = {
                name: value.to(inputs["lattices_rep"].dtype)
                for name, value in standardized(model, cfg, values).items()
            }
            cemb = model.cond_emb(**scaled)
            ones = torch.ones(len(TIMES), dtype=cemb.dtype)
            save(model.decoder(**inputs, cemb=cemb, guide_indicator=ones), suffix)
            save(model.decoder(**inputs), "_free" + suffix)
        else:
            save(model.decoder(**inputs), suffix)

    run(inputs, "")
    # the same in double precision: the network multiplies the rounding of
    # float32 near the time 1
    with double_precision(model):
        run(
            {
                name: value.double() if value.is_floating_point() else value
                for name, value in inputs.items()
            },
            "_double",
        )

    return result


class double_precision:  # pylint: disable=invalid-name
    """The model in double precision, as the default dtype; float32 after.

    The Miller indices are an attribute of the network, not a buffer.
    """

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.double()
        rec_emb = self.model.decoder.rec_emb
        rec_emb.millerindices = rec_emb.millerindices.double()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *error):
        torch.set_default_dtype(torch.float32)
        rec_emb = self.model.decoder.rec_emb
        rec_emb.millerindices = rec_emb.millerindices.float()
        self.model.float()


def sampling(model, cfg, generation, conditions, anneal, guide_factor):
    """The whole sampling of the first crystals, with its noise and its steps."""
    batch, _ = load_batch(range(NUM_SAMPLED))
    values = condition_values(conditions, NUM_SAMPLED)
    for name, value in standardized(model, cfg, values).items():
        batch[name] = value
    if generation:
        del batch.atom_types, batch.frac_coords

    calls = []
    names = ("t", "atom_types", "frac_coords", "lattices_rep", "lattices_mat")
    hook = model.decoder.register_forward_hook(
        lambda module, args, kwargs, output: calls.append(
            (
                {name: kwargs[name].clone() for name in names},
                tuple(value.clone() for value in output),
                kwargs.get("cemb") is not None,
            )
        ),
        with_kwargs=True,
    )
    settings = dict(
        N=NUM_STEPS, anneal_coords=anneal, anneal_slope=ANNEAL_SLOPE if anneal else 0.0
    )
    if conditions:
        settings["guide_factor"] = guide_factor
    with Recorder() as drawn:
        final, _ = model.sample(batch, **settings)
    hook.remove()

    # the lattice (B, 6), the coordinates, then the vectors of the elements;
    # with guidance, the call without the conditions then the one with them
    passes = 2 if conditions else 1
    assert len(drawn) == (3 if generation else 2) and len(calls) == passes * NUM_STEPS
    result = {
        "sample_num_atoms": batch.num_atoms,
        "sample_l0": drawn[0],
        "sample_f0": drawn[1],
        "sample_steps": torch.tensor(SAVED_STEPS),
        "sample_lattice": final["lattices"],
        "sample_x": final["frac_coords"],
    }
    if generation:
        result["sample_t0"] = drawn[2]
        result["sample_z"] = final["atom_types"]
    else:
        result["sample_z"] = batch.atom_types
    if conditions:
        result["sample_conditions"] = torch.stack(list(values.values()), dim=1)
        result["guide_factor"] = torch.tensor(guide_factor)
    for step in SAVED_STEPS:
        for inputs, output, guided in calls[passes * (step - 1) : passes * step]:
            suffix = "_free" if conditions and not guided else ""
            if not suffix:
                result[f"sample_step{step}_l"] = inputs["lattices_rep"]
                result[f"sample_step{step}_f"] = inputs["frac_coords"]
                result[f"sample_step{step}_matrix"] = inputs["lattices_mat"]
                if generation:
                    result[f"sample_step{step}_t"] = inputs["atom_types"]
            result[f"sample_step{step}_pred_l{suffix}"] = output[0]
            result[f"sample_step{step}_pred_f{suffix}"] = output[1]
            if generation:
                result[f"sample_step{step}_pred_t{suffix}"] = output[2]

    # the same sampling in double precision, the model and its noise converted,
    # and the network in double precision on the states of the steps above
    replayed = iter([value.double() for value in drawn])

    def replay(*args, **kwargs):
        del args, kwargs
        return next(replayed)

    functions = (torch.randn, torch.rand)
    with double_precision(model):
        for name in conditions:
            batch[name] = batch[name].double()
        for step in SAVED_STEPS:
            for inputs, _, guided in calls[passes * (step - 1) : passes * step]:
                suffix = "_free" if conditions and not guided else ""
                kwargs = {
                    name: value.double() if value.is_floating_point() else value
                    for name, value in inputs.items()
                }
                if guided:
                    kwargs["cemb"] = model.cond_emb(
                        **{name: batch[name] for name in conditions}
                    )
                    kwargs["guide_indicator"] = torch.ones(len(batch.num_atoms))
                output = model.decoder(
                    num_atoms=batch.num_atoms, node2graph=batch.batch, **kwargs
                )
                result[f"sample_step{step}_pred_l{suffix}_double"] = output[0]
                result[f"sample_step{step}_pred_f{suffix}_double"] = output[1]
                if generation:
                    result[f"sample_step{step}_pred_t{suffix}_double"] = output[2]
        torch.randn = torch.rand = replay
        try:
            final_double, _ = model.sample(batch, **settings)
        finally:
            torch.randn, torch.rand = functions
    result["sample_lattice_double"] = final_double["lattices"]
    result["sample_x_double"] = final_double["frac_coords"]
    if generation:
        result["sample_z_double"] = final_double["atom_types"]

    return result


def main():
    project, weights = (Path(arg).resolve() for arg in sys.argv[1:3])
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "scripts"))
    os.environ.setdefault("PROJECT_ROOT", str(project))
    from eval_utils import load_model

    torch.manual_seed(0)
    model, _, cfg = load_model(weights)
    model.eval()
    output, generation, conditions, anneal, guide_factor = MODELS[weights.name]

    with torch.no_grad():
        batch, lattices = load_batch()
        result = network(model, cfg, batch, lattices, generation, conditions)
        result.update(
            sampling(model, cfg, generation, conditions, anneal, guide_factor)
        )

    arrays = {
        name: (
            value.numpy().astype(np.float64)
            if value.is_floating_point()
            else value.numpy()
        )
        for name, value in result.items()
    }
    np.savez_compressed(DATA / output, **arrays)
    print({name: value.shape for name, value in arrays.items()})


if __name__ == "__main__":
    main()
