"""Reference outputs of the models of DiffCSP++, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
DiffCSP++. DiffCSP++ is not on PyPI: clone it first, at the commit the rewrite
was written for.

    git clone https://github.com/jiaor17/DiffCSP-PP.git
    git -C DiffCSP-PP checkout e82e7ffa7cb2a383bde69b067a343ca137d73e47

    uv run --isolated --no-project --python 3.9 \\
        --extra-index-url https://download.pytorch.org/whl/cpu \\
        --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html \\
        --index-strategy unsafe-best-match \\
        --with "torch==2.1.2+cpu" --with torch-scatter==2.1.2 \\
        --with torch-geometric==2.5.2 --with pytorch-lightning==2.0.3 \\
        --with lightning-utilities==0.9.0 --with torchmetrics==1.1.2 \\
        --with hydra-core==1.3.2 --with omegaconf==2.3.0 \\
        --with pymatgen==2024.4.13 --with pyxtal==0.6.7 --with spglib==2.4.0 \\
        --with scipy==1.13.0 --with "numpy<2" --with einops==0.7.0 \\
        --with p-tqdm==1.4.0 --with python-dotenv==1.0.0 --with networkx \\
        --with pandas --with scikit-learn --with "setuptools<81" \\
        python tests/references/reference_diffcsp_pp.py DiffCSP-PP NAME

The versions are the ones of the environment.yml of DiffCSP++, with torch 2.1
for the CPU; setuptools before 81 still has pkg_resources, which
lightning-utilities and pyxtal import.

NAME is the name of weights of src/torchms/models/diffcsp_pp/config.yaml,
read from the cache of torchms after a first from_pretrained()
(~/.cache/torchms/diffcsp_pp/NAME); the script finds which model it is. The
result is written to tests/data/diffcsp_pp_<NAME without diffcsp-pp->_reference.npz
and read by tests/models/test_diffcsp_pp.py. The checkpoints only hold
tensors: they are read by torch.load as they are.

What is saved:

* the numbers k of the lattices of the 40 structures of
  tests/data/orb_reference.npz (CrystalFamily.m2v of de_so3), and the network
  on these structures and on noisy ones, at times from 1 to 1000, as in the
  training;
* the Wyckoff sites of SITES, as sample_api.get_data_from_syminfo builds them
  from pyxtal: operations, first atom of each site, pseudo-inverses;
* the sampling of these sites, with the states of some steps and their noise;
* for a model given the elements, the whole sampling of two of them, with the
  draws it uses, and the same sampling in double precision from the same
  draws: the steps multiply the rounding of float32.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# resolved now: importing diffcsp changes the working directory
DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = Path("~/.cache/torchms/diffcsp_pp").expanduser()

# structures of the sampling: a space group, the Wyckoff positions of its sites
# and their elements, one structure for each crystal family
SITES = [
    (58, ["2a", "2d", "4g"], ["Mn", "Li", "O"]),
    (194, "abff", ["Tm", "Tm", "Ni", "As"]),
    (225, "ab", ["Na", "Cl"]),
    (221, "abc", ["Sr", "Ti", "O"]),
    (12, "ai", ["Cu", "O"]),
    (2, "ii", ["Si", "O"]),
    (166, "abc", ["Bi", "Te", "Se"]),
    (139, "ade", ["Ba", "Fe", "As"]),
]

# times of the sampling whose step is saved, with its noise: the first ones
# follow each other
SAVED_TIMES = [999, 998, 997, 750, 500, 250, 100, 10, 2, 1]
# for the generation, fewer times and structures: a step has 100 values per atom
GENERATION_SAVED_TIMES = [999, 998, 500, 2, 1]
GENERATION_STRUCTURES = [0, 3, 6]
# structures sampled whole, with every draw which is used: a monoclinic and a
# triclinic one, whose lattices and coordinates are free
WHOLE = [4, 5]


def load_model(weights: Path):
    """Model of DiffCSP++ built from hparams.yaml, with the weights of the checkpoint."""
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(weights.resolve()), version_base=None):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

    checkpoint = torch.load(weights / "last.ckpt", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    return model.eval()


def load_structures():
    """Structures of the tests as a batch of DiffCSP++, and their cells."""
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    crystals = [
        Data(
            frac_coords=torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]]).float(),
            atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
            num_atoms=int(data["num_atoms"][i]),
            num_nodes=int(data["num_atoms"][i]),
        )
        for i in range(len(data["num_atoms"]))
    ]

    return Batch.from_data_list(crystals), torch.from_numpy(data["lattice"]).float()


def load_sites(generation: bool, carbon: bool, structures=None):
    """Batch of the sites of SITES, built by the sampling API of DiffCSP++.

    Args:
        generation: If True, without the elements.
        carbon: If True, every site is carbon.
        structures: Indices of the structures of SITES, all of them if None.
    """
    from sample_api import get_data_from_syminfo
    from torch_geometric.data import Batch

    crystals = []
    chosen = SITES if structures is None else [SITES[i] for i in structures]
    for spacegroup, letters, elements in chosen:
        if carbon:
            elements = ["C"] * len(elements)
        crystals.append(
            get_data_from_syminfo(spacegroup, letters, None if generation else elements)
        )

    return Batch.from_data_list(crystals)


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


class Replay:
    """Gives back recorded draws, in double precision, in place of new ones."""

    def __init__(self, drawn):
        self.drawn = iter([value.double() for value in drawn])
        self.functions = (torch.randn, torch.rand, torch.randn_like)

    def next(self, *args, **kwargs):
        del args, kwargs
        return next(self.drawn)

    def __enter__(self):
        torch.randn = torch.rand = torch.randn_like = self.next

    def __exit__(self, *error):
        torch.randn, torch.rand, torch.randn_like = self.functions


class double_precision:  # pylint: disable=invalid-name
    """The model in double precision, as the default dtype; float32 after."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.double()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *error):
        torch.set_default_dtype(torch.float32)
        self.model.float()


def network(model, generation):
    """The network on the structures of the tests, as they are and with noise."""
    import torch.nn.functional as F

    batch, lattices = load_structures()
    n_struct, num_atoms = batch.num_graphs, batch.num_atoms
    family = model.crystal_family

    k = family.m2v(family.de_so3(lattices.clone()))
    times = torch.linspace(1, model.beta_scheduler.timesteps, n_struct).long()
    alphas_cumprod = model.beta_scheduler.alphas_cumprod[times]
    signal, deviation = torch.sqrt(alphas_cumprod), torch.sqrt(1.0 - alphas_cumprod)
    sigmas = model.sigma_scheduler.sigmas[times].repeat_interleave(num_atoms)

    noisy_k = signal[:, None] * k + deviation[:, None] * torch.randn_like(k)
    noisy_x = batch.frac_coords + sigmas[:, None] * torch.randn_like(batch.frac_coords)
    noisy_x = noisy_x % 1.0
    result = dict(k=k, times=times, noisy_k=noisy_k, noisy_x=noisy_x)

    if generation:
        one_hot = F.one_hot(batch.atom_types - 1, num_classes=100).float()
        noisy_t = signal.repeat_interleave(num_atoms)[
            :, None
        ] * one_hot + deviation.repeat_interleave(num_atoms)[
            :, None
        ] * torch.randn_like(
            one_hot
        )
        result["noisy_t"] = noisy_t
        clean_elements, noisy_elements = one_hot, noisy_t
    else:
        clean_elements = noisy_elements = batch.atom_types

    def run(elements, x, cells, suffix):
        outputs = model.decoder(
            model.time_embedding(times), elements, x, cells, num_atoms, batch.batch
        )
        result[f"pred_l{suffix}"], result[f"pred_x{suffix}"] = outputs[:2]
        if generation:
            result[f"pred_t{suffix}"] = outputs[2]

    run(clean_elements, batch.frac_coords, k, "")
    run(noisy_elements, noisy_x, noisy_k, "_noisy")

    return result


def sampling(model, generation, carbon, step_lr):
    """The whole sampling of the sites, with its draws and some of its steps."""
    batch = load_sites(generation, carbon)
    result = {
        "spacegroup": batch.spacegroup,
        "num_atoms": batch.num_atoms,
        "ops": batch.ops,
        "ops_inv": batch.ops_inv,
        "anchor_index": batch.anchor_index,
        "atom_types": batch.atom_types.long(),
    }

    # what the network is given at each of its calls: two per step
    given = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, arguments: given.append(arguments[1:4])
    )
    with Recorder() as drawn:
        outputs, trajectory = model.sample(batch, step_lr=step_lr)
    hook.remove()

    start = model.beta_scheduler.timesteps - 1
    # the start: coordinates, a lattice and with the generation the elements;
    # then at each step but the last one the noise of the corrector, of the
    # lattice, of the predictor and of the elements; only the one of the
    # lattice at the time 1
    per_step = 4 if generation else 3
    first = 3 if generation else 2
    assert len(drawn) == first + per_step * (start - 1) + 1
    assert len(given) == 2 * start

    def noise(time, position):
        """Noise of a step: 0 corrector, 1 lattice, 2 predictor, 3 elements."""
        if time == 1:
            if position == 1:
                return drawn[first + per_step * (start - 1)]
            return torch.zeros_like(drawn[(0, 1, 0, 2)[position]])

        return drawn[first + per_step * (start - time) + position]

    def state(time):
        """Lattice, coordinates and elements of a time, as the network got them."""
        elements, x, k = given[2 * (start - time)]
        return k, x, elements

    def next_state(time):
        """The state after the step of a time: the end of the sampling after 1.

        The vectors of the elements at the end are not returned by DiffCSP++,
        only their atomic numbers.
        """
        if time == 1:
            return outputs["crys_fam"], outputs["frac_coords"], None
        return state(time - 1)

    saved_times = GENERATION_SAVED_TIMES if generation else SAVED_TIMES
    for index, key in enumerate(["k", "x", "t"] if generation else ["k", "x"]):
        result[f"state_{key}"] = torch.stack([state(t)[index] for t in saved_times])
        result[f"next_{key}"] = torch.stack(
            [next_state(t)[index] for t in saved_times if key != "t" or t > 1]
        )
    names = ["corrector", "lattice", "predictor"] + (["elements"] if generation else [])
    for position, name in enumerate(names):
        result[f"noise_{name}"] = torch.stack(
            [noise(time, position) for time in saved_times]
        )
    result["saved_times"] = torch.tensor(saved_times)
    result["initial_x"], result["initial_k"] = drawn[0], drawn[1]
    if generation:
        result["initial_t"] = drawn[2]
    result["final_lattice"] = trajectory["all_lattices"][-1]
    result["final_k"] = outputs["crys_fam"]
    result["final_x"] = outputs["frac_coords"]
    if generation:
        result["final_z"] = outputs["atom_types"]

    if generation:
        # only some structures are kept in the states of the steps
        ptr = np.concatenate([[0], np.cumsum(batch.num_atoms.numpy())])
        atoms = np.concatenate(
            [np.arange(ptr[i], ptr[i + 1]) for i in GENERATION_STRUCTURES]
        )
        for key in ("state_t", "next_t", "noise_elements"):
            result[key] = result[key][:, atoms]
        result["generation_atoms"] = torch.from_numpy(atoms)

    return result


def whole_sampling(model, carbon, step_lr):
    """The whole sampling of the structures WHOLE, with every draw which is used.

    The noise of the coordinates is only read at the first atom of each site:
    the draws of these atoms are saved. The same sampling in double precision,
    from the same draws, tells how much the steps multiply the rounding.
    """
    batch = load_sites(False, carbon, WHOLE)
    anchors = torch.unique(batch.anchor_index)

    with Recorder() as drawn:
        outputs, _ = model.sample(batch, step_lr=step_lr)
    start = model.beta_scheduler.timesteps - 1
    steps = drawn[2:]
    # at each time but 1: corrector, lattice, predictor; at 1 the lattice only
    corrector = torch.stack(steps[0:-1:3] + [torch.zeros_like(drawn[0])])
    lattice = torch.stack(steps[1:-1:3] + [steps[-1]])
    predictor = torch.stack(steps[2:-1:3] + [torch.zeros_like(drawn[0])])
    assert len(corrector) == len(lattice) == len(predictor) == start

    with double_precision(model), Replay(drawn):
        for name in ("ops", "ops_inv"):
            batch[name] = batch[name].double()
        outputs_double, _ = model.sample(batch, step_lr=step_lr)

    return {
        "whole_structures": torch.tensor(WHOLE),
        "whole_anchors": anchors,
        "whole_initial_x": drawn[0][anchors],
        "whole_initial_k": drawn[1],
        "whole_corrector": corrector[:, anchors],
        "whole_lattice": lattice,
        "whole_predictor": predictor[:, anchors],
        "whole_final_k": outputs["crys_fam"],
        "whole_final_x": outputs["frac_coords"],
        "whole_final_k_double": outputs_double["crys_fam"],
        "whole_final_x_double": outputs_double["frac_coords"],
    }


def main():
    diffcsp_pp, name = Path(sys.argv[1]).resolve(), sys.argv[2]
    os.environ["PROJECT_ROOT"] = str(diffcsp_pp)
    sys.path[:0] = [str(diffcsp_pp), str(diffcsp_pp / "scripts")]

    from eval_utils import recommand_step_lr

    torch.manual_seed(0)
    model = load_model(CACHE / name)
    generation = bool(model.decoder.pred_type)
    carbon = "carbon" in name
    dataset = {"mp20": "mp_20", "mpts52": "mpts_52", "perov5": "perov_5"}
    dataset.update(carbon24="carbon_24")
    task = "gen" if generation or carbon else "csp"
    step_lr = recommand_step_lr[task][dataset[name.split("-")[-1]]]
    print(name, "generation" if generation else "prediction", "step", step_lr)

    with torch.no_grad():
        result = network(model, generation)
        result.update(sampling(model, generation, carbon, step_lr))
        if not generation:
            result.update(whole_sampling(model, carbon, step_lr))
    result["step_lr"] = torch.tensor(step_lr)

    output = (
        DATA
        / f"diffcsp_pp_{name.removeprefix('diffcsp-pp-').replace('-', '_')}_reference.npz"
    )
    np.savez_compressed(
        output,
        **{
            key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        },
    )
    print(output, sum(np.asarray(v).size for v in result.values()), "values")
    for key in ("pred_l", "pred_x", "pred_x_noisy"):
        print(key, float(result[key].abs().max()))
    print("final k", result["final_k"].numpy().round(3).tolist())
    if "whole_final_k" in result:
        gap = result["whole_final_k"].double() - result["whole_final_k_double"]
        print("whole sampling, k single against double", float(gap.abs().max()))


if __name__ == "__main__":
    main()
