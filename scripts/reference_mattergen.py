"""Reference outputs of the base model of MatterGen, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
MatterGen. Its package on PyPI pulls the whole of its evaluation (mattersim,
jupyterlab): clone it instead, at the commit the rewrite was written for, and
give its path to the script, which only imports the modules of the model and
of its sampling.

    git clone https://github.com/microsoft/mattergen.git
    git -C mattergen checkout 92423660a8bd70e83679086e88f88596d484dc16

    uv run --isolated --no-project --python 3.10 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html \
        --index-strategy unsafe-best-match \
        --with "torch==2.2.1+cpu" --with torch-scatter==2.1.2 \
        --with torch-sparse==0.6.18 --with torch-cluster==1.6.3 \
        --with torch-geometric==2.5.3 --with pytorch-lightning==2.0.6 \
        --with hydra-core==1.3.1 --with omegaconf==2.3.0 --with "numpy<2" \
        --with sympy --with scipy --with pymatgen==2024.6.10 \
        --with emmet-core==0.84.2 --with monty==2024.7.30 --with ase==3.23.0 \
        --with huggingface-hub --with fire --with tqdm --with "setuptools<81" \
        python scripts/reference_mattergen.py mattergen WEIGHTS

torch, pytorch-lightning, hydra-core, omegaconf and monty are the versions of
the pyproject.toml of MatterGen 1.0.3. The other ones are what its modules
import, at versions which install together: pymatgen of 2024.6, the last one
monty 2024.7.30 allows, and setuptools before 81, which still has the
pkg_resources lightning imports.

WEIGHTS is the directory of the two files of weights listed in
src/torchms/models/mattergen/config.yaml: the cache of torchms after a first
from_pretrained(), "~/.cache/torchms/mattergen/mattergen-base". The model is
loaded from it as MatterGen loads a checkpoint, with the mask of the elements
its script of generation adds. The result is written to
tests/data/mattergen_reference.npz and read by tests/test_mattergen.py.

The structures are the ones of tests/data/orb_reference.npz.

The network is very sensitive to the rounding of its inputs: a distance which
differs by its last bit changes its outputs by 1e-3. It is run a second time
in double precision, for the tests to check that the rewrite is as near to it
as MatterGen in single precision is. MatterGen does not run in double
precision as it is: two casts to single precision of the construction of its
graph are changed for this run only, see in_double_precision.
"""

import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"

# the mask of the elements of mattergen/scripts/generate.py
ELEMENT_MASK = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)

# numbers of atoms of the crystals of the sampling, which is run whole
SAMPLING_NUM_ATOMS = [4, 8, 2, 6]

# steps of the sampling which are saved, with what they draw: the first ones
# follow each other, the last one gives the crystals MatterGen returns
SAVED_STEPS = [0, 1, 2, 100, 300, 500, 700, 900, 998, 999]


def load_model(mattergen: Path, weights: Path):
    """DiffusionLightningModule of MatterGen, loaded from the directory of its weights."""
    sys.path.insert(0, str(mattergen))
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.common.utils.eval_utils import load_model_diffusion

    info = MatterGenCheckpointInfo(
        model_path=str(weights.resolve()),
        load_epoch="last",
        config_overrides=[ELEMENT_MASK],
    )

    return load_model_diffusion(info).to("cpu").eval()


def chemgraphs(lattice, x, z, num_atoms):
    """Batch of ChemGraph of MatterGen of crystals."""
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate

    ptr = np.concatenate([[0], np.cumsum(num_atoms)])

    return collate(
        [
            ChemGraph(
                pos=torch.as_tensor(x[ptr[i] : ptr[i + 1]], dtype=torch.float32),
                cell=torch.as_tensor(lattice[i : i + 1], dtype=torch.float32),
                atomic_numbers=torch.as_tensor(
                    z[ptr[i] : ptr[i + 1]], dtype=torch.long
                ),
                num_atoms=torch.tensor(int(num_atoms[i])),
                num_nodes=int(num_atoms[i]),
            )
            for i in range(len(num_atoms))
        ]
    )


def graph(model, batch) -> dict[str, np.ndarray]:
    """Edges and triplets of the graph GemNet-T builds, per structure."""
    from mattergen.common.utils.data_utils import frac_to_cart_coords_with_lattice

    gemnet = model.diffusion_module.model.gemnet
    num_atoms, cell = batch["num_atoms"], batch["cell"]
    positions = frac_to_cart_coords_with_lattice(batch["pos"], num_atoms, lattice=cell)
    edge_index, neighbors, distances, *_, id3_ca, _, _ = (
        gemnet.generate_interaction_graph(positions, cell, num_atoms, None, None, None)
    )
    edge_batch = batch.get_batch_idx("pos")[edge_index[1]]
    n_struct = len(num_atoms)

    return {
        "graph_num_edges": neighbors.numpy(),
        "graph_num_triplets": torch.bincount(
            edge_batch[id3_ca], minlength=n_struct
        ).numpy(),
        "graph_sum_distances": torch.zeros(n_struct, dtype=torch.float64)
        .index_add_(0, edge_batch, distances.double())
        .numpy(),
    }


def network(model, batch, prefix: str) -> dict[str, np.ndarray]:
    """Output of the denoiser on crystals at times from the first to the last."""
    n_struct = len(batch["num_atoms"])
    times = torch.linspace(1e-3, 1.0, n_struct)
    out = model.diffusion_module.model(batch, times)

    return {
        f"{prefix}_times": times.numpy(),
        f"{prefix}_pred_x": out["pos"].numpy(),
        f"{prefix}_pred_l": out["cell"].numpy(),
        f"{prefix}_pred_logits": out["atomic_numbers"].numpy(),
    }


class Recorder:
    """Records what torch draws while MatterGen samples, its scores and its posteriors."""

    def __init__(self, model, sampler):
        self.drawn: list[torch.Tensor] = []
        self.posteriors: list[torch.Tensor] = []
        self.scores: list = []
        self.sampler = sampler
        self.score_fn = sampler._score_fn
        self.functions = (torch.randn, torch.randn_like)
        self.categorical = torch.distributions.Categorical.sample
        self.d3pm = model.diffusion_module.corruption.discrete_corruptions[
            "atomic_numbers"
        ].d3pm
        self.posterior = self.d3pm.sample_and_compute_posterior_q

    def record(self, function):
        def wrapped(*args, **kwargs):
            self.drawn.append(function(*args, **kwargs))
            return self.drawn[-1]

        return wrapped

    def record_scores(self, *args, **kwargs):
        self.scores.append(self.score_fn(*args, **kwargs))
        return self.scores[-1]

    def record_posterior(self, *args, **kwargs):
        logits, samples = self.posterior(*args, **kwargs)
        self.posteriors.append(logits)
        return logits, samples

    def __enter__(self):
        torch.randn, torch.randn_like = map(self.record, self.functions)
        torch.distributions.Categorical.sample = self.record(self.categorical)
        self.d3pm.sample_and_compute_posterior_q = self.record_posterior
        self.sampler._score_fn = self.record_scores
        return self

    def __exit__(self, *error):
        torch.randn, torch.randn_like = self.functions
        torch.distributions.Categorical.sample = self.categorical
        self.d3pm.sample_and_compute_posterior_q = self.posterior
        del self.sampler._score_fn


def sampling(model, mattergen: Path) -> dict[str, np.ndarray]:
    """The whole sampling of a few crystals, with everything it draws."""
    import hydra
    from hydra.utils import instantiate
    from mattergen.common.data.collate import collate
    from mattergen.common.data.dataset import NumAtomsCrystalDataset

    config_dir = mattergen / "mattergen" / "sampling_conf"
    with hydra.initialize_config_dir(str(config_dir.resolve()), version_base=None):
        config = hydra.compose(config_name="default")
    sampler = instantiate(config.sampler_partial)(pl_module=model)

    dataset = NumAtomsCrystalDataset(num_atoms=np.array(SAMPLING_NUM_ATOMS))
    conditions = collate([dataset[i] for i in range(len(dataset))])

    with Recorder(model, sampler) as recorder:
        sample, mean, states = sampler.sample_with_record(conditions)

    # positions and cell of the prior, then at each step the noise of the
    # corrector of the positions and of the cell, of the predictor of the
    # positions and of the cell, and two draws of elements, of which only the
    # second one is used
    steps = sampler.N
    drawn, posteriors, scores = recorder.drawn, recorder.posteriors, recorder.scores
    n_struct, n_atoms = len(SAMPLING_NUM_ATOMS), sum(SAMPLING_NUM_ATOMS)
    assert len(drawn) == 2 + 6 * steps and len(posteriors) == steps
    assert len(scores) == 2 * steps
    assert len(states) == 2 * steps
    shapes = [(n_atoms, 3), (n_struct, 3, 3), (n_atoms, 3), (n_struct, 1, 1)]
    shapes += [(n_atoms,), (n_atoms,)]
    for i, shape in enumerate(shapes):
        assert drawn[2 + i].shape == shape, (i, drawn[2 + i].shape)

    def draw(step: int, position: int) -> torch.Tensor:
        return drawn[2 + 6 * step + position]

    # state before the corrector and before the predictor of each step, and
    # after the last step
    before = [states[2 * step] for step in SAVED_STEPS]
    corrected = [states[2 * step + 1] for step in SAVED_STEPS]
    after = [
        states[2 * step + 2] if step < steps - 1 else sample for step in SAVED_STEPS
    ]

    def saved(values) -> np.ndarray:
        return torch.stack(values).numpy()

    def fields(name: str, values) -> dict[str, np.ndarray]:
        return {
            f"{name}_x": saved([v["pos"] for v in values]),
            f"{name}_l": saved([v["cell"] for v in values]),
            f"{name}_z": saved([v["atomic_numbers"] for v in values]),
        }

    return {
        "sampling_num_atoms": np.array(SAMPLING_NUM_ATOMS),
        "sampling_steps": np.array(SAVED_STEPS),
        "sampling_times": torch.linspace(1.0, sampler._eps_t, steps)[
            SAVED_STEPS
        ].numpy(),
        **fields("before", before),
        **fields("corrected", corrected),
        **fields("after", after),
        "corrector_noise_x": saved([draw(s, 0) for s in SAVED_STEPS]),
        "corrector_noise_l": saved([draw(s, 1) for s in SAVED_STEPS]),
        "predictor_noise_x": saved([draw(s, 2) for s in SAVED_STEPS]),
        "predictor_noise_l": saved([draw(s, 3).flatten() for s in SAVED_STEPS]),
        "posterior_logits": saved([posteriors[s] for s in SAVED_STEPS]),
        # scores of the corrector and of the predictor of each step
        "corrector_score_x": saved([scores[2 * s]["pos"] for s in SAVED_STEPS]),
        "corrector_score_l": saved([scores[2 * s]["cell"] for s in SAVED_STEPS]),
        "predictor_score_x": saved([scores[2 * s + 1]["pos"] for s in SAVED_STEPS]),
        "predictor_score_l": saved([scores[2 * s + 1]["cell"] for s in SAVED_STEPS]),
        "predictor_score_logits": saved(
            [scores[2 * s + 1]["atomic_numbers"] for s in SAVED_STEPS]
        ),
        "mean_x": mean["pos"].numpy(),
        "mean_l": mean["cell"].numpy(),
        "mean_z": mean["atomic_numbers"].numpy(),
    }


class in_double_precision:  # pylint: disable=invalid-name
    """Context in which MatterGen computes in double precision.

    Its graph casts the images of the atoms to single precision twice: the
    cells of the radius graph of the Open Catalyst Project, and the images in
    get_pbc_distances. Here the radius graph is built in single precision,
    only its edges are used, and get_pbc_distances is the same function with
    the images in the precision of the lattice.
    """

    def __init__(self):
        import inspect

        import mattergen.common.gemnet.gemnet as gemnet
        import mattergen.common.utils.data_utils as data_utils

        self.gemnet, self.data_utils = gemnet, data_utils
        self.radius_graph = data_utils.radius_graph_pbc_ocp
        self.distances = gemnet.get_pbc_distances

        source = inspect.getsource(data_utils.get_pbc_distances).replace(
            "to_jimages.float()", "to_jimages.to(lattice.dtype)"
        )
        namespace = dict(vars(data_utils))
        exec(source, namespace)  # pylint: disable=exec-used
        self.distances_in_double = namespace["get_pbc_distances"]

    def radius_graph_in_single(self, pos, cell, **kwargs):
        torch.set_default_dtype(torch.float32)
        try:
            return self.radius_graph(pos=pos.float(), cell=cell.float(), **kwargs)
        finally:
            torch.set_default_dtype(torch.float64)

    def __enter__(self):
        self.data_utils.radius_graph_pbc_ocp = self.radius_graph_in_single
        self.gemnet.get_pbc_distances = self.distances_in_double
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *error):
        torch.set_default_dtype(torch.float32)
        self.data_utils.radius_graph_pbc_ocp = self.radius_graph
        self.gemnet.get_pbc_distances = self.distances


def network_in_double(model, batches, times) -> dict[str, np.ndarray]:
    """Output of the denoiser in double precision, on the same inputs."""
    denoiser = model.diffusion_module.model.double()
    result = {}

    with in_double_precision():
        for prefix, batch in batches.items():
            batch = batch.replace(
                pos=batch["pos"].double(), cell=batch["cell"].double()
            )
            out = denoiser(batch, torch.from_numpy(times).double())
            # rounded to single precision, which is enough for differences of
            # 1e-3 and halves the file
            result[f"{prefix}_double_x"] = out["pos"].float().numpy()
            result[f"{prefix}_double_l"] = out["cell"].float().numpy()
            result[f"{prefix}_double_logits"] = out["atomic_numbers"].float().numpy()

    denoiser.float()
    return result


def main():
    mattergen, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()

    torch.manual_seed(0)
    model = load_model(mattergen, weights)

    data = np.load(DATA / "orb_reference.npz")
    batch = chemgraphs(data["lattice"], data["x"], data["z"], data["num_atoms"])
    print("structures", len(data["num_atoms"]), "atoms", int(data["num_atoms"].sum()))

    with torch.no_grad():
        result = graph(model, batch)
        result.update(network(model, batch, "clean"))

        # the same with the noise of the diffusion at the time of each
        # structure, as in the training: elements are masked
        times = result["clean_times"]
        noisy = model.diffusion_module.corruption.sample_marginal(
            batch, torch.from_numpy(times)
        )
        result.update(
            noisy_x=noisy["pos"].numpy(),
            noisy_l=noisy["cell"].numpy(),
            noisy_z=noisy["atomic_numbers"].numpy(),
        )
        result.update(network(model, noisy, "noisy"))

        result.update(sampling(model, mattergen))
        result.update(network_in_double(model, {"clean": batch, "noisy": noisy}, times))

    np.savez_compressed(DATA / "mattergen_reference.npz", **result)
    print("edges", result["graph_num_edges"].tolist())
    for name in ("clean_pred_x", "clean_pred_l", "noisy_pred_x", "noisy_pred_l"):
        print(name, np.abs(result[name]).max(), np.abs(result[name]).mean())
    print("masked atoms", int((result["noisy_z"] == 101).sum()))
    print("final elements", result["mean_z"].tolist())
    print("final cells", result["mean_l"].round(2).tolist())


if __name__ == "__main__":
    main()
