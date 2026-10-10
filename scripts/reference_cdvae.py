"""Reference graphs and outputs of CDVAE, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
cdvae. cdvae is not on PyPI: clone it first, at the commit the rewrite was
written for.

    git clone https://github.com/txie-93/cdvae.git
    git -C cdvae checkout f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea

    uv run --isolated --no-project --python 3.8 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-1.8.0+cpu.html \
        --index-strategy unsafe-best-match \
        --with "torch==1.8.1+cpu" --with torch-scatter==2.0.8 \
        --with torch-sparse==0.6.12 --with torch-geometric==2.0.1 \
        --with pytorch-lightning==1.3.8 --with hydra-core==1.1.0 \
        --with pymatgen==2020.12.31 --with p-tqdm==1.3.3 --with python-dotenv \
        --with "numpy<1.24" --with scikit-learn --with pandas --with networkx \
        --with sympy --with "torchmetrics<0.8" --with "setuptools<60" \
        --with "ruamel.yaml<0.18" --with monty==2021.8.17 \
        python scripts/reference_cdvae.py cdvae WEIGHTS

WEIGHTS is the directory of the three files of the weights listed in
src/torchms/models/cdvae/config.yaml: the cache of torchms after a first
CdvaeModel.from_pretrained(), "~/.cache/torchms/cdvae/cdvae-mp20-cowboys".

The structures are the ones of tests/data/orb_reference.npz, with at most 20
atoms. The result is written to tests/data/cdvae_reference.npz and read by
tests/test_cdvae.py.

The checkpoint and the scaler are pickles of objects of pytorch-lightning,
omegaconf and cdvae: they are read with an unpickler which only builds tensors
and containers, as torchms does.
"""

import collections
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent.parent / "tests" / "data"
CHECKPOINT = "epoch=745-step=79076.ckpt"

# Langevin dynamics checked step by step: structures, noise levels and steps
NUM_DYNAMICS = 6
NUM_LEVELS = 5
NUM_STEPS = 2


class Placeholder:
    """Stands for an object of a class which is not imported."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __setstate__(self, state):
        self.state = state


class Unpickler(pickle.Unpickler):
    """Unpickler which only builds tensors, containers and placeholders."""

    ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch", "FloatStorage"): torch.FloatStorage,
        ("torch", "LongStorage"): torch.LongStorage,
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return self.ALLOWED[(module, name)]
        if (module, name) == ("copyreg", "_reconstructor"):
            return lambda cls, base, state: object.__new__(cls)

        return type(name, (Placeholder,), {})


PICKLE = types.SimpleNamespace(
    __name__="pickle",
    Unpickler=Unpickler,
    load=lambda file, **kwargs: Unpickler(file, **kwargs).load(),
)


def read(file_name):
    return torch.load(file_name, map_location="cpu", pickle_module=PICKLE)


def load_model(cdvae: Path, weights: Path):
    """Model of cdvae built from hparams.yaml, with the weights of the checkpoint."""
    import hydra
    from cdvae.common.data_utils import StandardScalerTorch
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

    model.load_state_dict(read(weights / CHECKPOINT)["state_dict"])

    scaler = read(weights / "lattice_scaler.pt").state
    model.lattice_scaler = StandardScalerTorch(scaler["means"], scaler["stds"])

    return model.eval()


def load_batch(max_atoms: int):
    """Structures of the tests as a batch of cdvae, with their CrystalNN graphs."""
    from cdvae.common.data_utils import build_crystal_graph
    from pymatgen.core import Lattice, Structure
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    index, graphs = [], []
    for i in range(len(data["num_atoms"])):
        if data["num_atoms"][i] > max_atoms:
            continue

        given = Structure(
            Lattice(data["lattice"][i].astype(np.float64)),
            data["z"][ptr[i] : ptr[i + 1]],
            data["x"][ptr[i] : ptr[i + 1]].astype(np.float64),
        )
        # cdvae describes a cell by its lengths and its angles
        crystal = Structure(
            Lattice.from_parameters(*given.lattice.parameters),
            given.species,
            given.frac_coords,
        )

        frac_coords, atom_types, lengths, angles, edges, images, num_atoms = (
            build_crystal_graph(crystal, "crystalnn")
        )
        index.append(i)
        graphs.append(
            Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(edges.T).contiguous(),
                to_jimages=torch.LongTensor(images),
                num_atoms=num_atoms,
                num_bonds=edges.shape[0],
                num_nodes=num_atoms,
            )
        )

    return np.array(index), Batch.from_data_list(graphs)


def per_structure(values, batch, n_struct):
    """Sum of values of the edges or of the atoms of each structure."""
    total = torch.zeros(n_struct, dtype=torch.float64)

    return total.index_add_(0, batch, values.double()).numpy()


def main():
    cdvae, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()
    os.environ["PROJECT_ROOT"] = str(cdvae)
    sys.path.insert(0, str(cdvae))

    from cdvae.common.data_utils import frac_to_cart_coords

    torch.manual_seed(0)
    model = load_model(cdvae, weights)
    index, batch = load_batch(model.hparams.max_atoms)
    n_struct = len(index)
    num_atoms = batch.num_atoms
    result = {"index": index}

    with torch.no_grad():
        # encoder, on the graphs of CrystalNN
        encoder = model.encoder
        *_, idx_kj, _ = encoder.triplets(batch.edge_index, batch.atom_types.size(0))
        edge_batch = batch.batch[batch.edge_index[1]]
        hidden = encoder(batch)
        mu, log_var = model.fc_mu(hidden), model.fc_var(hidden)

        result.update(
            lengths=batch.lengths.numpy(),
            angles=batch.angles.numpy(),
            # an edge of cdvae j, i with the image of j is the edge i → j
            sender=batch.edge_index[1].numpy().astype(np.int16),
            receiver=batch.edge_index[0].numpy().astype(np.int16),
            shift=batch.to_jimages.numpy().astype(np.int8),
            num_bonds=batch.num_bonds.numpy(),
            num_pairs=np.bincount(edge_batch[idx_kj].numpy(), minlength=n_struct),
            hidden=hidden.numpy(),
            mu=mu.numpy(),
            log_var=log_var.numpy(),
        )

        # numbers of atoms, cells and compositions decoded from the mean
        logits, _, lengths, angles, composition = model.decode_stats(mu, num_atoms)
        result.update(
            num_atoms_logits=logits.numpy(),
            decoded_lengths=lengths.numpy(),
            decoded_angles=angles.numpy(),
            composition_logits=composition.numpy(),
        )

        # decoder, on atoms moved at random: no two neighbours at the same
        # distance, the 20 nearest ones are well defined
        gemnet = model.decoder.gemnet
        noisy = (batch.frac_coords + 0.05 * torch.randn_like(batch.frac_coords)) % 1.0
        positions = frac_to_cart_coords(noisy, batch.lengths, batch.angles, num_atoms)
        edge_index, neighbors, distance, _, _, id3_ba, _, _ = (
            gemnet.generate_interaction_graph(
                positions, batch.lengths, batch.angles, num_atoms, None, None, None
            )
        )
        edge_batch = batch.batch[edge_index[1]]
        score, type_logits = model.decoder(
            mu, noisy, batch.atom_types, num_atoms, batch.lengths, batch.angles
        )

        result.update(
            noisy=noisy.numpy(),
            num_edges=neighbors.numpy(),
            distance_sum=per_structure(distance, edge_batch, n_struct),
            num_triplets=np.bincount(edge_batch[id3_ba].numpy(), minlength=n_struct),
            score=score.numpy(),
            types_logits=type_logits.numpy(),
        )

        # a few steps of the dynamics at the lowest noise levels, with the
        # atoms it starts from and the noise it adds
        first = batch.batch < NUM_DYNAMICS
        drawn = []

        def record(function):
            def wrapped(*args, **kwargs):
                drawn.append(function(*args, **kwargs))
                return drawn[-1]

            return wrapped

        model.sigmas.data = model.sigmas.data[-NUM_LEVELS:]
        uniform, normal = torch.rand, torch.randn_like
        torch.rand, torch.randn_like = record(uniform), record(normal)
        samples = model.langevin_dynamics(
            mu[:NUM_DYNAMICS],
            types.SimpleNamespace(
                n_step_each=NUM_STEPS,
                step_lr=1e-4,
                min_sigma=0,
                save_traj=False,
                disable_bar=True,
            ),
            num_atoms[:NUM_DYNAMICS],
            batch.atom_types[first],
        )
        torch.rand, torch.randn_like = uniform, normal

        result.update(
            dynamics_start=drawn[0].numpy(),
            dynamics_noise=torch.stack(drawn[1:]).numpy(),
            dynamics_end=samples["frac_coords"].numpy(),
        )

    np.savez_compressed(DATA / "cdvae_reference.npz", **result)
    print("structures", n_struct, "atoms", int(num_atoms.sum()))
    print("bonds", result["num_bonds"].tolist())
    print("edges", result["num_edges"].tolist())
    print("triplets", result["num_triplets"].tolist())
    print("mu", np.abs(result["mu"]).mean(), "score", np.abs(result["score"]).mean())


if __name__ == "__main__":
    main()
