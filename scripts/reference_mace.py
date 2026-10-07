"""Converted weights and reference energies of MACE, for the rewrite of pycrystalgen.

Run once, outside of the environment of the project, which does not depend on
mace:

    uv run --isolated --no-project --python 3.11 --with mace-torch==0.3.16 \
        python scripts/reference_mace.py

The weights of MACE are a pickled model which only mace can read. They are
converted to a file of tensors, written to the cache of pycrystalgen
("$PYCRYSTALGEN_CACHE", "$XDG_CACHE_HOME/pycrystalgen" or
"~/.cache/pycrystalgen") as mace/mace-mp-0-small.pt, which MaceEnergyModel
loads. It holds the parameters of the model, the tensors coupling its
features, which mace stores with the weights, and two constants of e3nn.

The energies of the structures of tests/data/orb_reference.npz are written to
tests/data/mace_reference.npz and read by tests/test_mace.py. With --debug, the
features of the first structures after each block are written next to the
converted weights.
"""

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

DATA = Path(__file__).parent.parent / "tests" / "data"
URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-10-mace-128-L0_energy_epoch-249.model"
)
SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def cache_directory() -> Path:
    cache = os.environ.get("PYCRYSTALGEN_CACHE")
    if cache is not None:
        return Path(cache)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pycrystalgen"


def load_structures() -> list[Atoms]:
    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    return [
        Atoms(
            numbers=data["z"][ptr[i] : ptr[i + 1]],
            scaled_positions=data["x"][ptr[i] : ptr[i + 1]].astype(np.float64),
            cell=data["lattice"][i].astype(np.float64),
            pbc=True,
        )
        for i in range(len(data["num_atoms"]))
    ]


def normalization_constants(model: torch.nn.Module) -> dict[str, float]:
    """Constants of the normalized activations of e3nn, computed from a sample."""
    constants = {}
    for name, module in model.named_modules():
        cst = getattr(module, "cst", None)
        if isinstance(cst, (float, torch.Tensor)):
            constants[name] = float(cst)
    return constants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    directory = cache_directory() / "mace"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "2023-12-10-mace-128-L0_energy_epoch-249.model"
    if not source.exists():
        urllib.request.urlretrieve(URL, source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SHA256

    model = torch.load(source, map_location="cpu", weights_only=False).double()
    constants = normalization_constants(model)
    print("constants", constants)

    converted = {
        "state_dict": {name: value.clone() for name, value in model.state_dict().items()},
        "avg_num_neighbors": float(model.interactions[0].avg_num_neighbors),
        "activation_constants": constants,
        "source": URL,
        "source_sha256": SHA256,
    }
    torch.save(converted, directory / "mace-mp-0-small.pt")

    calculator = MACECalculator(
        model_paths=str(source), device="cpu", default_dtype="float64"
    )
    structures = load_structures()

    energy, num_edges = [], []
    for atoms in structures:
        atoms.calc = calculator
        energy.append(atoms.get_potential_energy())
        batch = calculator._atoms_to_batch(atoms)  # pylint: disable=protected-access
        num_edges.append(batch["edge_index"].shape[1])

    np.savez(
        DATA / "mace_reference.npz",
        energy=np.array(energy, dtype=np.float64),
        num_edges=np.array(num_edges),
    )
    print("edges", num_edges)
    print("energy", np.array(energy).round(3))

    if args.debug:
        debug = {}
        for index in (0, 3):
            data = calculator._atoms_to_batch(structures[index]).to_dict()  # pylint: disable=protected-access
            with torch.no_grad():
                vectors = data["positions"][data["edge_index"][1]] - data["positions"][data["edge_index"][0]] + data["shifts"]
                lengths = vectors.norm(dim=-1, keepdim=True)
                attrs = data["node_attrs"]
                feats = model.node_embedding(attrs)
                debug[f"{index}_embedding"] = feats.numpy()
                edge_attrs = model.spherical_harmonics(vectors)
                edge_feats, cut = model.radial_embedding(lengths, attrs, data["edge_index"], model.atomic_numbers)
                debug[f"{index}_edge_index"] = data["edge_index"].numpy()
                debug[f"{index}_vectors"] = vectors.numpy()
                debug[f"{index}_edge_attrs"] = edge_attrs.numpy()
                debug[f"{index}_edge_feats"] = edge_feats.numpy()
                for layer, (interaction, product) in enumerate(zip(model.interactions, model.products)):
                    debug[f"{index}_tp_weights_{layer}"] = interaction.conv_tp_weights(edge_feats).numpy()
                    message, sc = interaction(node_attrs=attrs, node_feats=feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=data["edge_index"], cutoff=cut, first_layer=layer == 0)
                    debug[f"{index}_message_{layer}"] = message.numpy()
                    debug[f"{index}_sc_{layer}"] = sc.numpy()
                    contracted = product.symmetric_contractions(message, attrs)
                    debug[f"{index}_contracted_{layer}"] = contracted.numpy()
                    feats = product(node_feats=message, sc=sc, node_attrs=attrs)
                    debug[f"{index}_product_{layer}"] = feats.numpy()
                    debug[f"{index}_readout_{layer}"] = model.readouts[layer](feats).numpy()
        np.savez(directory / "mace_debug.npz", **debug)


if __name__ == "__main__":
    main()
