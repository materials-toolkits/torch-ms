"""Reference graphs and predictions of ALIGNN, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
alignn:

    uv run --isolated --no-project --with alignn==2026.8.11 \
        python scripts/reference_alignn.py

The structures are the ones of tests/data/orb_reference.npz. The graphs and the
predictions are written to tests/data/alignn_reference.npz, read by
tests/test_alignn.py. The features of the elements, which are not stored with
the weights of the model, are written to
src/torchms/models/alignn/cgcnn_features.npz.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)
from alignn.pretrained import get_alignn2_model
from alignn.torch_graph_builder import build_pure_torch_graph
from ase.data import chemical_symbols
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes

ROOT = Path(__file__).parent.parent
DATA = ROOT / "tests" / "data"
MODEL = "formation_energy_peratom_radius"


def load_structures() -> list[Atoms]:
    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    return [
        Atoms(
            lattice_mat=data["lattice"][i].astype(np.float64),
            coords=data["x"][ptr[i] : ptr[i + 1]].astype(np.float64),
            elements=[chemical_symbols[z] for z in data["z"][ptr[i] : ptr[i + 1]]],
            cartesian=False,
        )
        for i in range(len(data["num_atoms"]))
    ]


def element_features() -> np.ndarray:
    """Features "cgcnn" of jarvis-tools for each atomic number, zeros if unknown."""
    features = np.zeros((119, 92), dtype=np.uint8)
    for z in range(1, 119):
        try:
            row = np.asarray(get_node_attributes(chemical_symbols[z], "cgcnn"))
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if row.shape == (92,):
            assert set(np.unique(row)) <= {0, 1}
            features[z] = row
    return features


def main():
    with tempfile.TemporaryDirectory() as cache:
        paths = get_alignn2_model(MODEL, cache_dir=cache)
        with open(paths["config.json"], encoding="utf-8") as fp:
            config = json.load(fp)
        model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**config["model"]))
        model.load_state_dict(
            torch.load(paths["best_model.pt"], map_location="cpu", weights_only=True)
        )
    model.eval()

    # graphs of the training of the model
    settings = {
        "two_body_cutoff": float(config["cutoff"]),
        "three_body_cutoff": float(config["three_body_cutoff"]),
        "max_neighbors": int(config["max_neighbors"]),
        "atom_features": config["atom_features"],
        "use_matscipy_topology": False,
    }
    print(settings)

    num_edges, num_angles, cosine_sum, prediction = [], [], [], []
    senders, receivers, shifts = [], [], []
    for atoms in load_structures():
        graph, line_graph = build_pure_torch_graph(atoms=atoms, **settings)
        lattice = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())

        with torch.no_grad():
            output = model((graph, line_graph, lattice))
        if isinstance(output, dict):
            output = output["out"]

        num_edges.append(graph.num_edges)
        num_angles.append(line_graph.num_edges)
        cosine_sum.append(float(line_graph.edata["h"].double().sum()))
        prediction.append(float(output))
        senders.append(graph.src.numpy())
        receivers.append(graph.dst.numpy())
        shifts.append(graph.edata["images"].round().long().numpy())

    np.savez_compressed(
        DATA / "alignn_reference.npz",
        num_edges=np.array(num_edges),
        num_angles=np.array(num_angles),
        cosine_sum=np.array(cosine_sum),
        prediction=np.array(prediction, dtype=np.float64),
        sender=np.concatenate(senders).astype(np.int16),
        receiver=np.concatenate(receivers).astype(np.int16),
        shift=np.concatenate(shifts).astype(np.int8),
    )
    np.savez_compressed(
        ROOT / "src/torchms/models/alignn/cgcnn_features.npz",
        features=element_features(),
    )
    print("edges", num_edges)
    print("angles", num_angles)
    print("prediction", np.array(prediction).round(3))


if __name__ == "__main__":
    main()
