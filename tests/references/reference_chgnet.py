"""Reference graphs and energies of CHGNet, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
chgnet:

    uv run --isolated --no-project --with chgnet==0.4.2 \
        python tests/references/reference_chgnet.py

The structures are the ones of tests/data/orb_reference.npz. The result is
written to tests/data/chgnet_reference.npz and read by tests/test_chgnet.py.
"""

from pathlib import Path

import numpy as np
import torch
from chgnet.model import CHGNet
from pymatgen.core import Lattice, Structure

DATA = Path(__file__).resolve().parent.parent / "data"


def load_structures() -> list[Structure]:
    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    return [
        Structure(
            Lattice(data["lattice"][i].astype(np.float64)),
            data["z"][ptr[i] : ptr[i + 1]],
            data["x"][ptr[i] : ptr[i + 1]].astype(np.float64),
        )
        for i in range(len(data["num_atoms"]))
    ]


def main():
    model = CHGNet.load(model_name="0.3.0", use_device="cpu")
    model.graph_converter.set_isolated_atom_response("ignore")

    structures = load_structures()
    graphs = [model.graph_converter(structure) for structure in structures]

    num_edges, num_bonds, num_angles, cosine_sum = [], [], [], []
    for graph in graphs:
        num_edges.append(len(graph.atom_graph))
        num_bonds.append(len(graph.undirected2directed))
        num_angles.append(len(graph.bond_graph))

        # cosines of the angles, as computed by the encoders of the model
        cart = graph.atom_frac_coord @ graph.lattice
        vectors = cart[graph.atom_graph[:, 0]] - (
            cart[graph.atom_graph[:, 1]] + graph.neighbor_image @ graph.lattice
        )
        vectors = vectors / vectors.norm(dim=1, keepdim=True)
        if len(graph.bond_graph) > 0:
            cosine = (
                vectors[graph.bond_graph[:, 2]] * vectors[graph.bond_graph[:, 4]]
            ).sum(dim=1)
            cosine_sum.append(float(cosine.double().sum()))
        else:
            cosine_sum.append(0.0)

    with torch.no_grad():
        predictions = model.predict_graph(graphs, task="e", batch_size=8)

    num_atoms = np.array([len(structure) for structure in structures])
    energy_per_atom = np.array([float(p["e"]) for p in predictions], dtype=np.float64)

    np.savez(
        DATA / "chgnet_reference.npz",
        num_atoms=num_atoms,
        num_edges=np.array(num_edges),
        num_bonds=np.array(num_bonds),
        num_angles=np.array(num_angles),
        cosine_sum=np.array(cosine_sum),
        energy=energy_per_atom * num_atoms,
    )
    print("edges", num_edges)
    print("angles", num_angles)
    print("energy per atom", energy_per_atom.round(4))


if __name__ == "__main__":
    main()
