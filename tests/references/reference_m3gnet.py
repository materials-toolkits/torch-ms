"""Reference graphs and energies of M3GNet, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
matgl:

    uv run --isolated --no-project --python 3.12 --with matgl==4.1.0 \
        python tests/references/reference_m3gnet.py

The structures are the ones of tests/data/orb_reference.npz. The result is
written to tests/data/m3gnet_reference.npz and read by tests/test_m3gnet.py.
"""

from pathlib import Path

import matgl
import numpy as np
import torch
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph._compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph_torch,
)
from matgl.utils.maths import SPHERICAL_BESSEL_ROOTS
from pymatgen.core import Lattice, Structure

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL = "M3GNet-PES-MatPES-PBE-2025.2"


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
    potential = matgl.load_model(MODEL)
    potential.calc_forces = False
    potential.calc_stresses = False
    potential.eval()
    model = potential.model

    converter = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)

    num_edges, num_angles, cosine_sum, energy = [], [], [], []
    for structure in load_structures():
        graph, lattice, state = converter.get_graph(structure)

        # edges, angles and their cosines, as computed by the model
        pos = graph.frac_coords @ lattice[0]
        offshift = graph.pbc_offset @ lattice[0]
        vectors, distances = compute_pair_vector_and_distance(
            pos, graph.edge_index, offshift
        )
        line_graph = create_line_graph_torch(
            graph.edge_index, distances, vectors, pos.shape[0], model.threebody_cutoff
        )
        angles = compute_theta_and_phi(
            vectors, distances, line_graph["line_edge_index"]
        )

        num_edges.append(graph.edge_index.shape[1])
        num_angles.append(line_graph["line_edge_index"].shape[1])
        cosine_sum.append(float(angles["cos_theta"].double().sum()))

        with torch.no_grad():
            output = potential(graph, lattice, torch.tensor(state))
        energy.append(float(output[0]))

    np.savez(
        DATA / "m3gnet_reference.npz",
        num_edges=np.array(num_edges),
        num_angles=np.array(num_angles),
        cosine_sum=np.array(cosine_sum),
        energy=np.array(energy, dtype=np.float64),
        bessel_roots=SPHERICAL_BESSEL_ROOTS[:4, :4].double().numpy(),
    )
    print("edges", num_edges)
    print("angles", num_angles)
    print("energy", np.array(energy).round(3))


if __name__ == "__main__":
    main()
