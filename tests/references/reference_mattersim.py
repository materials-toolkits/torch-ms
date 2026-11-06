"""Reference energies of MatterSim, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
MatterSim:

    uv run --isolated --no-project --python 3.12 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        --with mattersim==1.2.5 --with "torch==2.8.0+cpu" \
        --with "torchvision==0.23.0+cpu" --with "torchaudio==2.8.0+cpu" \
        --with pymatgen==2026.10.24 --with torch-geometric==2.8.1 \
        python tests/references/reference_mattersim.py WEIGHTS

mattersim 1.2.5 is its last release (tag v1.2.5, commit
40a1eb8f1189a53af310957b4f2c5dfbfe68d647); it needs torch 2.8 or later through
torch-sim. pymatgen builds the graph (find_points_in_spheres).

WEIGHTS is the file listed in src/torchms/models/mattersim/config.yaml,
"~/.cache/torchms/mattersim/mattersim-v1.0.0-1M.pth" after a first
from_pretrained(): the one MatterSim loads by default. The model is loaded by
Potential.from_checkpoint and run as its calculator runs it, on graphs built
by build_dataloader and batch_to_dict, the default path of MatterSimCalculator,
one structure per batch as the calculator: mattersim 1.2.5 does not offset
three_body_indices when it batches several structures, which gives wrong
energies.

The structures are the 40 crystals of tests/data/orb_reference.npz. The result
is written to tests/data/mattersim_reference.npz and read by
tests/models/test_mattersim.py: the energy of each structure in float32 and in
float64 (the model and its inputs converted), the energy of each atom, the
number of bonds and of triplets of each structure, and the sum of the lengths
of its bonds.
"""

import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"


def load_atoms():
    """The crystals of the tests as ase Atoms."""
    from ase import Atoms

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


def main():
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import Potential, batch_to_dict

    weights = Path(sys.argv[1]).expanduser().resolve()
    potential = Potential.from_checkpoint(
        load_path=str(weights), device="cpu", load_training_state=False
    )
    model = potential.model.eval()
    args = model.get_model_args()

    # one structure per batch, as MatterSimCalculator: mattersim 1.2.5 does not
    # offset three_body_indices when it batches several structures, which
    # gives wrong energies (up to 29 eV on these structures)
    loader = build_dataloader(
        load_atoms(),
        model_type="m3gnet",
        cutoff=args["cutoff"],
        threebody_cutoff=args["threebody_cutoff"],
        batch_size=1,
        only_inference=True,
    )

    per_atom = []
    hook = model.normalizer.register_forward_hook(
        lambda module, inputs, output: per_atom.append(output.detach().clone())
    )
    rows = {
        name: [] for name in ("energy", "num_edges", "num_triplets", "edge_length_sum")
    }
    graphs = []
    with torch.no_grad():
        for batch in loader:
            graph = batch_to_dict(batch, device="cpu")
            graphs.append(graph)
            energy = potential.forward(
                dict(graph), include_forces=False, include_stresses=False
            )
            edge_index = graph["edge_index"]
            vectors = graph["atom_pos"][edge_index[0]] - (
                graph["atom_pos"][edge_index[1]]
                + graph["pbc_offsets"] @ graph["cell"][0]
            )
            rows["energy"].append(energy["total_energy"])
            rows["num_edges"].append(torch.tensor([edge_index.shape[1]]))
            rows["num_triplets"].append(
                torch.tensor([graph["three_body_indices"].shape[0]])
            )
            rows["edge_length_sum"].append(
                vectors.double().norm(dim=1).sum().reshape(1)
            )
    hook.remove()

    model.double()
    with torch.no_grad():
        energy_double = [
            model(
                {
                    k: v.double() if v.is_floating_point() else v
                    for k, v in graph.items()
                }
            )
            for graph in graphs
        ]

    result = {name: torch.cat(values) for name, values in rows.items()}
    result["energy_double"] = torch.cat(energy_double)
    result["atom_energy"] = torch.cat(per_atom)
    arrays = {
        name: (
            value.numpy().astype(np.float64)
            if value.is_floating_point()
            else value.numpy()
        )
        for name, value in result.items()
    }
    np.savez_compressed(DATA / "mattersim_reference.npz", **arrays)
    print({name: value.shape for name, value in arrays.items()})
    print(arrays["energy"][:5], arrays["num_edges"][:5], arrays["num_triplets"][:5])


if __name__ == "__main__":
    main()
