"""What the layers of CSPNet in DiffCSP receive and return, for the layer tests.

Run in the environment of tests/references/reference_diffcsp.py, with the same
arguments: the clone of DiffCSP and the directory of its weights.

    uv run --isolated --no-project --python 3.8 ... \\
        python tests/references/reference_layers_diffcsp.py DiffCSP WEIGHTS

The decoder is run once on small structures of tests/data/orb_reference.npz
(STRUCTURES), and every layer of CSPNet in LAYERS is recorded by
tests/references/_reference_layers.py, with the complete graph CSPNet builds
(gen_edges). Read by tests/models/layers/test_cspnet.py:

* "~/.cache/torchms/diffcsp/diffcsp-mp20", given the elements, written to
  tests/data/layers/diffcsp_cspnet.npz;
* "~/.cache/torchms/diffcsp/diffcsp-gen-mp20", which also generates them from a
  vector of 100 values per atom, written to
  tests/data/layers/diffcsp_cspnet_generation.npz.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
from _reference_layers import LayerRecorder
from reference_diffcsp import DATA, load_model

# crystals of 1, 2, 4 and 12 atoms
STRUCTURES = [12, 34, 19, 30]
TIMES = [10, 300, 600, 990]

LAYERS = [
    "",
    "node_embedding",
    "atom_latent_emb",
    "dis_emb",
    "csp_layer_0",
    "csp_layer_0.edge_mlp",
    "csp_layer_0.node_mlp",
    "csp_layer_5",
    # the update of the last layer is small next to its input: its two
    # networks are recorded before the sum
    "csp_layer_5.edge_mlp",
    "csp_layer_5.node_mlp",
    "final_layer_norm",
    "coord_out",
    "lattice_out",
]
GENERATION_LAYERS = LAYERS + ["type_out"]


def crystals():
    """The structures as a batch of DiffCSP, and their cells."""
    from torch_geometric.data import Batch, Data

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])
    batch = Batch.from_data_list(
        [
            Data(
                frac_coords=torch.from_numpy(data["x"][ptr[i] : ptr[i + 1]]),
                atom_types=torch.from_numpy(data["z"][ptr[i] : ptr[i + 1]]),
                num_atoms=int(data["num_atoms"][i]),
                num_nodes=int(data["num_atoms"][i]),
            )
            for i in STRUCTURES
        ]
    )

    return batch, torch.from_numpy(data["lattice"][STRUCTURES])


def main():
    diffcsp, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()
    os.environ["PROJECT_ROOT"] = str(diffcsp)
    sys.path.insert(0, str(diffcsp))

    torch.manual_seed(0)
    model = load_model(weights)
    batch, lattices = crystals()
    decoder = model.decoder

    generation = bool(decoder.pred_type)
    if generation:
        # a noisy vector of the elements, as the sampling gives the network
        num_elements = decoder.node_embedding.in_features
        one_hot = torch.nn.functional.one_hot(batch.atom_types - 1, num_elements)
        elements = 0.8 * one_hot.float() + 0.3 * torch.randn(one_hot.shape)
        layers, output = GENERATION_LAYERS, "diffcsp_cspnet_generation.npz"
    else:
        elements = batch.atom_types
        layers, output = LAYERS, "diffcsp_cspnet.npz"

    recorder = LayerRecorder(decoder, layers)
    recorder.wrap(decoder, "gen_edges", "graph")
    with torch.no_grad():
        decoder(
            model.time_embedding(torch.tensor(TIMES)),
            elements,
            batch.frac_coords,
            lattices,
            batch.num_atoms,
            batch.batch,
        )
    recorder.remove()

    (DATA / "layers").mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / "layers" / output,
        **recorder.arrays(),
        structure_index=np.array(STRUCTURES),
        times=np.array(TIMES),
    )

    size = sum(array.nbytes for array in recorder.records.values())
    print(output, len(recorder.records), "arrays,", size // 1024, "KiB")
    for key, array in sorted(recorder.records.items()):
        print(f"  {key} {array.shape} {array.dtype}")
    print("aliases", recorder.aliases)


if __name__ == "__main__":
    main()
