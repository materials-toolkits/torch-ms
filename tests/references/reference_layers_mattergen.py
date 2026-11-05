"""What the layers of GemNet-T in MatterGen receive and return, for the layer tests.

Run in the environment of tests/references/reference_mattergen.py, with the same
arguments: the clone of MatterGen and the directory of its weights.

    uv run --isolated --no-project --python 3.10 ... \\
        python tests/references/reference_layers_mattergen.py mattergen WEIGHTS

The denoiser is run once on two small structures of tests/data/orb_reference.npz
(STRUCTURES), and every layer of GemNet-T in LAYERS is recorded by
tests/references/_reference_layers.py: its inputs, its outputs, and the graph MatterGen
builds (generate_interaction_graph). Read by tests/models/layers/test_gemnet.py:

* "~/.cache/torchms/mattergen/mattergen-base", the base model, written to
  tests/data/layers/mattergen_gemnet.npz;
* "~/.cache/torchms/mattergen/mattergen-chemical-system", conditioned on the
  chemical system of each structure: its adapters, written to
  tests/data/layers/mattergen_gemnet_adapters.npz.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
from _reference_layers import LayerRecorder
from reference_mattergen import (
    DATA,
    chemgraphs,
    conditions,
    load_model,
    own_chemical_systems,
    saved_values,
    with_conditions,
)

# a crystal of one atom in an oblique cell, and one of four in a triclinic one:
# about 240 edges, which keeps the file small
STRUCTURES = [12, 19]
TIMES = [0.3, 0.7]

# one instance of each layer of GemNet-T, by name in MatterGen
LAYERS = [
    "",
    "radial_basis",
    "cbf_basis3",
    "mlp_rbf3",
    "mlp_cbf3",
    "mlp_rbf_h",
    "mlp_rbf_out",
    "atom_emb",
    "atom_latent_emb",
    "edge_emb",
    "angle_edge_emb",
    "out_blocks.0",
    "mlp_rbf_lattice",
    "lattice_out_blocks.0",
    "int_blocks.0",
    "int_blocks.0.trip_interaction",
    "int_blocks.0.atom_update",
]
# the adapters of a property, before the first block, and the bases of the
# triplets, whose cosines the whole network is given
ADAPTERS = ["", "cbf_basis3", "cond_adapt_layers.{}.0", "cond_mixin_layers.{}.0"]


def main():  # pylint: disable=too-many-locals
    mattergen, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()
    model = load_model(mattergen, weights)
    gemnet = model.diffusion_module.model.gemnet

    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])
    atoms = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in STRUCTURES])
    lattice, x, z = data["lattice"][STRUCTURES], data["x"][atoms], data["z"][atoms]
    num_atoms = data["num_atoms"][STRUCTURES]
    batch = chemgraphs(lattice, x, z, num_atoms)

    names = conditions(model)
    result = {}
    if names:
        assert names == ["chemical_system"], names
        systems = own_chemical_systems(z, num_atoms)
        conditioned = np.ones(len(STRUCTURES), dtype=bool)
        batch = with_conditions(batch, {"chemical_system": systems}, conditioned)
        layers = [layer.format(names[0]) for layer in ADAPTERS]
        output = "mattergen_gemnet_adapters.npz"
        result["chemical_systems"] = saved_values("chemical_system", systems)
    else:
        layers = LAYERS
        output = "mattergen_gemnet.npz"

    recorder = LayerRecorder(gemnet, layers)
    recorder.wrap(gemnet, "generate_interaction_graph", "graph")
    with torch.no_grad():
        model.diffusion_module.model(batch, torch.tensor(TIMES))
    recorder.remove()

    result.update(recorder.arrays())
    result.update(structure_index=np.array(STRUCTURES), times=np.array(TIMES))
    (DATA / "layers").mkdir(exist_ok=True)
    np.savez_compressed(DATA / "layers" / output, **result)

    size = sum(array.nbytes for array in recorder.records.values())
    print(output, len(recorder.records), "arrays,", size // 1024, "KiB")
    for key, array in sorted(recorder.records.items()):
        print(f"  {key} {array.shape} {array.dtype}")
    print("aliases", recorder.aliases)


if __name__ == "__main__":
    main()
