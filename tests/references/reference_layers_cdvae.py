"""What the layers of DimeNet++ and GemNet-T in cdvae receive and return.

For the layer tests.

Run in the environment of tests/references/reference_cdvae.py, with the same
arguments: the clone of cdvae and the directory of its weights.

    uv run --isolated --no-project --python 3.8 ... \\
        python tests/references/reference_layers_cdvae.py cdvae WEIGHTS

Small structures of tests/data/orb_reference.npz (STRUCTURES) are encoded on
their CrystalNN graphs, then decoded from the mean of their latent vector with
moved atoms. Every layer in ENCODER_LAYERS and DECODER_LAYERS is recorded by
tests/references/_reference_layers.py, with the graphs: the pairs of bonds of
the encoder (triplets) and the graph GemNet-T builds
(generate_interaction_graph). Written to tests/data/layers/cdvae_dimenet.npz
and tests/data/layers/cdvae_gemnet.npz, read by
tests/models/layers/test_dimenet.py and test_gemnet.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
from _reference_layers import LayerRecorder
from reference_cdvae import DATA, load_batch, load_model

# crystals of 1, 2, 4 and 12 atoms
STRUCTURES = [12, 34, 19, 30]

ENCODER_LAYERS = [
    "",
    "rbf",
    "sbf",
    "emb",
    "interaction_blocks.0",
    "interaction_blocks.3",
    "output_blocks.0",
    "output_blocks.4",
]
DECODER_LAYERS = [
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
    "out_blocks.0",
    "int_blocks.0",
    "int_blocks.0.trip_interaction",
    "int_blocks.0.atom_update",
]


def save(name: str, recorder: LayerRecorder, **extra) -> None:
    (DATA / "layers").mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / "layers" / name,
        **recorder.arrays(),
        structure_index=np.array(STRUCTURES),
        **extra,
    )
    size = sum(array.nbytes for array in recorder.records.values())
    print(name, len(recorder.records), "arrays,", size // 1024, "KiB")
    for key, array in sorted(recorder.records.items()):
        print(f"  {key} {array.shape} {array.dtype}")


def main():
    cdvae, weights = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).expanduser()
    os.environ["PROJECT_ROOT"] = str(cdvae)
    sys.path.insert(0, str(cdvae))

    from cdvae.common.data_utils import lattice_params_to_matrix_torch

    torch.manual_seed(0)
    model = load_model(cdvae, weights)
    index, batch = load_batch(model.hparams.max_atoms, STRUCTURES)
    assert index.tolist() == STRUCTURES, index
    # the cells cdvae builds from their lengths and angles, as rows
    lattice = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    crystals = {
        "frac_coords": batch.frac_coords.numpy(),
        "atom_types": batch.atom_types.numpy(),
        "num_atoms": batch.num_atoms.numpy(),
        "lattice": lattice.numpy(),
        # the CrystalNN graph: an edge j, i with the image of j
        "edge_index": batch.edge_index.numpy(),
        "to_jimages": batch.to_jimages.numpy(),
    }

    with torch.no_grad():
        encoder = model.encoder
        recorder = LayerRecorder(encoder, ENCODER_LAYERS)
        recorder.wrap(encoder, "triplets", "pairs")
        mu = model.fc_mu(encoder(batch))
        recorder.remove()
        save("cdvae_dimenet.npz", recorder, **crystals)

        gemnet = model.decoder.gemnet
        noisy = (batch.frac_coords + 0.05 * torch.randn_like(batch.frac_coords)) % 1.0
        recorder = LayerRecorder(gemnet, DECODER_LAYERS)
        recorder.wrap(gemnet, "generate_interaction_graph", "graph")
        model.decoder(
            mu, noisy, batch.atom_types, batch.num_atoms, batch.lengths, batch.angles
        )
        recorder.remove()
        save("cdvae_gemnet.npz", recorder, lattice=lattice.numpy())


if __name__ == "__main__":
    main()
