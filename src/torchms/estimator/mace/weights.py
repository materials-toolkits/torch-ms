"""Download of the pretrained weights of MACE."""

from __future__ import annotations

import os
from pathlib import Path

from torchms.estimator.weights import Checkpoint, fetch, local_path

# pickled models of the releases of mace, read by reader.py without mace
CHECKPOINTS = {
    "mace-mp-0-small": Checkpoint(
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
        "2023-12-10-mace-128-L0_energy_epoch-249.model",
        "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736",
    ),
}


def _checkpoint(name: str) -> Checkpoint:
    if name not in CHECKPOINTS:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(CHECKPOINTS)}')

    return CHECKPOINTS[name]


def checkpoint_path(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Path of the weights of a model in the cache, downloaded or not.

    Raises:
        KeyError: If the model is unknown.
    """
    return local_path(_checkpoint(name), "mace", cache)


def download(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Get the weights of a model, downloaded on the first call.

    Raises:
        KeyError: If the model is unknown.
        InvalidCheckpoint: If the weights do not have the expected SHA-256.
    """
    return fetch(_checkpoint(name), "mace", cache)
