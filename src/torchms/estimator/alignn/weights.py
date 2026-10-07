"""Download of the pretrained weights of ALIGNN."""

from __future__ import annotations

import os
from pathlib import Path

from torchms.estimator.weights import Checkpoint, fetch, local_path

# archives of the ALIGNN2 project of Figshare, with the parameters of a model
# (best_model.pt) and its configuration (config.json)
CHECKPOINTS = {
    "alignn-formation-energy-radius": Checkpoint(
        "https://ndownloader.figshare.com/files/67163411",
        "148c9069f16c43d1ab3fc0c4df756549a31fcd662131252ad54b0cf0e2f2a114",
        "formation_energy_peratom_radius.zip",
    ),
}


def _checkpoint(name: str) -> Checkpoint:
    if name not in CHECKPOINTS:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(CHECKPOINTS)}')

    return CHECKPOINTS[name]


def checkpoint_path(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Path of the archive of a model in the cache, downloaded or not.

    Raises:
        KeyError: If the model is unknown.
    """
    return local_path(_checkpoint(name), "alignn", cache)


def download(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Get the archive of a model, downloaded on the first call.

    Raises:
        KeyError: If the model is unknown.
        InvalidCheckpoint: If the archive does not have the expected SHA-256.
    """
    return fetch(_checkpoint(name), "alignn", cache)
