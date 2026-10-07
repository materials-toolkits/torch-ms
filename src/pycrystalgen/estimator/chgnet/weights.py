"""Download of the pretrained weights of CHGNet."""

from __future__ import annotations

import os
from pathlib import Path

from pycrystalgen.estimator.weights import Checkpoint, fetch, local_path

_CHGNET_URL = (
    "https://raw.githubusercontent.com/CederGroupHub/chgnet/v0.4.2/chgnet/pretrained/"
)

CHECKPOINTS = {
    "chgnet-0.3.0": Checkpoint(
        _CHGNET_URL + "0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
        "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
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
    return local_path(_checkpoint(name), "chgnet", cache)


def download(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Get the weights of a model, downloaded on the first call.

    Raises:
        KeyError: If the model is unknown.
        InvalidCheckpoint: If the weights do not have the expected SHA-256.
    """
    return fetch(_checkpoint(name), "chgnet", cache)
