"""Download of the pretrained weights of the orb models."""

from __future__ import annotations

import os
from pathlib import Path

from pycrystalgen.estimator.weights import (
    Checkpoint,
    InvalidCheckpoint,
    cache_directory,
    fetch,
    local_path,
    sha256,
)

__all__ = [
    "CHECKPOINTS",
    "Checkpoint",
    "InvalidCheckpoint",
    "cache_directory",
    "checkpoint_path",
    "download",
    "sha256",
]

_ORB_URL = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/"

CHECKPOINTS = {
    "orb-v3-conservative-inf-omat": Checkpoint(
        _ORB_URL + "orb-v3-conservative-inf-omat-20250404.ckpt",
        "0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0",
    ),
    "orb-v3-conservative-inf-mpa": Checkpoint(
        _ORB_URL + "orb-v3-conservative-inf-mpa-20250404.ckpt",
        "bd1840d1488f3cd01d213970d9618a63f971590ad45e5e83bf5404d4b95d216a",
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
    return local_path(_checkpoint(name), "orb", cache)


def download(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Get the weights of a model, downloaded on the first call.

    Args:
        name: Name of the model, a key of CHECKPOINTS.
        cache: Directory of the weights, see cache_directory by default.

    Returns:
        Path of the weights.

    Raises:
        KeyError: If the model is unknown.
        InvalidCheckpoint: If the weights do not have the expected SHA-256.
    """
    return fetch(_checkpoint(name), "orb", cache)
