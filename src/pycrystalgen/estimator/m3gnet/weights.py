"""Download of the pretrained weights of M3GNet."""

from __future__ import annotations

import os
from pathlib import Path

from pycrystalgen.estimator.weights import Checkpoint, fetch, local_path

_M3GNET_URL = (
    "https://huggingface.co/materialyze/M3GNet-PES-MatPES-PBE-2025.2/resolve/"
    "a972e0dd982eaf2a4cf1e5c8de2c9e78c0013f72/"
)

# parameters of the model (state.pt) and its arguments (model.pt)
CHECKPOINTS = {
    "m3gnet-matpes-pbe-2025.2": (
        Checkpoint(
            _M3GNET_URL + "state.pt",
            "ba5b96249e366456cb3e518fcc93f58c49353f564c79648f309b916926b10fa0",
        ),
        Checkpoint(
            _M3GNET_URL + "model.pt",
            "8204c34dc4df513e03d080017c30d7628fa3e4fcc13b9fe47076da0bcd3129da",
        ),
    ),
}


def _checkpoints(name: str) -> tuple[Checkpoint, Checkpoint]:
    if name not in CHECKPOINTS:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(CHECKPOINTS)}')

    return CHECKPOINTS[name]


def checkpoint_path(
    name: str, cache: str | os.PathLike | None = None
) -> tuple[Path, Path]:
    """Paths of the parameters and of the arguments of a model in the cache.

    Raises:
        KeyError: If the model is unknown.
    """
    state, args = _checkpoints(name)
    return (
        local_path(state, f"m3gnet/{name}", cache),
        local_path(args, f"m3gnet/{name}", cache),
    )


def download(name: str, cache: str | os.PathLike | None = None) -> tuple[Path, Path]:
    """Get the parameters and the arguments of a model, downloaded on the first call.

    Raises:
        KeyError: If the model is unknown.
        InvalidCheckpoint: If a file does not have the expected SHA-256.
    """
    state, args = _checkpoints(name)
    return fetch(state, f"m3gnet/{name}", cache), fetch(args, f"m3gnet/{name}", cache)
