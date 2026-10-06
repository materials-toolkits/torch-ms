"""Download of the pretrained weights of the energy models."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_ORB_URL = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/"


@dataclass(frozen=True)
class Checkpoint:
    """Location and SHA-256 of the weights of a pretrained model."""

    url: str
    sha256: str


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


class InvalidCheckpoint(Exception):
    """Raised when weights do not have the expected content."""


def cache_directory() -> Path:
    """Directory of the downloaded weights.

    "$PYCRYSTALGEN_CACHE" if set, else "$XDG_CACHE_HOME/pycrystalgen" or
    "~/.cache/pycrystalgen".
    """
    cache = os.environ.get("PYCRYSTALGEN_CACHE")
    if cache is not None:
        return Path(cache)

    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pycrystalgen"
    )


def sha256(file_name: str | os.PathLike) -> str:
    """SHA-256 of a file, as an hexadecimal string."""
    digest = hashlib.sha256()

    with open(file_name, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def checkpoint_path(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Path of the weights of a model in the cache, downloaded or not.

    Raises:
        KeyError: If the model is unknown.
    """
    if name not in CHECKPOINTS:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(CHECKPOINTS)}')

    directory = cache_directory() if cache is None else Path(cache)
    return directory / "orb" / CHECKPOINTS[name].url.rsplit("/", maxsplit=1)[-1]


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
    path = checkpoint_path(name, cache)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(CHECKPOINTS[name].url, partial)
        os.replace(partial, path)

    found = sha256(path)
    if found != CHECKPOINTS[name].sha256:
        raise InvalidCheckpoint(
            f"{path} has SHA-256 {found}, expected {CHECKPOINTS[name].sha256}"
        )

    return path
