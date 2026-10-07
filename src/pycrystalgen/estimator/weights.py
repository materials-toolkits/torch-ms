"""Download of the pretrained weights of the models, checked against a SHA-256."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    """Location and SHA-256 of the weights of a pretrained model."""

    url: str
    sha256: str


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


def local_path(
    checkpoint: Checkpoint, model: str, cache: str | os.PathLike | None = None
) -> Path:
    """Path of weights in the cache, downloaded or not.

    Args:
        checkpoint: Weights to locate.
        model: Name of the subdirectory of the cache, one per kind of model.
        cache: Directory of the weights, see cache_directory by default.
    """
    directory = cache_directory() if cache is None else Path(cache)
    return directory / model / checkpoint.url.rsplit("/", maxsplit=1)[-1]


def fetch(
    checkpoint: Checkpoint, model: str, cache: str | os.PathLike | None = None
) -> Path:
    """Get weights, downloaded on the first call.

    Args:
        checkpoint: Weights to get.
        model: Name of the subdirectory of the cache, one per kind of model.
        cache: Directory of the weights, see cache_directory by default.

    Returns:
        Path of the weights.

    Raises:
        InvalidCheckpoint: If the weights do not have the expected SHA-256.
    """
    path = local_path(checkpoint, model, cache)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(checkpoint.url, partial)
        os.replace(partial, path)

    found = sha256(path)
    if found != checkpoint.sha256:
        raise InvalidCheckpoint(
            f"{path} has SHA-256 {found}, expected {checkpoint.sha256}"
        )

    return path
