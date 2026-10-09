"""Download of the pretrained weights of the models, checked against a SHA-256.

The weights of a model are listed in its configuration, the file config.yaml of
its directory, see torchms.estimator.config.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeVar

from torchms.estimator.config import config_files, load_config

Entry = TypeVar("Entry")


@dataclass(frozen=True)
class Checkpoint:
    """Location and SHA-256 of the weights of a pretrained model.

    Args:
        url: Where to download the weights.
        sha256: Expected SHA-256 of the file.
        filename: Name of the file in the cache, the last part of the URL by
            default.
    """

    url: str
    sha256: str
    filename: str | None = None


class InvalidCheckpoint(Exception):
    """Raised when weights do not have the expected content."""


def named(checkpoints: Mapping[str, Entry], name: str) -> Entry:
    """Weights of a model from their name.

    Args:
        checkpoints: Weights of each model of a kind, by name.
        name: Name of a model.

    Raises:
        KeyError: If the model is unknown.
    """
    if name not in checkpoints:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(checkpoints)}')

    return checkpoints[name]


def cache_directory() -> Path:
    """Directory of the downloaded weights.

    "$TORCHMS_CACHE" if set, else "$XDG_CACHE_HOME/torchms" or
    "~/.cache/torchms".
    """
    cache = os.environ.get("TORCHMS_CACHE")
    if cache is not None:
        return Path(cache)

    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torchms"


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
    filename = checkpoint.filename or checkpoint.url.rsplit("/", maxsplit=1)[-1]
    return directory / model / filename


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


def read_checkpoints(
    file_name: str | os.PathLike,
) -> dict[str, tuple[Checkpoint, ...]]:
    """Read the pretrained weights of a model from its configuration.

    Args:
        file_name: File config.yaml of a model, see torchms.estimator.config.

    Returns:
        Files of each of the weights of the model, by name.
    """
    checkpoints = load_config(file_name)["checkpoints"]

    return {
        name: tuple(Checkpoint(**file) for file in files)
        for name, files in checkpoints.items()
    }


# files of the weights of each model, by model then by name
CHECKPOINTS = {
    model: read_checkpoints(file_name) for model, file_name in config_files().items()
}


def _files(model: str, name: str) -> tuple[tuple[Checkpoint, ...], str]:
    """Files of weights of a model, and their directory in the cache."""
    files = named(named(CHECKPOINTS, model), name)

    # several files have the same names for other weights of the model
    return files, model if len(files) == 1 else f"{model}/{name}"


def checkpoint_paths(
    model: str, name: str, cache: str | os.PathLike | None = None
) -> tuple[Path, ...]:
    """Paths of the files of weights in the cache, downloaded or not.

    Args:
        model: Model of the weights, a key of CHECKPOINTS: "orb", "mace"...
        name: Name of the weights of this model.
        cache: Directory of the weights, see cache_directory by default.

    Raises:
        KeyError: If the model or its weights are unknown.

    Examples:
        >>> (path,) = checkpoint_paths("chgnet", "chgnet-0.3.0", cache="cache")
        >>> path.as_posix()
        'cache/chgnet/chgnet_0.3.0_e29f68s314m37.pth.tar'
        >>> paths = checkpoint_paths("m3gnet", "m3gnet-matpes-pbe-2025.2")
        >>> [path.name for path in paths]
        ['state.pt', 'model.pt']
    """
    files, directory = _files(model, name)

    return tuple(local_path(file, directory, cache) for file in files)


def download(
    model: str, name: str, cache: str | os.PathLike | None = None
) -> tuple[Path, ...]:
    """Get the files of weights, downloaded on the first call.

    Args:
        model: Model of the weights, a key of CHECKPOINTS: "orb", "mace"...
        name: Name of the weights of this model.
        cache: Directory of the weights, see cache_directory by default.

    Returns:
        Path of each file, in the order of the configuration of the model.

    Raises:
        KeyError: If the model or its weights are unknown.
        InvalidCheckpoint: If a file does not have the expected SHA-256.
    """
    files, directory = _files(model, name)

    return tuple(fetch(file, directory, cache) for file in files)
