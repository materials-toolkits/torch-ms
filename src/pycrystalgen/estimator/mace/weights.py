"""Location of the converted weights of MACE.

The weights published for MACE are a pickled model which only mace can read.
scripts/reference_mace.py, run once outside of the project, converts them to a
file of tensors in the cache of pycrystalgen.
"""

from __future__ import annotations

import os
from pathlib import Path

from pycrystalgen.estimator.weights import cache_directory

# name of the model → file of its converted weights in the cache
CONVERTED = {
    "mace-mp-0-small": "mace-mp-0-small.pt",
}


def checkpoint_path(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Path of the converted weights of a model in the cache, converted or not.

    Raises:
        KeyError: If the model is unknown.
    """
    if name not in CONVERTED:
        raise KeyError(f'unknown model "{name}", must be one of {sorted(CONVERTED)}')

    directory = cache_directory() if cache is None else Path(cache)
    return directory / "mace" / CONVERTED[name]
