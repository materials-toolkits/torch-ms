"""Configuration of the pretrained models: one YAML file per model.

Each pretrained model of torchms.models has a file
config.yaml next to its module model.py, with:

* default_weights: name of the weights loaded by default;
* checkpoints: files of each of its pretrained weights, see
  torchms.models.estimator.weights;
* unused_parameters: parameters of these weights the model does not have;
* graph: radii and numbers of neighbours of its graph;
* model: hyperparameters of the network;
* upstream: what is checked in the weights of the project it comes from.

A model reads no value from elsewhere: model.py has the names of the values,
config.yaml the values.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"


@functools.lru_cache(maxsize=None)
def _read(file_name: Path) -> dict[str, Any]:
    with open(file_name, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_config(module_file: str | os.PathLike) -> dict[str, Any]:
    """Configuration of a model, from the file config.yaml next to its module.

    Args:
        module_file: File of the module of the model, its __file__, or the
            file of the configuration itself.

    Returns:
        Content of the file. It is read once: do not modify it.

    Examples:
        >>> from torchms.models.mace import model
        >>> config = load_config(model.__file__)
        >>> config["default_weights"], config["graph"]["radius"]
        ('mace-mp-0-small', 6.0)
    """
    path = Path(module_file)
    if path.name != CONFIG_NAME:
        path = path.parent / CONFIG_NAME

    return _read(path.resolve())


def config_files() -> dict[str, Path]:
    """File of the configuration of each model, by name of its directory.

    The models are the directories of torchms.models with a configuration.

    Examples:
        >>> sorted(config_files())
        ['alignn', 'cdvae', 'chgnet', 'diffcsp', 'm3gnet', 'mace', 'orb']
    """
    directory = Path(__file__).parent.parent

    return {
        path.parent.name: path for path in sorted(directory.glob(f"*/{CONFIG_NAME}"))
    }
