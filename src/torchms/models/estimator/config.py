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

A model reads no value from elsewhere: config.yaml has the values, and
config.py, next to it, a dataclass Config with their names and their types,
built from the file by YamlConfig.load. The model and its layers are given
this object: no module keeps a value of the file.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

CONFIG_NAME = "config.yaml"

# section of a configuration which is not a field of its Config: the files of
# the weights, read by torchms.models.estimator.weights
CHECKPOINTS_SECTION = "checkpoints"

Loaded = TypeVar("Loaded", bound="YamlConfig")


class InvalidConfig(Exception):
    """Raised when a configuration does not have the fields of its dataclass."""


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
        >>> sorted(config_files())  # doctest: +NORMALIZE_WHITESPACE
        ['adit', 'alignn', 'cdvae', 'chgnet', 'crystalflow', 'diffcsp', 'diffcsp_pp', 'flowmm',
         'm3gnet', 'mace', 'mattergen', 'mattersim', 'orb']
    """
    directory = Path(__file__).parent.parent

    return {
        path.parent.name: path for path in sorted(directory.glob(f"*/{CONFIG_NAME}"))
    }


def _named(path: str) -> str:
    return f'"{path}"' if path else "the configuration"


def _build(hint: Any, value: Any, path: str) -> Any:
    """Value of a field of a configuration, from what the file has for it.

    Args:
        hint: Type of the field.
        value: Value read from the file.
        path: Name of the field in the file, with dots, for the errors.
    """
    # pylint: disable=too-many-return-statements
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise InvalidConfig(f"{_named(path)} must be a section")

        hints = get_type_hints(hint)
        fields = {field.name: field for field in dataclasses.fields(hint)}
        prefix = f"{path}." if path else ""

        unknown = sorted(set(value) - set(fields))
        if unknown:
            raise InvalidConfig(f"unknown fields {unknown} in {_named(path)}")

        missing = sorted(
            name
            for name, field in fields.items()
            if name not in value
            and field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )
        if missing:
            raise InvalidConfig(f"missing fields {missing} in {_named(path)}")

        return hint(
            **{
                name: _build(hints[name], given, prefix + name)
                for name, given in value.items()
            }
        )

    origin = get_origin(hint)

    if origin in (Union, types.UnionType):
        none = type(None)
        if value is None and none in get_args(hint):
            return None
        (inner,) = [arg for arg in get_args(hint) if arg is not none]

        return _build(inner, value, path)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise InvalidConfig(f"{_named(path)} must be a list")

        inner = get_args(hint)
        if len(inner) == 2 and inner[1] is Ellipsis:
            inner = (inner[0],) * len(value)
        if len(inner) != len(value):
            raise InvalidConfig(f"{_named(path)} must have {len(inner)} values")

        return tuple(
            _build(arg, item, f"{path}[{i}]")
            for i, (arg, item) in enumerate(zip(inner, value))
        )

    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    # the exact type: True is not an integer of a configuration
    # pylint: disable-next=unidiomatic-typecheck
    if hint in (bool, int, float, str) and type(value) is not hint:
        raise InvalidConfig(
            f"{_named(path)} must be {hint.__name__}, found {type(value).__name__}"
        )

    # a mapping by nature, or a type which is not checked
    return value


@functools.lru_cache(maxsize=None)
def _load(cls: type[Loaded]) -> Loaded:
    module_file = sys.modules[cls.__module__].__file__
    assert module_file is not None

    return cls.from_mapping(load_config(module_file))


@dataclasses.dataclass(frozen=True, kw_only=True)
class YamlConfig:
    """Base of the configuration of a pretrained model, read from its config.yaml.

    A model defines a frozen dataclass Config in the module config.py of its
    directory, with one field per section of the file and one dataclass per
    section. The object is given to the model and to its layers, which keep
    no value of the file in a global variable.

    Args:
        default_weights: Name of the weights loaded by from_pretrained.
        unused_parameters: Prefixes of the parameters of the pretrained
            weights which the model does not have.

    Examples:
        >>> from torchms.models import m3gnet
        >>> config = m3gnet.Config.load()
        >>> config.default_weights, config.graph.radius, config.model.num_blocks
        ('m3gnet-matpes-pbe-2025.2', 5.0, 3)

        Another model than the one of the file, to train:

        >>> model = dataclasses.replace(config.model, num_blocks=2)
        >>> dataclasses.replace(config, model=model).model.num_blocks
        2
    """

    default_weights: str
    unused_parameters: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls: type[Loaded], values: Mapping[str, Any]) -> Loaded:
        """Configuration from the content of a file.

        The fields of the dataclass and of the dataclasses of its sections are
        the keys of the file: a list becomes a tuple, an integer given for a
        float becomes a float, a mapping stays a dictionary. The section of
        the files of the weights is left out.

        Args:
            values: Content of a config.yaml.

        Raises:
            InvalidConfig: If a key is unknown, if a field without default is
                missing or if a value does not have the type of its field.

        Examples:
            >>> YamlConfig.from_mapping({"default_weights": "a", "checkpoints": {}})
            YamlConfig(default_weights='a', unused_parameters=())
            >>> YamlConfig.from_mapping({"default_weights": "a", "radius": 5.0})
            Traceback (most recent call last):
            ...
            torchms.models.estimator.config.InvalidConfig: unknown fields \
['radius'] in the configuration
        """
        fields = {
            name: value for name, value in values.items() if name != CHECKPOINTS_SECTION
        }

        return _build(cls, fields, "")

    @classmethod
    def load(cls: type[Loaded]) -> Loaded:
        """Configuration of the file config.yaml next to the module of the class.

        The file is read once.
        """
        return _load(cls)
