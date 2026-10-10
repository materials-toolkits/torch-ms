"""Reading of the weights of MACE without mace.

mace publishes its models as pickled Python objects: the whole model, not its
tensors. Unpickling normally imports the classes of mace and e3nn. Here every
class which is not a tensor or a container of the standard library is replaced
by a placeholder which only keeps what the pickle gives it. The tensors of the
model are then collected from the placeholders, with the names they have in
the state dict of mace.

Nothing of the file is executed: the unpickler only resolves the names of
ALLOWED, a pickle cannot call anything else.
"""

from __future__ import annotations

import codecs
import collections
import importlib
import os
import pickle
import types
from typing import Any

import numpy as np
import torch

# functions of torch which rebuild a tensor from its storage
_TORCH_UTILS = importlib.import_module("torch._utils")

# the only objects a pickle can create or call, besides placeholders
ALLOWED: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): getattr(_TORCH_UTILS, "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"): getattr(_TORCH_UTILS, "_rebuild_parameter"),
    ("torch", "DoubleStorage"): torch.DoubleStorage,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch", "IntStorage"): torch.IntStorage,
    ("torch", "BoolStorage"): torch.BoolStorage,
    ("numpy", "dtype"): np.dtype,
    # bytes of a scalar of numpy, written as text by old pickles
    ("_codecs", "encode"): codecs.encode,
}


# containers of the standard library, under their name in old and new pickles
_BUILTINS: dict[str, Any] = {"object": object, "set": set, "frozenset": frozenset}


class Placeholder:
    """Stands for an object of a class which is not imported.

    Attributes:
        args: Arguments its class or its function was called with.
        state: State the pickle restored, the attributes of the object.
    """

    path = ""

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def attributes(self) -> dict[str, Any]:
        """Attributes of the object, empty if its state is not a dictionary."""
        return self.state if isinstance(self.state, dict) else {}


def _numpy_scalar(dtype: np.dtype, data: bytes) -> Any:
    """Scalar of numpy from its bytes, as numpy pickles it."""
    return np.frombuffer(data, dtype=dtype)[0].item()


def _reconstructor(cls: type, base: type, state: Any) -> Any:
    """Object created without calling its constructor, as copyreg does."""
    del base, state
    return object.__new__(cls)


class Unpickler(pickle.Unpickler):
    """Unpickler which never imports: unknown names become placeholders."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ALLOWED:
            return ALLOWED[(module, name)]
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "scalar":
                return _numpy_scalar
        if (module, name) == ("copyreg", "_reconstructor"):
            return _reconstructor
        if module in ("builtins", "__builtin__") and name in _BUILTINS:
            return _BUILTINS[name]

        return type(name, (Placeholder,), {"path": f"{module}.{name}"})


def _load(file: Any, **kwargs: Any) -> Any:
    """Unpickle an object with placeholders."""
    return Unpickler(file, **kwargs).load()


# given to torch.load in place of the module pickle
_PICKLE = types.SimpleNamespace(__name__="pickle", Unpickler=Unpickler, load=_load)


def read_pickle(file_name: str | os.PathLike) -> Any:
    """Read a file written by torch.save without importing any class.

    For the files which torch.load refuses with weights_only=True: checkpoints
    with objects of other libraries next to their tensors.

    Args:
        file_name: File written by torch.save.

    Returns:
        What the file holds, with tensors and containers as they are and a
        Placeholder in place of every other object.
    """
    return torch.load(
        file_name,
        map_location="cpu",
        pickle_module=_PICKLE,
        weights_only=False,
    )


def _collect(
    module: Placeholder,
    prefix: str,
    tensors: dict[str, torch.Tensor],
    modules: dict[str, Placeholder],
) -> None:
    """Tensors and submodules of a placeholder of a torch module, recursively."""
    attributes = module.attributes()
    modules[prefix.rstrip(".")] = module

    for name, parameter in attributes.get("_parameters", {}).items():
        if isinstance(parameter, torch.Tensor):
            tensors[prefix + name] = parameter.detach()

    skipped = attributes.get("_non_persistent_buffers_set", set())
    for name, buffer in attributes.get("_buffers", {}).items():
        if isinstance(buffer, torch.Tensor) and name not in skipped:
            tensors[prefix + name] = buffer

    for name, child in attributes.get("_modules", {}).items():
        if isinstance(child, Placeholder):
            _collect(child, f"{prefix}{name}.", tensors, modules)


def read_model(
    file_name: str | os.PathLike,
) -> tuple[dict[str, torch.Tensor], dict[str, Placeholder]]:
    """Read a pickled torch model without importing its classes.

    Args:
        file_name: File written by torch.save(model).

    Returns:
        Parameters and buffers of the model, with their names in its state
        dict, and placeholders of its modules by name ("" for the model),
        which hold their other attributes.

    Raises:
        pickle.UnpicklingError: If the file is not a pickled torch module.
    """
    model = read_pickle(file_name)
    if not isinstance(model, Placeholder) or "_modules" not in model.attributes():
        raise pickle.UnpicklingError(f"{file_name} is not a pickled torch module")

    tensors: dict[str, torch.Tensor] = {}
    modules: dict[str, Placeholder] = {}
    _collect(model, "", tensors, modules)

    return tensors, modules
