"""Reading of the weights of MACE without mace.

mace publishes its models as pickled Python objects: the whole model, not its
tensors. Unpickling normally imports the classes of mace and e3nn. The file is
read with placeholders in place of these classes, see
torchms.models.estimator.reader. The tensors of the model are then collected
from the placeholders, with the names they have in the state dict of mace.
"""

from __future__ import annotations

import os
import pickle

import torch

from torchms.models.estimator.reader import Placeholder, read_pickle


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
