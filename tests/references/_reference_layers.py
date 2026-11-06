"""Record what submodules of an original model receive and return, for the layer tests.

Imported by the tests/references/reference_layers_<model>.py scripts, which run in the
isolated environment of the original code (torch 1.8 for cdvae, 2.2 for
MatterGen): only torch, numpy and the standard library.

    recorder = LayerRecorder(model.gemnet, ["radial_basis", "int_blocks.0"])
    model(batch)
    arrays = recorder.arrays()   # "int_blocks.0:in:h", "int_blocks.0:out:0", ...

The first call of each module is kept. A module's forward is wrapped on its
instance, which works with every version of torch, where forward hooks with
keyword arguments need torch 2. Tensors are saved in float32 (int64 for
indices), copied when recorded. A tensor already kept under another name, the
output of one layer given to the next one, is kept once: its other names are in
"aliases", read by tests/helpers/references.py.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import torch


class LayerRecorder:
    """Inputs and outputs of the first call of named submodules of a module.

    Args:
        root: Module whose submodules are recorded.
        names: Their names, as in root.named_modules(); "" is root itself,
            recorded as "network".
    """

    def __init__(self, root: torch.nn.Module, names: list):
        modules = dict(root.named_modules())
        self.records: dict = {}
        self.aliases: dict = {}
        self._kept: dict = {}
        self._restore: list = []
        for name in names:
            self.wrap(modules[name], "forward", name or "network")

    def wrap(self, owner, method: str, label: str) -> None:
        """Record the first call of a method of an object, under a label."""
        function = getattr(owner, method)
        signature = inspect.signature(function)
        called = []

        def wrapped(*args, **kwargs):
            first = not called
            if first:
                called.append(True)
                # inputs before the call, which may change them in place
                bound = signature.bind(*args, **kwargs)
                for argument, value in bound.arguments.items():
                    self.keep(f"{label}:in:{argument}", value)
            out = function(*args, **kwargs)
            if first:
                self.keep(f"{label}:out", out)
            return out

        setattr(owner, method, wrapped)
        self._restore.append((owner, method))

    def keep(self, key: str, value) -> None:
        """Keep the tensors of a value: a tensor, a tuple, a dict or a dataclass."""
        if isinstance(value, torch.Tensor):
            self.keep_tensor(key, value)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.keep(f"{key}:{i}", item)
        elif isinstance(value, dict):
            for name, item in value.items():
                self.keep(f"{key}:{name}", item)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                self.keep(f"{key}:{field.name}", getattr(value, field.name))

    def keep_tensor(self, key: str, tensor: torch.Tensor) -> None:
        # a copy: the original changes some outputs in place after (F_st += F)
        array = tensor.detach().cpu().clone()
        if array.is_floating_point():
            array = array.float()
        elif array.dtype != torch.bool:
            array = array.long()

        # the same view of the same storage, with the same values, is the
        # same tensor: an output given to the next layer
        identity = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        kept = self._kept.get(identity)
        if kept is not None and torch.equal(kept[1], array):
            self.aliases[key] = kept[0]
            return
        self._kept[identity] = (key, array)
        self.records[key] = array.numpy()

    def arrays(self) -> dict:
        """The recorded arrays, with the aliases as a JSON string under "aliases"."""
        return {**self.records, "aliases": np.array(json.dumps(self.aliases))}

    def remove(self) -> None:
        """Restore the wrapped methods."""
        for owner, method in self._restore:
            delattr(owner, method)
        self._restore.clear()
