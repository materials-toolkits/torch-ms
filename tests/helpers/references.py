"""What an original code computed, recorded by tests/references/reference_*.py."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

from helpers.paths import DATA
from torchms.data.batching import Structures


@dataclass(frozen=True)
class RecordedCall:
    """Inputs and outputs of one call of a layer of an original code.

    Attributes:
        inputs: Tensors of each argument, by name; "name:i" for the i-th tensor
            of a tuple, "name:key" for a dict.
        outputs: Tensors returned: "" for a single tensor, "0", "1"... for a
            tuple, the field names for a dataclass.
    """

    inputs: Mapping[str, torch.Tensor]
    outputs: Mapping[str, torch.Tensor]

    def input(self, name: str) -> torch.Tensor:
        """A tensor given to the layer.

        Raises:
            KeyError: If it was not recorded, with the names which were.
        """
        if name not in self.inputs:
            raise KeyError(f"no input {name!r}, recorded: {sorted(self.inputs)}")

        return self.inputs[name]

    def output(self, name: str = "") -> torch.Tensor:
        """A tensor returned by the layer, the only one by default.

        Raises:
            KeyError: If it was not recorded, with the names which were.
        """
        if name not in self.outputs:
            raise KeyError(f"no output {name!r}, recorded: {sorted(self.outputs)}")

        return self.outputs[name]


@dataclass(frozen=True)
class Reference:
    """Arrays recorded from an original code, as tensors, aliases resolved.

    tests/references/_reference_layers.py keeps a tensor given to several layers once,
    and the other names in "aliases": here every name has its tensor.

    Examples:
        >>> reference = Reference.from_arrays(
        ...     {
        ...         "dense:in:x": np.ones((2, 3)),
        ...         "dense:out": np.zeros(2),
        ...         "aliases": np.array('{"next:in:x": "dense:out"}'),
        ...     }
        ... )
        >>> reference.call("next").input("x")
        tensor([0., 0.], dtype=torch.float64)
        >>> reference.call("dense").output().shape
        torch.Size([2])
    """

    arrays: Mapping[str, torch.Tensor]

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> Reference:
        aliases = json.loads(str(arrays["aliases"])) if "aliases" in arrays else {}
        tensors = {
            key: torch.from_numpy(np.asarray(value))
            for key, value in arrays.items()
            if key != "aliases"
        }
        tensors.update({alias: tensors[key] for alias, key in aliases.items()})

        return cls(tensors)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.arrays[key]

    def call(self, module: str) -> RecordedCall:
        """What a module received and returned, by its name in the original.

        Raises:
            KeyError: If the module was not recorded.
        """
        inputs = {
            key[len(module) + 4 :]: value
            for key, value in self.arrays.items()
            if key.startswith(f"{module}:in:")
        }
        outputs = {
            key[len(module) + 5 :]: value
            for key, value in self.arrays.items()
            if key == f"{module}:out" or key.startswith(f"{module}:out:")
        }
        if not inputs and not outputs:
            modules = sorted({key.split(":")[0] for key in self.arrays})
            raise KeyError(f"{module!r} was not recorded, recorded: {modules}")

        return RecordedCall(inputs, outputs)


def load_reference(path: str | Path) -> Reference:
    """Arrays of a reference file, relative to tests/data or absolute."""
    with np.load(DATA / path) as arrays:
        return Reference.from_arrays(dict(arrays))


def reference_structures(indices: Sequence[int] | None = None) -> Structures:
    """Structures of tests/data/orb_reference.npz, which every reference reads.

    Args:
        indices: Which structures, in this order; all 40 by default.

    Examples:
        >>> reference_structures([12, 19]).num_atoms
        tensor([1, 4])
    """
    with np.load(DATA / "orb_reference.npz") as data:
        lattice, x, z, num_atoms = (
            torch.from_numpy(data[key]) for key in ("lattice", "x", "z", "num_atoms")
        )
    structures = Structures(lattice=lattice, x=x, z=z, num_atoms=num_atoms)
    if indices is None:
        return structures

    return structures.select(torch.tensor(list(indices)))


@contextmanager
def submodule_inputs(
    module: torch.nn.Module, names: Sequence[str]
) -> Iterator[dict[str, torch.Tensor]]:
    """The first tensor given to named submodules of a torchms module, while it runs.

    To compare what a layer gives its own networks with what the original
    gave them, when the layer adds a small update to a large input: its output
    alone would hide an error of the update.

    Examples:
        >>> net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
        >>> with submodule_inputs(net, ["1"]) as inputs:
        ...     _ = net(torch.ones(1, 2))
        >>> inputs["1"].shape
        torch.Size([1, 3])
    """
    modules = dict(module.named_modules())
    inputs: dict[str, torch.Tensor] = {}

    def keep(name: str):
        def hook(_: torch.nn.Module, args: tuple) -> None:
            inputs.setdefault(name, args[0].detach())

        return hook

    handles = [modules[name].register_forward_pre_hook(keep(name)) for name in names]
    try:
        yield inputs
    finally:
        for handle in handles:
            handle.remove()
