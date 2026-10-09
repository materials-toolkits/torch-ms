"""Loading of pretrained weights, shared by the models of the estimators."""

from __future__ import annotations

import os
from typing import Mapping, TypeVar

import torch
from jaxtyping import Int
from torch import nn

from torchms.data.augmentation.transforms import Transform
from torchms.data.batching import Structures

Model = TypeVar("Model", bound="PretrainedModel")


class InvalidWeights(Exception):
    """Raised when weights do not match the parameters of the model."""


def different_arguments(
    found: Mapping[str, object], expected: Mapping[str, object]
) -> dict[str, object]:
    """Arguments of a model which are not the ones a rewrite is written for.

    Args:
        found: Arguments stored with the weights.
        expected: Arguments of the rewrite.

    Returns:
        Value found for each argument which is given with another value than
        the expected one. An argument which is not given is not checked.

    Examples:
        >>> expected = {"cutoff": 5.0, "units": 64}
        >>> different_arguments({"cutoff": 5.0}, expected)
        {}
        >>> different_arguments({"cutoff": 6.0, "other": 1}, expected)
        {'cutoff': 6.0}
    """
    return {
        key: found.get(key)
        for key, value in expected.items()
        if found.get(key, value) != value
    }


def element_table(
    atomic_numbers: Int[torch.Tensor, "n_elements"],
) -> Int[torch.Tensor, "119"]:
    """Index of each atomic number in the elements of a model, -1 if it has none.

    Examples:
        >>> element_table(torch.tensor([1, 6, 8]))[:9]
        tensor([-1,  0, -1, -1, -1, -1,  1, -1,  2])
    """
    device = atomic_numbers.device
    table = torch.full((119,), -1, dtype=torch.int64, device=device)
    table[atomic_numbers] = torch.arange(len(atomic_numbers), device=device)

    return table


def element_index(
    z: Int[torch.Tensor, "n_atoms"], table: Int[torch.Tensor, "119"]
) -> Int[torch.Tensor, "n_atoms"]:
    """Index of the element of each atom in the elements of a model.

    Args:
        z: Atomic number of each atom.
        table: Index of each atomic number, see element_table.

    Raises:
        ValueError: If an atom has an element the model does not know.

    Examples:
        >>> table = element_table(torch.tensor([1, 6, 8]))
        >>> element_index(torch.tensor([8, 1, 1]), table)
        tensor([2, 0, 0])
    """
    element = table[z]
    if (element < 0).any():
        unknown = z[element < 0].unique().tolist()
        raise ValueError(f"elements {unknown} are not supported by the model")

    return element


class PretrainedModel(nn.Module):
    """Model which loads the pretrained weights of the project it is derived from.

    A model gives the name of its default weights, reads their files in
    load_checkpoint and names the parameters of these files it does not use.

    A model does not build graphs: it reads the graph and the tuples of edges
    attached to the structures, and gives the transform which attaches them.
    The transform is applied when the structures are loaded:

    ```python
    dataset = HDF5Dataset("structures.hdf5", transform=Model.transform())
    ```

    or to a batch, `model(Model.transform()(structures))`. Structures without
    them are refused, see check_graphs.

    Attributes:
        DEFAULT_WEIGHTS: Name of the weights loaded by from_pretrained.
        UNUSED_PARAMETERS: Prefixes of the parameters of the pretrained weights
            which the model does not have.
        GRAPH_FIELDS: Fields of Structures the model reads, set by transform:
            "graph", and "line_graph", "triplets", "quadruplets" or "bonds".
    """

    DEFAULT_WEIGHTS: str
    UNUSED_PARAMETERS: tuple[str, ...] = ()
    GRAPH_FIELDS: tuple[str, ...] = ("graph",)

    @classmethod
    def transform(cls) -> Transform:
        """Transform attaching the graph and the tuples of edges of the model.

        It does not depend on the weights: it is given by the class, to build
        a dataset before a model is loaded.
        """
        raise NotImplementedError

    def check_graphs(self, structures: Structures) -> None:
        """Check that structures have the graph and the tuples of edges of the model.

        What is attached is expected to be what transform attaches: only its
        presence is checked.

        Raises:
            ValueError: If a field of GRAPH_FIELDS is not attached.
        """
        missing = [
            name for name in self.GRAPH_FIELDS if getattr(structures, name) is None
        ]
        if missing:
            name = type(self).__name__
            raise ValueError(
                f"{name} needs structures with their {', '.join(missing)}: "
                f"apply {name}.transform() to the structures"
            )

    @classmethod
    def from_pretrained(
        cls: type[Model],
        name: str | None = None,
        device: torch.device | str | None = None,
        cache: str | os.PathLike | None = None,
    ) -> Model:
        """Load a model with pretrained weights, downloaded on the first call.

        The model is in evaluation mode and its parameters have no gradient.

        Args:
            name: Name of the weights, listed in the config.yaml of the model.
                DEFAULT_WEIGHTS by default.
            device: Device of the model.
            cache: Directory of the downloaded weights.

        Raises:
            InvalidWeights: If the weights are not the ones of the model.
        """
        model = cls()
        model.load_checkpoint(name or cls.DEFAULT_WEIGHTS, cache)

        return model.to(device).eval().requires_grad_(False)

    def load_checkpoint(
        self, name: str, cache: str | os.PathLike | None = None
    ) -> None:
        """Download pretrained weights, read their files and load them.

        Args:
            name: Name of the weights.
            cache: Directory of the downloaded weights.
        """
        raise NotImplementedError

    def load_parameters(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load parameters, which must be the ones of the model.

        Parameters whose name starts with one of UNUSED_PARAMETERS are left out.

        Raises:
            InvalidWeights: If parameters are missing, unknown or do not have
                the shape of the ones of the model.
        """
        used = {
            name: values
            for name, values in state_dict.items()
            if not name.startswith(self.UNUSED_PARAMETERS)
        }

        try:
            self.load_state_dict(used, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e
