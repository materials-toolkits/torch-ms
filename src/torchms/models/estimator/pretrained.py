"""Loading of pretrained weights, shared by the pretrained models."""

from __future__ import annotations

import os
from typing import Generic, Mapping, TypeVar

import torch
from jaxtyping import Int
from torch import nn

from torchms.data.augmentation.transforms import Transform
from torchms.data.batching import Structures
from torchms.models.estimator.config import YamlConfig

Model = TypeVar("Model", bound="PretrainedModel")
ConfigT = TypeVar("ConfigT", bound=YamlConfig)


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


def check_configuration(
    found: Mapping[str, object], expected: Mapping[str, Mapping[str, object]]
) -> None:
    """Check the configuration of a training against the one a rewrite is written for.

    For the weights published with the configuration of their training, a
    file of nested sections (hparams.yaml of hydra).

    Args:
        found: Configuration stored with the weights.
        expected: Arguments of the rewrite, by section. A section inside
            another one is named with dots: "model.decoder".

    Raises:
        InvalidWeights: If a section is missing, or if one of its arguments
            has another value than the expected one, see different_arguments.

    Examples:
        >>> expected = {"model": {"radius": 7.0}, "model.decoder": {"layers": 3}}
        >>> check_configuration({"model": {"radius": 7.0, "decoder": {}}}, expected)
        >>> check_configuration({"model": {"decoder": {"layers": 4}}}, expected)
        Traceback (most recent call last):
        ...
        torchms.models.estimator.pretrained.InvalidWeights: weights of a model with \
another configuration: {'model.decoder.layers': 4}
    """
    different: dict[str, object] = {}

    for section, arguments in expected.items():
        values: object = found
        for key in section.split("."):
            values = values.get(key) if isinstance(values, Mapping) else None
        if not isinstance(values, Mapping):
            raise InvalidWeights(f"the configuration has no {section}")

        different.update(
            {
                f"{section}.{key}": value
                for key, value in different_arguments(values, arguments).items()
            }
        )

    if different:
        raise InvalidWeights(
            f"weights of a model with another configuration: {different}"
        )


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


class PretrainedModel(nn.Module, Generic[ConfigT]):
    """Model which loads the pretrained weights of the project it is derived from.

    A model is built from its configuration, the Config of its directory, which
    it is declared with, `class M3GNet(PretrainedModel[Config])`:
    without argument, the one of its config.yaml, which the pretrained weights
    are for. It reads the files of its weights in load_checkpoint.

    A model does not build graphs: it reads the graph and the tuples of edges
    attached to the structures, and gives the transform which attaches them.
    The transform is applied when the structures are loaded:

    ```python
    dataset = HDF5Dataset("structures.hdf5", transform=Model.transform())
    ```

    or to a batch, `model(Model.transform()(structures))`. Structures without
    them are refused, see check_graphs.

    Args:
        config: Configuration of the model: its hyperparameters, the name of
            its default weights and the parameters of these weights it does
            not have. The one of its config.yaml by default.

    Attributes:
        CONFIG: Class of the configuration of the model, the Config of its
            directory.
        GRAPH_FIELDS: Fields of Structures the model reads, set by transform:
            "graph", and "line_graph", "triplets", "quadruplets" or "bonds".
        config: Configuration the model was built from.
    """

    CONFIG: type[ConfigT]
    GRAPH_FIELDS: tuple[str, ...] = ("graph",)

    def __init__(self, config: ConfigT | None = None):
        super().__init__()
        self.config: ConfigT = self.CONFIG.load() if config is None else config

    def default_weights(self) -> str:
        """Name of the weights loaded by from_pretrained, from the configuration."""
        return self.config.default_weights

    def unused_parameters(self) -> tuple[str, ...]:
        """Prefixes of the parameters of the pretrained weights the model does not have.

        From the configuration. They are left out by load_parameters.
        """
        return self.config.unused_parameters

    @classmethod
    def transform(cls, config: ConfigT | None = None) -> Transform:
        """Transform attaching the graph and the tuples of edges of the model.

        It does not depend on the weights: it is given by the class, to build
        a dataset before a model is loaded.

        Args:
            config: Configuration of the model the graph is for. The one of
                its config.yaml by default.
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
                The default_weights of its configuration by default.
            device: Device of the model.
            cache: Directory of the downloaded weights.

        Raises:
            InvalidWeights: If the weights are not the ones of the model.
        """
        model = cls()
        model.load_checkpoint(name or model.default_weights(), cache)

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

        Parameters whose name starts with one of unused_parameters are left out.

        Raises:
            InvalidWeights: If parameters are missing, unknown or do not have
                the shape of the ones of the model.
        """
        used = {
            name: values
            for name, values in state_dict.items()
            if not name.startswith(self.unused_parameters())
        }

        try:
            self.load_state_dict(used, strict=True)
        except RuntimeError as e:
            raise InvalidWeights(str(e)) from e
