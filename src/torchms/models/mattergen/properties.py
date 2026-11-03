# Derived from MatterGen (https://github.com/microsoft/mattergen), Copyright (c)
# Microsoft Corporation, MIT license (see LICENSE in this directory).
#
# Changes: the embeddings of the properties of the models of MatterGen
# fine-tuned on them (mattergen/property_embeddings.py, StandardScalerTorch of
# mattergen/common/utils/data_utils.py, NoiseLevelEncoding of
# mattergen/diffusion/model_utils.py), for inference only: no fitting of the
# scaler, no unconditional embedding (a property without value is not given to
# the network, as the zero embedding of the adapters of MatterGen). Names of
# modules and parameters follow MatterGen to load its weights.
"""Embeddings of the properties MatterGen is fine-tuned on.

A model fine-tuned on properties holds one PropertyEmbedding per property in
property_embeddings_adapt, named as in MatterGen. A property is one of three
kinds:

* a chemical system, the elements a crystal can have: 1 per atomic number
  through a linear layer. The scores of the other elements are lowered;
* a space group, from 1 to 230: a learned vector per group;
* a number (band gap, magnetic density, energy above the hull, bulk
  modulus): standardized by the statistics of the training set, for the bulk
  modulus the ones of its decimal logarithm, then embedded by sines and
  cosines as the time.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import nn

from torchms.data.constants import symbol_to_number
from torchms.models.layers.basis import sinusoidal_frequencies, sinusoids
from torchms.models.mattergen.config import Config

CHEMICAL_SYSTEM = "chemical_system"
SPACE_GROUP = "space_group"
# space groups of the international tables, from 1
NUM_SPACE_GROUPS = 230


class MultiHotEmbedding(nn.Module):
    """Linear layer on 1 per atomic number of a chemical system, from 0."""

    def __init__(self, num_elements: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Linear(num_elements + 1, hidden_dim)

    def forward(
        self, x: Float[torch.Tensor, "n num_elements_1"]
    ) -> Float[torch.Tensor, "n hidden_dim"]:
        return self.embedding(x)


class SpaceGroupVector(nn.Module):
    """A learned vector per space group, from 1."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(NUM_SPACE_GROUPS, hidden_dim)

    def forward(self, x: torch.Tensor) -> Float[torch.Tensor, "n hidden_dim"]:
        return self.embedding(x.long() - 1)


class NoiseLevelEncoding(nn.Module):
    """Sines and cosines of a number, each sine before its cosine, as the time.

    Attributes:
        div_term: Frequencies, from the weights: the ones of the time.
    """

    div_term: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        network = config.model
        self.register_buffer(
            "div_term",
            sinusoidal_frequencies(
                network.hidden_dim, network.time_max_period, endpoint=False
            ),
        )

    def forward(
        self, x: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "n hidden_dim"]:
        return sinusoids(x.unsqueeze(1) * self.div_term, interleave=True)


class StandardScaler(nn.Module):
    """A number minus the mean of the training set, over its deviation.

    Args:
        log10: Whether the number is first replaced by its decimal logarithm,
            the statistics being the ones of the logarithms.

    Attributes:
        means: Mean of the training set, from the weights.
        stds: Its deviation, from the weights.
    """

    means: torch.Tensor
    stds: torch.Tensor

    def __init__(self, log10: bool = False):
        super().__init__()
        self.log10 = log10
        self.register_buffer("means", torch.zeros(1))
        self.register_buffer("stds", torch.ones(1))

    def forward(self, x: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "n"]:
        if self.log10:
            x = torch.log10(x)
        return (x - self.means) / self.stds


class PropertyEmbedding(nn.Module):
    """Embedding of a property, read by the adapters of GemNet-T.

    Args:
        name: Name of the property in MatterGen.
        conditional_embedding_module: Embedding of its values.
        scaler: Standardization of its values before, for a number.
    """

    def __init__(
        self,
        name: str,
        conditional_embedding_module: nn.Module,
        scaler: nn.Module | None = None,
    ):
        super().__init__()
        self.name = name
        self.conditional_embedding_module = conditional_embedding_module
        self.scaler = scaler if scaler is not None else nn.Identity()

    def forward(self, values: torch.Tensor) -> Float[torch.Tensor, "n hidden_dim"]:
        return self.conditional_embedding_module(self.scaler(values))

    def parse(
        self, values: Sequence[object] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Values of the property of each crystal, as the model reads them.

        A number per crystal by default.

        Raises:
            ValueError: If a value is not a finite number.
        """
        numbers = torch.as_tensor(values, dtype=torch.float32, device=device)
        if not numbers.isfinite().all():
            raise ValueError(
                f"values of {self.name} must be finite numbers: {numbers.tolist()}"
            )

        return numbers

    def kept_classes(
        self, values: torch.Tensor, conditioned: Bool[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct num_classes"] | None:
        """Classes of the elements each crystal can take, 1 for a kept one.

        None, every class, but for a chemical system.
        """
        del values, conditioned


class ChemicalSystemEmbedding(PropertyEmbedding):
    """Embedding of a chemical system, which also restricts the elements."""

    def __init__(self, config: Config):
        network = config.model
        super().__init__(
            CHEMICAL_SYSTEM, MultiHotEmbedding(network.num_elements, network.hidden_dim)
        )
        self.num_elements = network.num_elements

    def parse(
        self, values: Sequence[object] | torch.Tensor, device: torch.device
    ) -> Float[torch.Tensor, "n_struct num_elements_1"]:
        """1 per atomic number of the elements of each system, from 0.

        Args:
            values: Chemical system of each crystal, its elements joined by
                dashes: "Li-O". Or the values already parsed.

        Raises:
            ValueError: If an element is not known or comes after fermium.
        """
        if isinstance(values, torch.Tensor):
            return values.to(device=device, dtype=torch.float32)

        pairs = [
            (row, symbol)
            for row, system in enumerate(values)
            for symbol in str(system).split("-")
        ]
        numbers = []
        for _, symbol in pairs:
            try:
                number = symbol_to_number(symbol)
            except KeyError:
                number = 0
            if not 1 <= number <= self.num_elements:
                raise ValueError(f"element {symbol} is not supported by the model")
            numbers.append(number)

        systems = torch.zeros((len(values), self.num_elements + 1), device=device)
        rows = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=device)
        systems[rows, torch.tensor(numbers, dtype=torch.long, device=device)] = 1.0

        return systems

    def kept_classes(
        self,
        values: Float[torch.Tensor, "n_struct num_elements_1"],
        conditioned: Bool[torch.Tensor, "n_struct"],
    ) -> Float[torch.Tensor, "n_struct num_classes"]:
        """The elements of the chemical system of each conditioned crystal.

        The classes are the elements from hydrogen, then the mask, which is
        never kept. A crystal without condition keeps every class.
        """
        keep = F.pad(values[:, 1:], (0, 1))
        return torch.where(conditioned.unsqueeze(1), keep, torch.ones_like(keep))


class NumberEmbedding(PropertyEmbedding):
    """Embedding of a number: standardized, then sines and cosines.

    Args:
        name: Name of the property in MatterGen.
        config: Configuration of the model.
        log10: Whether the number is standardized after its decimal
            logarithm, which needs it positive.
    """

    def __init__(self, name: str, config: Config, log10: bool = False):
        super().__init__(name, NoiseLevelEncoding(config), StandardScaler(log10))
        self.log10 = log10

    def parse(
        self, values: Sequence[object] | torch.Tensor, device: torch.device
    ) -> Float[torch.Tensor, "n_struct"]:
        """A number per crystal.

        Raises:
            ValueError: If a value is not a finite number, or is not positive
                for a number standardized after its logarithm.
        """
        numbers = super().parse(values, device)
        if self.log10 and not (numbers > 0).all():
            raise ValueError(
                f"values of {self.name} must be positive: {numbers.tolist()}"
            )

        return numbers


class SpaceGroupEmbedding(PropertyEmbedding):
    """Embedding of a space group: a learned vector per group."""

    def __init__(self, config: Config):
        super().__init__(SPACE_GROUP, SpaceGroupVector(config.model.hidden_dim))

    def parse(
        self, values: Sequence[object] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Number of the space group of each crystal, from 1 to 230.

        Raises:
            ValueError: If a number is not one of a space group.
        """
        numbers = torch.as_tensor(values, dtype=torch.long, device=device)
        outside = (numbers < 1) | (numbers > NUM_SPACE_GROUPS)
        if outside.any():
            raise ValueError(f"no space group {numbers[outside].tolist()}")

        return numbers


def property_embedding(
    name: str, config: Config, fine_tuning: str | None = None
) -> PropertyEmbedding:
    """Embedding of a property of MatterGen, from its name.

    Args:
        name: Name of the property in MatterGen.
        config: Configuration of the model.
        fine_tuning: The model fine-tuned on it, see Config.condition: a
            number is standardized after its decimal logarithm if its
            configuration says so, see Config.log10_scaled.

    Examples:
        >>> embedding = property_embedding("energy_above_hull", Config.load())
        >>> type(embedding.conditional_embedding_module).__name__
        'NoiseLevelEncoding'
        >>> property_embedding("chemical_system", Config.load()).parse(
        ...     ["Li-O"], torch.device("cpu")
        ... ).nonzero().tolist()
        [[0, 3], [0, 8]]
    """
    if name == CHEMICAL_SYSTEM:
        return ChemicalSystemEmbedding(config)
    if name == SPACE_GROUP:
        return SpaceGroupEmbedding(config)

    return NumberEmbedding(name, config, config.log10_scaled(fine_tuning, name))
