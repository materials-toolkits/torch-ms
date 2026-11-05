"""Pretrained models in tests, skipped when their weights are not in the cache."""

from __future__ import annotations

from typing import TypeVar

import pytest
import torch

from torchms.models.estimator import weights

Model = TypeVar("Model", bound=torch.nn.Module)


def cached_or_skip(model: str, name: str) -> None:
    """Skip the test when the files of some weights are not all in the cache.

    Args:
        model: Model of the weights, a key of weights.CHECKPOINTS: "mattergen"...
        name: Name of the weights.
    """
    if not all(path.exists() for path in weights.checkpoint_paths(model, name)):
        pytest.skip(f"weights {name} of {model} are not in the cache")


def pretrained(model_class: type[Model], model: str, name: str) -> Model:
    """A pretrained model on the CPU, or a skip if its weights are not cached.

    Args:
        model_class: Its class, with from_pretrained.
        model: Model of the weights, a key of weights.CHECKPOINTS.
        name: Name of the weights.
    """
    cached_or_skip(model, name)

    return model_class.from_pretrained(name, device="cpu")  # type: ignore[attr-defined]
