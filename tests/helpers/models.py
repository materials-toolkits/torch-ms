"""Running the models in the tests."""

from __future__ import annotations

import torch

from torchms.data import Structures


def predict(model: torch.nn.Module, structures: Structures) -> torch.Tensor:
    """Prediction of a model for structures without graph: with its transform.

    A pretrained model reads the graphs its transform attaches, a classmethod.
    """
    return model(model.transform()(structures))  # type: ignore[operator]
