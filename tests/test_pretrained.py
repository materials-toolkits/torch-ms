"""Tests for the loading of pretrained weights shared by the estimators."""

import pytest
import torch
from torch import nn

from torchms.data import Structures
from torchms.models import (
    AlignnModel,
    ChgnetEnergyModel,
    M3gnetEnergyModel,
    MaceEnergyModel,
    OrbEnergyModel,
    alignn,
    chgnet,
    m3gnet,
    mace,
    orb,
)
from torchms.models.estimator import weights
from torchms.models.estimator.pretrained import (
    InvalidWeights,
    PretrainedModel,
    different_arguments,
    element_index,
    element_table,
)


class Small(PretrainedModel):
    """Model with one layer, whose weights are written with another parameter."""

    DEFAULT_WEIGHTS = "small"
    UNUSED_PARAMETERS = ("head.",)

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.loaded = None

    def load_checkpoint(self, name, cache=None):
        self.loaded = (name, cache)
        self.load_parameters(
            {**nn.Linear(2, 1).state_dict(prefix="layer."), "head.bias": torch.ones(1)}
        )


def test_invalid_weights_is_shared():
    packages = (alignn, chgnet, m3gnet, mace, orb)
    assert all(package.InvalidWeights is InvalidWeights for package in packages)


def test_different_arguments():
    expected = {"cutoff": 5.0, "layers": [64, 64]}

    assert different_arguments({}, expected) == {}
    assert different_arguments({"cutoff": 5.0, "layers": [64, 64]}, expected) == {}
    assert different_arguments({"cutoff": 5, "unknown": 1}, expected) == {}
    assert different_arguments({"cutoff": None, "layers": [64]}, expected) == {
        "cutoff": None,
        "layers": [64],
    }


def test_element_index():
    table = element_table(torch.tensor([1, 6, 8, 94]))

    assert table.shape == (119,)
    assert (table >= 0).sum() == 4
    assert element_index(torch.tensor([94, 6, 1, 6]), table).tolist() == [3, 1, 0, 1]

    with pytest.raises(ValueError, match=r"elements \[2, 26\] are not supported"):
        element_index(torch.tensor([26, 1, 2, 26]), table)


def test_named():
    checkpoints = {"a": 1, "b": 2}

    assert weights.named(checkpoints, "b") == 2
    with pytest.raises(KeyError, match='unknown model "c", must be one of'):
        weights.named(checkpoints, "c")


def test_load_parameters():
    model = Small()
    state_dict = nn.Linear(2, 1).state_dict(prefix="layer.")

    model.load_parameters({**state_dict, "head.weight": torch.zeros(3)})
    assert torch.equal(model.layer.weight, state_dict["layer.weight"])

    with pytest.raises(InvalidWeights, match="Unexpected"):
        model.load_parameters({**state_dict, "other.weight": torch.zeros(3)})
    with pytest.raises(InvalidWeights, match="Missing"):
        model.load_parameters({"layer.weight": state_dict["layer.weight"]})
    with pytest.raises(InvalidWeights, match="size mismatch"):
        model.load_parameters({**state_dict, "layer.bias": torch.zeros(2)})


def test_from_pretrained(tmp_path):
    model = Small.from_pretrained()

    assert isinstance(model, Small)
    assert model.loaded == ("small", None)
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())

    assert Small.from_pretrained("other", cache=tmp_path).loaded == ("other", tmp_path)

    with pytest.raises(NotImplementedError):
        PretrainedModel().load_checkpoint("small")


MODELS = (
    AlignnModel,
    ChgnetEnergyModel,
    M3gnetEnergyModel,
    MaceEnergyModel,
    OrbEnergyModel,
)


def test_transform_is_not_implemented_by_default():
    with pytest.raises(NotImplementedError):
        PretrainedModel.transform()


# the layers of some models are slower than the default timeout
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cls", MODELS)
def test_model_reads_the_graphs_of_its_transform(cls):
    torch.manual_seed(0)
    model = cls().eval().requires_grad_(False)
    items = [Structures.diamond(), Structures.zincblende()]

    # the transform is given by the class
    transform = cls.transform()
    expected = model(transform(Structures.collate(items)))
    assert expected.shape == (2,) and expected.isfinite().all()

    # applied structure by structure, before they are batched
    prepared = Structures.collate([transform(item) for item in items])
    for name in cls.GRAPH_FIELDS:
        assert getattr(prepared, name) is not None
    assert torch.equal(model(prepared), expected)


@pytest.mark.parametrize("cls", MODELS)
def test_model_does_not_build_graphs(cls):
    model = cls().eval()
    message = f"apply {cls.__name__}.transform"

    with pytest.raises(ValueError, match=message):
        model(Structures.diamond())

    # a graph alone is not completed with the tuples of edges of the model
    if len(cls.GRAPH_FIELDS) > 1:
        from torchms.graph.transforms import AddRadiusGraph

        structures = AddRadiusGraph(3.0, self_images=True)(Structures.diamond())
        with pytest.raises(ValueError, match=message):
            model(structures)


def test_checkpoints_are_read_from_the_configurations(tmp_path):
    assert set(weights.CHECKPOINTS) == {
        "orb",
        "chgnet",
        "m3gnet",
        "mace",
        "alignn",
        "cdvae",
        "diffcsp",
    }
    for model, names in weights.CHECKPOINTS.items():
        for name, files in names.items():
            assert name.startswith(model)
            for file in files:
                assert file.url.startswith("https://") and len(file.sha256) == 64

    listing = tmp_path / "config.yaml"
    listing.write_text(
        "checkpoints:\n"
        "  small-1:\n"
        "    - url: https://example.org/a.pt\n"
        "      sha256: abc\n"
        "    - url: https://example.org/b\n"
        "      sha256: def\n"
        "      filename: b.pt\n"
    )
    assert weights.read_checkpoints(listing) == {
        "small-1": (
            weights.Checkpoint("https://example.org/a.pt", "abc"),
            weights.Checkpoint("https://example.org/b", "def", "b.pt"),
        )
    }


@pytest.mark.parametrize("cls", MODELS)
def test_configuration_of_a_model(cls):
    import importlib

    from torchms.models.estimator.config import config_files, load_config

    module = importlib.import_module(cls.__module__)
    directory = cls.__module__.split(".")[-2]
    config = load_config(module.__file__)

    # one file per model, next to its module
    assert config_files()[directory].samefile(
        module.__file__.replace("model.py", "config.yaml")
    )
    assert load_config(config_files()[directory]) is config
    assert module.CONFIG is config

    # the model takes its values from it
    assert cls.DEFAULT_WEIGHTS == config["default_weights"]
    assert cls.DEFAULT_WEIGHTS in config["checkpoints"]
    assert tuple(cls.UNUSED_PARAMETERS) == tuple(config.get("unused_parameters", ()))
    assert module.RADIUS == config["graph"]["radius"]
    assert isinstance(config["model"], dict) and len(config["model"]) > 0


def test_checkpoint_paths(tmp_path):
    # one file: in the directory of the model
    (path,) = weights.checkpoint_paths("mace", "mace-mp-0-small", tmp_path)
    assert path.parent == tmp_path / "mace"

    # several files: in a directory of their own
    paths = weights.checkpoint_paths("m3gnet", "m3gnet-matpes-pbe-2025.2", tmp_path)
    assert [path.name for path in paths] == ["state.pt", "model.pt"]
    assert {path.parent for path in paths} == {
        tmp_path / "m3gnet" / "m3gnet-matpes-pbe-2025.2"
    }

    with pytest.raises(KeyError, match='unknown model "resnet"'):
        weights.checkpoint_paths("resnet", "resnet-50")
    with pytest.raises(KeyError, match='unknown model "mace-9"'):
        weights.download("mace", "mace-9", tmp_path)
