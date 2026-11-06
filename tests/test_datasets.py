"""Tests for the aliases of the published datasets."""

import sys
import types

import pytest

from torchms.data import HDF5Dataset, Structures
from torchms.data.datasets import (
    DATASETS,
    available_datasets,
    dataset_url,
    load_dataset,
    resolve,
)


def test_available_datasets():
    assert available_datasets() == {
        "carbon-24": ("train", "val", "test"),
        "core-mof": ("asr", "fsr"),
        "core-mof-2024": ("asr", "fsr", "ion", "ncr"),
        "lematerial": ("all", "unique"),
        "materials-project": ("all",),
        "mp-20": ("train", "val", "test"),
        "mpts-52": ("train", "val", "test"),
        "oqmd": ("all", "unique"),
        "perov-5": ("train", "val", "test"),
        "qm9": ("all",),
        "qmof": ("all",),
        "zinc250k": ("all",),
    }
    assert set(DATASETS) == set(available_datasets())


@pytest.mark.parametrize(
    "name,split,expected",
    [
        ("mp-20", None, ("materials-toolkits/mp-20", "train.hdf5")),
        ("mp-20", "test", ("materials-toolkits/mp-20", "test.hdf5")),
        ("mp-20/val", None, ("materials-toolkits/mp-20", "val.hdf5")),
        ("MP_20", "val", ("materials-toolkits/mp-20", "val.hdf5")),
        ("perov_5/test", None, ("materials-toolkits/perov-5", "test.hdf5")),
        ("carbon-24", "train", ("materials-toolkits/carbon-24", "train.hdf5")),
        (
            "materials-project",
            None,
            ("materials-toolkits/materials-project", "all.hdf5"),
        ),
        ("oqmd", None, ("materials-toolkits/oqmd", "all.hdf5")),
        ("oqmd/unique", None, ("materials-toolkits/oqmd", "unique.hdf5")),
        ("lematerial", "unique", ("materials-toolkits/lematerial", "unique.hdf5")),
        ("qm9", None, ("materials-toolkits/qm9", "all.hdf5")),
        ("QMOF/all", None, ("materials-toolkits/qmof", "all.hdf5")),
        ("core_mof", None, ("materials-toolkits/core-mof", "asr.hdf5")),
        ("core-mof/fsr", None, ("materials-toolkits/core-mof", "fsr.hdf5")),
        ("zinc250k", None, ("materials-toolkits/zinc250k", "all.hdf5")),
        ("core-mof-2024/ncr", None, ("materials-toolkits/core-mof-2024", "ncr.hdf5")),
    ],
)
def test_resolve(name, split, expected):
    assert resolve(name, split) == expected


def test_resolve_errors():
    with pytest.raises(KeyError, match="unknown dataset"):
        resolve("mp-21")
    with pytest.raises(KeyError, match="unknown split"):
        resolve("mp-20", "all")
    with pytest.raises(KeyError, match="unknown split"):
        resolve("oqmd/train")
    with pytest.raises(ValueError, match="twice"):
        resolve("mp-20/train", "test")


def test_dataset_url():
    assert dataset_url("mp-20", "test") == (
        "https://huggingface.co/datasets/materials-toolkits/mp-20/resolve/main/test.hdf5"
    )
    assert dataset_url("lematerial/unique") == (
        "https://huggingface.co/datasets/materials-toolkits/lematerial/resolve/main/unique.hdf5"
    )
    assert dataset_url("oqmd", revision="abc123").endswith(
        "/oqmd/resolve/abc123/all.hdf5"
    )


def test_load_dataset(tmp_path, monkeypatch):
    hdf5_path = tmp_path / "test.hdf5"
    HDF5Dataset.from_structures(hdf5_path, [Structures.diamond()])
    calls = []

    def hf_hub_download(repo_id, filename, repo_type=None, **kwargs):
        calls.append((repo_id, filename, repo_type, kwargs))
        return str(hdf5_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    dataset = load_dataset("mp-20", "test")
    assert isinstance(dataset, HDF5Dataset) and len(dataset) == 1

    def double(structures):
        structures.z = structures.z * 2
        return structures

    transformed = load_dataset("oqmd/unique", transform=double, revision="abc123")
    assert transformed[0].z.tolist() == [12] * 8

    assert calls == [
        ("materials-toolkits/mp-20", "test.hdf5", "dataset", {}),
        ("materials-toolkits/oqmd", "unique.hdf5", "dataset", {"revision": "abc123"}),
    ]
