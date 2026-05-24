"""Tests for ModelRegistry (T-16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rat_ml.models.registry import ModelRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "artifacts")


def test_save_creates_version_dir(registry: ModelRegistry) -> None:
    path = registry.save("mymodel", {"weights": [1, 2, 3]})
    assert path.exists()
    assert (path / "model.joblib").exists()
    assert (path / "metadata.json").exists()


def test_save_updates_registry_json(registry: ModelRegistry) -> None:
    registry.save("mymodel", {"w": 1})
    index = json.loads((registry.root / "registry.json").read_text())
    assert "mymodel" in index


def test_load_round_trip(registry: ModelRegistry) -> None:
    obj = {"weights": [1.0, 2.0], "bias": 0.5}
    registry.save("mymodel", obj, metadata={"pr_auc": 0.75})
    loaded, meta = registry.load("mymodel")
    assert loaded == obj
    assert meta["pr_auc"] == pytest.approx(0.75)
    assert meta["model_name"] == "mymodel"


def test_load_latest_after_two_saves(registry: ModelRegistry, monkeypatch) -> None:
    """load() must return the most recently saved version."""
    import time
    registry.save("mymodel", {"v": 1})
    time.sleep(0.01)  # ensure different timestamp
    registry.save("mymodel", {"v": 2})
    loaded, _ = registry.load("mymodel")
    assert loaded == {"v": 2}


def test_load_missing_model_raises_key_error(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError, match="ghost"):
        registry.load("ghost")


def test_list_models_empty(registry: ModelRegistry) -> None:
    assert registry.list_models() == {}


def test_list_models_after_save(registry: ModelRegistry) -> None:
    registry.save("a", 1)
    registry.save("b", 2)
    models = registry.list_models()
    assert "a" in models
    assert "b" in models


def test_metadata_stored(registry: ModelRegistry) -> None:
    registry.save("x", "obj", metadata={"foo": "bar", "n": 42})
    _, meta = registry.load("x")
    assert meta["foo"] == "bar"
    assert meta["n"] == 42


def test_latest_path(registry: ModelRegistry) -> None:
    path = registry.save("m", object())
    assert registry.latest_path("m") == path
