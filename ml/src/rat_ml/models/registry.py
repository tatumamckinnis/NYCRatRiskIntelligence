"""Versioned model artifact registry (T-16).

Saves and loads trained model objects alongside metadata to a timestamped
directory structure.  A root-level ``registry.json`` tracks the latest
version of each named model so the API can load by name without knowing
the timestamp.

Directory layout::

    <artifacts_dir>/
      registry.json                    ← {model_name: latest_version_path}
      tabular/
        catboost/
          2024-01-15T12-30-00/
            model.joblib
            metadata.json
        lightgbm/
          2024-01-15T12-31-00/
            model.joblib
            metadata.json

Usage::

    registry = ModelRegistry("ml/artifacts")
    path = registry.save("catboost", fitted_model, metadata={"test_pr_auc": 0.71})
    model, meta = registry.load("catboost")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, artifacts_dir: str | Path) -> None:
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_index(self) -> dict[str, str]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _write_index(self, index: dict[str, str]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model_name: str,
        model_obj: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialise *model_obj* under a timestamped version directory.

        Args:
            model_name: Logical name, e.g. ``"catboost"``.
            model_obj:  Any joblib-serialisable object (sklearn estimator,
                        CatBoost model wrapped in a dict, etc.).
            metadata:   Arbitrary JSON-serialisable dict stored alongside
                        the model binary (metrics, feature lists, etc.).

        Returns:
            Path to the version directory that was created.
        """
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        version_dir = self.root / "tabular" / model_name / ts
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_obj, version_dir / "model.joblib")

        meta = metadata or {}
        meta["model_name"] = model_name
        meta["saved_at"] = ts
        (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

        index = self._read_index()
        index[model_name] = str(version_dir)
        self._write_index(index)

        return version_dir

    def load(self, model_name: str) -> tuple[Any, dict[str, Any]]:
        """Load the latest version of *model_name*.

        Returns:
            ``(model_obj, metadata)``

        Raises:
            KeyError: if *model_name* has never been saved.
            FileNotFoundError: if the version directory is missing.
        """
        index = self._read_index()
        if model_name not in index:
            raise KeyError(
                f"Model '{model_name}' not found in registry at {self._index_path}. "
                "Run train_tabular.py first."
            )
        version_dir = Path(index[model_name])
        model_obj = joblib.load(version_dir / "model.joblib")
        metadata = json.loads((version_dir / "metadata.json").read_text())
        return model_obj, metadata

    def list_models(self) -> dict[str, str]:
        """Return ``{model_name: version_dir_path}`` for all registered models."""
        return self._read_index()

    def latest_path(self, model_name: str) -> Path:
        """Return the version directory Path for the latest *model_name*."""
        index = self._read_index()
        if model_name not in index:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        return Path(index[model_name])
