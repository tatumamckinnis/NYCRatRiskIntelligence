"""Model artifact loading at API startup (T-18).

Called once during FastAPI lifespan; results stored in ``app.state``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# loader.py lives at api/src/rat_api/ml/loader.py — repo root is 4 levels up
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve(path: str) -> Path:
    """Resolve a path relative to the repo root if it isn't absolute."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def load_models(artifacts_dir: str, model_name: str) -> dict[str, Any]:
    """Load the named model and its metadata from the registry.

    Args:
        artifacts_dir: Root artifact directory (e.g. ``"ml/artifacts"``).
        model_name:    Logical model name (e.g. ``"catboost"``).

    Returns:
        ``{"model": <fitted model>, "metadata": <dict>, "model_name": str}``

    Raises:
        FileNotFoundError: if the registry or artifact files are missing.
        KeyError: if *model_name* is not in the registry.
    """
    registry_path = _resolve(artifacts_dir) / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {registry_path} "
            f"(repo root: {_REPO_ROOT}). "
            "Run ml/scripts/train_tabular.py first."
        )

    import joblib  # noqa: PLC0415

    index: dict[str, str] = json.loads(registry_path.read_text())
    if model_name not in index:
        raise KeyError(
            f"Model '{model_name}' not found in registry {registry_path}. "
            f"Available: {list(index.keys())}"
        )

    version_dir = _resolve(index[model_name])
    model = joblib.load(version_dir / "model.joblib")
    metadata: dict[str, Any] = json.loads((version_dir / "metadata.json").read_text())

    return {
        "model": model,
        "metadata": metadata,
        "model_name": model_name,
        "version": version_dir.name,
    }
