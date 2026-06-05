"""TFT model trainer for the 12-week rat-risk forecast (T-23).

Uses ``darts.models.TFTModel`` (PyTorch Forecasting backend).

Training strategy
-----------------
- Input chunk length:  52 weeks (1 year of context)
- Output chunk length: 12 weeks (forecast horizon)
- Quantile regression: p10, p50, p90 — gives CI bands for the API
- Early stopping on validation loss (patience=5)
- Model saved via :class:`~rat_ml.models.registry.ModelRegistry`

Quantile output is stored as ``{p10, p50, p90}`` in the metadata so that
the API can expose ``ci_low = p10``, ``risk_score = p50``, ``ci_high = p90``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rat_ml.features.tft_dataset import TFTDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters (defaults; override via train_tft())
# ---------------------------------------------------------------------------

INPUT_CHUNK_LENGTH: int = 52   # weeks of history fed to the encoder
OUTPUT_CHUNK_LENGTH: int = 12  # weeks to forecast
N_EPOCHS: int = 50
BATCH_SIZE: int = 64
HIDDEN_SIZE: int = 64
LSTM_LAYERS: int = 1
NUM_ATTENTION_HEADS: int = 4
DROPOUT: float = 0.1
LEARNING_RATE: float = 1e-3
QUANTILES: list[float] = [0.1, 0.5, 0.9]


@dataclass
class TFTTrainResult:
    model_name: str
    model_path: Path
    metadata: dict[str, Any]
    val_loss: float
    n_series: int


def train_tft(
    train_dataset: TFTDataset,
    val_dataset: TFTDataset,
    *,
    artifacts_dir: str = "ml/artifacts",
    input_chunk_length: int = INPUT_CHUNK_LENGTH,
    output_chunk_length: int = OUTPUT_CHUNK_LENGTH,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    lstm_layers: int = LSTM_LAYERS,
    num_attention_heads: int = NUM_ATTENTION_HEADS,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
    quantiles: list[float] | None = None,
    accelerator: str = "auto",
) -> TFTTrainResult:
    """Train a TFT model on *train_dataset* and save to the model registry.

    Args:
        train_dataset:       Training series bundles.
        val_dataset:         Validation series bundles (for early stopping).
        artifacts_dir:       Root artifact directory.
        input_chunk_length:  Encoder context length (weeks).
        output_chunk_length: Forecast horizon (weeks).
        n_epochs:            Maximum training epochs.
        batch_size:          Mini-batch size.
        hidden_size:         Hidden dimension for LSTM encoder/decoder.
        lstm_layers:         Number of LSTM stacked layers.
        num_attention_heads: Number of self-attention heads.
        dropout:             Dropout probability.
        learning_rate:       Initial Adam learning rate.
        quantiles:           Quantile levels for probabilistic output.
        accelerator:         PyTorch Lightning accelerator (``"auto"``, ``"cpu"``, ``"gpu"``).

    Returns:
        :class:`TFTTrainResult`
    """
    from darts.models import TFTModel  # noqa: PLC0415
    from pytorch_lightning.callbacks import EarlyStopping  # noqa: PLC0415

    from rat_ml.models.registry import ModelRegistry  # noqa: PLC0415

    if quantiles is None:
        quantiles = QUANTILES

    registry = ModelRegistry(artifacts_dir)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-4,
        mode="min",
    )

    from darts.utils.likelihood_models import QuantileRegression  # noqa: PLC0415

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        likelihood=QuantileRegression(quantiles),
        optimizer_kwargs={"lr": learning_rate},
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "callbacks": [early_stop],
            "enable_progress_bar": True,
        },
        model_name="tft_rat_risk",
        work_dir=str(Path(artifacts_dir) / "tft_checkpoints"),
        save_checkpoints=True,
        force_reset=True,
        random_state=42,
    )

    log.info(
        "Training TFT on %d series, input_chunk=%d, output_chunk=%d",
        len(train_dataset.bundles),
        input_chunk_length,
        output_chunk_length,
    )

    model.fit(
        series=train_dataset.targets,
        past_covariates=train_dataset.past_covariates,
        future_covariates=train_dataset.future_covariates,
        val_series=val_dataset.targets,
        val_past_covariates=val_dataset.past_covariates,
        val_future_covariates=val_dataset.future_covariates,
        verbose=True,
    )

    # Retrieve best validation loss from trainer
    try:
        val_loss = float(
            model.trainer.callback_metrics.get("val_loss", float("nan"))  # type: ignore[union-attr]
        )
    except Exception:  # noqa: BLE001
        val_loss = float("nan")

    metadata = {
        "model_type": "TFTModel",
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "quantiles": quantiles,
        "hidden_size": hidden_size,
        "lstm_layers": lstm_layers,
        "num_attention_heads": num_attention_heads,
        "dropout": dropout,
        "n_series_train": len(train_dataset.bundles),
        "n_series_val": len(val_dataset.bundles),
        "val_loss": val_loss if not np.isnan(val_loss) else None,
        "nta_ids": train_dataset.nta_ids(),
    }

    model_path = registry.save("tft", model, metadata=metadata)
    log.info("TFT model saved to %s", model_path)

    return TFTTrainResult(
        model_name="tft",
        model_path=model_path,
        metadata=metadata,
        val_loss=val_loss,
        n_series=len(train_dataset.bundles),
    )


def load_tft(artifacts_dir: str = "ml/artifacts") -> Any:
    """Load the latest TFT model from its Darts checkpoint.

    Uses ``TFTModel.load_from_checkpoint`` (native Darts serialisation) rather
    than joblib, because PyTorch-Lightning internals are not joblib-serialisable.

    Returns:
        ``darts.models.TFTModel`` instance (loaded from best checkpoint).
    """
    from darts.models import TFTModel  # noqa: PLC0415

    checkpoint_work_dir = str(Path(artifacts_dir) / "tft_checkpoints")
    log.info("Loading TFT from checkpoint: %s / tft_rat_risk", checkpoint_work_dir)
    return TFTModel.load_from_checkpoint(
        model_name="tft_rat_risk",
        work_dir=checkpoint_work_dir,
        best=True,
    )


def forecast_nta(
    model: Any,
    bundle_target: "Any",  # darts.TimeSeries
    bundle_past_cov: "Any",
    bundle_future_cov: "Any",
    *,
    horizon: int = OUTPUT_CHUNK_LENGTH,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Produce a probabilistic forecast for one NTA.

    Args:
        model:             Fitted ``TFTModel``.
        bundle_target:     Target ``TimeSeries`` (full history, up to now).
        bundle_past_cov:   Past covariate ``TimeSeries`` (must extend at least to now).
        bundle_future_cov: Future covariate ``TimeSeries`` (must extend ``horizon`` steps beyond now).
        horizon:           Number of weeks to forecast.
        num_samples:       Monte Carlo samples for quantile estimates.

    Returns:
        DataFrame with columns ``week``, ``p10``, ``p50``, ``p90``.
    """
    pred = model.predict(
        n=horizon,
        series=bundle_target,
        past_covariates=bundle_past_cov,
        future_covariates=bundle_future_cov,
        num_samples=num_samples,
    )

    # Extract quantiles from probabilistic forecast
    # Darts >= 0.44 renamed quantile_timeseries() → quantile()
    weeks = pred.time_index
    p10 = pred.quantile(0.10).values().flatten()
    p50 = pred.quantile(0.50).values().flatten()
    p90 = pred.quantile(0.90).values().flatten()

    # Clip to [0, 1] — these are probability estimates
    p10 = np.clip(p10, 0.0, 1.0)
    p50 = np.clip(p50, 0.0, 1.0)
    p90 = np.clip(p90, 0.0, 1.0)

    return pd.DataFrame(
        {
            "week": [w.date() for w in weeks],
            "p10": p10.round(6),
            "p50": p50.round(6),
            "p90": p90.round(6),
        }
    )
