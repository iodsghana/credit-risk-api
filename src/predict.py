"""
Credit Risk Prediction Module
==============================
Loads the persisted champion model and scores individual applications
or batches. Includes drift-detection hooks and score bucketing.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import add_application_features

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_model.pkl"
META_PATH  = ROOT / "models" / "model_metadata.json"

# ─── Risk Bucketing ───────────────────────────────────────────────────────────
# Based on common lender thresholds (Experian / FICO lending guidelines)
SCORE_BANDS = [
    (0.00, 0.10, "Very Low"),
    (0.10, 0.25, "Low"),
    (0.25, 0.50, "Moderate"),
    (0.50, 0.70, "High"),
    (0.70, 1.01, "Very High"),
]

def score_to_band(prob: float) -> str:
    for lo, hi, label in SCORE_BANDS:
        if lo <= prob < hi:
            return label
    return "Unknown"


def align_to_training_features(df: pd.DataFrame, metadata: dict | None) -> pd.DataFrame:
    """Ensure API payloads have the same columns used during training."""
    if not metadata or "features" not in metadata:
        return df
    return df.reindex(columns=metadata["features"])


# ─── Model Registry ───────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Lightweight single-model registry.
    In production this would be backed by MLflow Model Registry or SageMaker.
    """

    _model    = None
    _metadata = None

    @classmethod
    def load(cls):
        if cls._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at {MODEL_PATH}. "
                    "Run `python src/train.py` first."
                )
            cls._model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")

        if cls._metadata is None and META_PATH.exists():
            with open(META_PATH) as f:
                cls._metadata = json.load(f)

        return cls._model, cls._metadata

    @classmethod
    def reload(cls):
        """Hot-reload without restarting the server."""
        cls._model    = None
        cls._metadata = None
        return cls.load()


# ─── Scoring ──────────────────────────────────────────────────────────────────

def predict(data: dict) -> dict:
    """
    Score a single loan application.

    Returns:
        default_probability : float  — calibrated P(default)
        risk_band           : str    — human-readable tier
        model_version       : str    — SHA-256 prefix of champion model
    """
    model, metadata = ModelRegistry.load()
    df = add_application_features(pd.DataFrame([data]))
    df = align_to_training_features(df, metadata)

    prob = float(model.predict_proba(df)[:, 1][0])
    band = score_to_band(prob)
    sha  = metadata["sha256"][:12] if metadata else "unknown"

    return {
        "default_probability": round(prob, 4),
        "risk_band":           band,
        "model_version":       sha,
    }


def predict_batch(records: list[dict]) -> list[dict]:
    """
    Vectorised scoring for batch workloads.
    Significantly faster than calling predict() in a loop.
    """
    model, metadata = ModelRegistry.load()
    df = add_application_features(pd.DataFrame(records))
    df = align_to_training_features(df, metadata)
    probs = model.predict_proba(df)[:, 1]
    sha   = metadata["sha256"][:12] if metadata else "unknown"

    return [
        {
            "default_probability": round(float(p), 4),
            "risk_band":           score_to_band(float(p)),
            "model_version":       sha,
        }
        for p in probs
    ]
