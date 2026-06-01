"""
Train a Home Credit default risk model.

The champion is selected by validation ROC-AUC after joining application,
bureau, and previous-application history.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "monitoring"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from data_validation import write_validation_report
from preprocess import RAW_DATA_DIR, build_preprocessing_pipeline, load_training_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "training.log")],
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_candidates(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", build_preprocessing_pipeline()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", build_preprocessing_pipeline()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=30,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("prep", build_preprocessing_pipeline()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("prep", build_preprocessing_pipeline()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="auc",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        verbosity=0,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, y_pred)
    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def plot_score_distribution(model: Pipeline, X_test, y_test, model_name: str) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="Repaid", color="#17324D")
    ax.hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Default", color="#A83F39")
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Applications")
    ax.set_title(f"Home Credit Score Distribution - {model_name}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(LOG_DIR / f"score_distribution_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(model: Pipeline, X_test, y_test, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[0].set_title("ROC Curve")
    axes[1].set_title("Precision-Recall Curve")
    fig.suptitle(f"Home Credit Evaluation - {model_name}")
    fig.tight_layout()
    fig.savefig(LOG_DIR / f"evaluation_curves_{model_name}.png", dpi=150)
    plt.close(fig)


def compute_permutation_importance(model: Pipeline, X_test, y_test) -> list[dict]:
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[X_sample.index]
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        sorted(
            zip(X_sample.columns, result.importances_mean, result.importances_std),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
    )
    rows = [
        {"feature": feature, "importance_mean": round(float(mean), 6), "importance_std": round(float(std), 6)}
        for feature, mean, std in importance
    ]
    with open(LOG_DIR / "feature_importance.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    return rows


def write_model_card(metadata: dict, feature_importance: list[dict]) -> Path:
    metrics = metadata["test_metrics"]
    confusion = metrics["confusion_matrix"]
    importance_lines = "\n".join(
        f"- `{row['feature']}`: {row['importance_mean']}"
        for row in feature_importance[:10]
    )
    content = f"""# Model Card: Home Credit Default Risk

## Intended Use

Estimate loan default probability for Home Credit application records. This is a portfolio-grade risk scoring system, not a production credit decision engine.

## Dataset

- Source: Home Credit Default Risk Kaggle competition.
- Rows used for this run: {metadata['train_rows'] + metadata['test_rows']:,}
- Holdout split: {metadata['test_rows']:,} rows
- Observed default rate: {metadata['default_rate']:.2%}
- Run mode: {metadata.get('training_mode', 'full_dataset')}

## Model Selection

Champion model: `{metadata['model_name']}`

| Metric | Value |
| --- | ---: |
| ROC-AUC | {metrics['roc_auc']} |
| Average precision | {metrics['avg_precision']} |
| Brier score | {metrics['brier_score']} |

Confusion matrix at threshold 0.50:

```text
[[TN={confusion[0][0]}, FP={confusion[0][1]}],
 [FN={confusion[1][0]}, TP={confusion[1][1]}]]
```

## Top Permutation Importances

{importance_lines}

## Risk and Governance Notes

- `TARGET` is removed before training.
- `SK_ID_CURR` is used only for joins and removed before fitting.
- The API returns a probability and risk band; final lending decisions require policy, fairness, compliance, and human governance.
- Raw Kaggle CSVs are not committed because of size and license considerations.
"""
    out_path = LOG_DIR / "model_card.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def train() -> dict:
    logger.info("Loading Home Credit training data from %s", RAW_DATA_DIR)
    validation_report_path = write_validation_report(RAW_DATA_DIR, LOG_DIR)
    logger.info("Data validation report saved to %s", validation_report_path)
    X, y = load_training_data(RAW_DATA_DIR)

    sample_rows = int(os.getenv("TRAIN_SAMPLE_ROWS", "0"))
    training_mode = "full_dataset"
    if sample_rows and sample_rows < len(X):
        sampled = X.assign(_target=y).sample(sample_rows, random_state=RANDOM_STATE)
        y = sampled.pop("_target")
        X = sampled
        training_mode = f"sampled_{sample_rows}_rows"
        logger.info("Using TRAIN_SAMPLE_ROWS=%s for a faster local run.", sample_rows)

    logger.info("Training rows: %s | default rate: %.2f%%", f"{len(X):,}", y.mean() * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = round(negative / max(positive, 1), 3)

    candidates = build_candidates(scale_pos_weight)
    results = {}

    for name, pipeline in candidates.items():
        logger.info("Fitting %s", name)
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        logger.info("%s ROC-AUC=%s", name, results[name]["roc_auc"])

    champion_name = max(results, key=lambda name: results[name]["roc_auc"])
    champion = candidates[champion_name]
    metrics = results[champion_name]
    logger.info("Champion: %s | ROC-AUC=%s", champion_name, metrics["roc_auc"])

    plot_score_distribution(champion, X_test, y_test, champion_name)
    plot_roc_pr_curves(champion, X_test, y_test, champion_name)
    feature_importance = compute_permutation_importance(champion, X_test, y_test)

    buffer = io.BytesIO()
    joblib.dump(champion, buffer)
    fingerprint = hashlib.sha256(buffer.getvalue()).hexdigest()

    model_path = MODEL_DIR / "credit_model.pkl"
    joblib.dump(champion, model_path)

    metadata = {
        "dataset": "Home Credit Default Risk",
        "training_mode": training_mode,
        "model_name": champion_name,
        "trained_at": datetime.utcnow().isoformat(),
        "sha256": fingerprint,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "default_rate": round(float(y.mean()), 5),
        "candidate_results": results,
        "test_metrics": metrics,
        "top_permutation_importance": feature_importance,
        "features": list(X.columns),
        "raw_data_dir": str(RAW_DATA_DIR),
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    model_card_path = write_model_card(metadata, feature_importance)
    logger.info("Model card saved to %s", model_card_path)

    logger.info("Saved model to %s", model_path)
    return metadata


if __name__ == "__main__":
    train()
