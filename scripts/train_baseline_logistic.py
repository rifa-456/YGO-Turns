from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from data.schema.constants import TARGET
from training.augmentation import swap_perspective
from training.config import PathConfig
from training.scalar_dataset import build_scalar_array
from training.turns import load_split

_PATHS = PathConfig()
_OUTPUT_DIR = _PATHS.logistic_model.parent


def _evaluate(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """Computes AUROC, Brier score, and accuracy for a fitted model on a split.

    Args:
        model: Fitted LogisticRegression instance.
        X: Float32 feature matrix of shape (N, scalar_dim).
        y: Binary target array of shape (N,).
        split_name: Human-readable label printed in the results table.

    Returns:
        Dict with keys 'auroc', 'brier', 'accuracy'.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auroc": float(roc_auc_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
        "accuracy": float((preds == y).mean()),
    }

    print(f"\n[{split_name}]")
    print(f"  AUROC    : {metrics['auroc']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")

    return metrics


def train() -> None:
    """Runs the logistic regression baseline training and evaluation pipeline.

    Fits a logistic regression model on the scalar feature set, prints train
    and validation metrics, then saves the fitted model and metrics to disk.
    """

    train_df = load_split("train")
    val_df = load_split("validation")
    train_df = pd.concat([train_df, swap_perspective(train_df)], ignore_index=True)

    print(f"Train rows : {len(train_df):,}")
    print(f"Val rows   : {len(val_df):,}")

    X_train = build_scalar_array(train_df)
    y_train = train_df[TARGET].astype(np.float32).values

    X_val = build_scalar_array(val_df)
    y_val = val_df[TARGET].astype(np.float32).values

    print("\nFitting logistic regression …")
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Done.")

    train_metrics = _evaluate(model, X_train, y_train, "train")
    val_metrics = _evaluate(model, X_val, y_val, "val")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = _PATHS.logistic_model
    joblib.dump(model, model_path)
    print(f"\nModel saved → {model_path}")

    metrics_path = _OUTPUT_DIR / "scalar_logistic_metrics.json"
    metrics_path.write_text(
        json.dumps({"train": train_metrics, "val": val_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    train()
