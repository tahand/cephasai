"""Task-aware evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

# The metric used to rank models, and whether a larger value is better.
PRIMARY_METRIC = {"regression": "r2", "classification": "accuracy"}
HIGHER_IS_BETTER = {
    "r2": True,
    "adjusted_r2": True,
    "rmse": False,
    "mae": False,
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
}
# scikit-learn scorer names for cross-validation during tuning.
CV_SCORING = {"regression": "r2", "classification": "accuracy"}


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    r2 = skm.r2_score(y_true, y_pred) if n > 1 else float("nan")
    result = {
        "r2": r2,
        "rmse": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "mae": skm.mean_absolute_error(y_true, y_pred),
    }
    if n_features is not None and n - n_features - 1 > 0:
        result["adjusted_r2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {k: float(v) for k, v in result.items()}


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    result = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_proba.shape[1] == 2:
                result["roc_auc"] = skm.roc_auc_score(y_true, y_proba[:, 1])
            else:
                result["roc_auc"] = skm.roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            pass  # e.g. the test split is missing a class the model was trained on
    return {k: float(v) for k, v in result.items()}
