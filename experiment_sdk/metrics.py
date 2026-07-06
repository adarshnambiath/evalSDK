"""Standard metric computation for supported task types."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .constants import TaskType
from .exceptions import InvalidTaskError


def compute_metrics(
    task: TaskType,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict:
    """Compute standard metrics for the given task.

    Args:
        task: The ML task type.
        ground_truth: True labels or values.
        predictions: Predicted labels or values.
        probabilities: Optional predicted probabilities (classification only).

    Returns:
        Dictionary of metric names to scalar values (and confusion matrix as list).
    """
    if task == TaskType.CLASSIFICATION:
        return _compute_classification_metrics(ground_truth, predictions, probabilities)
    if task == TaskType.REGRESSION:
        return _compute_regression_metrics(ground_truth, predictions)
    raise InvalidTaskError(f"Unsupported task type: {task}")


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict:
    unique = np.unique(y_true)
    average = "binary" if len(unique) == 2 else "macro"

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "matrix": cm.tolist(),
        },
    }

    if probabilities is not None:
        metrics["average_confidence"] = float(np.mean(np.max(probabilities, axis=1)))

    return metrics


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
