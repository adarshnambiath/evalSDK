"""Tests for standard metric computation."""

import math

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from experiment_sdk.constants import TaskType
from experiment_sdk.metrics import compute_metrics


def test_classification_accuracy():
    metrics = compute_metrics(
        TaskType.CLASSIFICATION,
        np.array([0, 1, 0, 1, 0]),
        np.array([0, 1, 0, 0, 1]),
    )

    assert metrics["accuracy"] == pytest.approx(0.6)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics

    cm = metrics["confusion_matrix"]
    assert "labels" in cm
    assert "matrix" in cm
    assert cm["labels"] == ["0", "1"]
    matrix = np.array(cm["matrix"])
    assert matrix.shape == (2, 2)


def test_classification_with_probabilities():
    metrics = compute_metrics(
        TaskType.CLASSIFICATION,
        np.array([0, 1, 0, 1, 0]),
        np.array([0, 1, 0, 0, 1]),
        probabilities=np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
        ]),
    )

    assert "average_confidence" in metrics
    expected = np.mean([0.9, 0.8, 0.8, 0.7, 0.6])
    assert metrics["average_confidence"] == pytest.approx(expected)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.2, 2.8])

    metrics = compute_metrics(TaskType.REGRESSION, y_true, y_pred)

    assert "mae" in metrics
    assert metrics["mae"] == pytest.approx(0.16666666)

    assert "rmse" in metrics
    expected_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    assert metrics["rmse"] == pytest.approx(expected_rmse)

    assert "r2" in metrics


def test_regression_perfect_predictions():
    metrics = compute_metrics(
        TaskType.REGRESSION,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
    )

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_negative_residuals():
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([2.0, 3.0])

    metrics = compute_metrics(TaskType.REGRESSION, y_true, y_pred)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)


def test_multiclass_classification():
    metrics = compute_metrics(
        TaskType.CLASSIFICATION,
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([0, 2, 1, 0, 1, 2]),
    )

    assert metrics["accuracy"] == pytest.approx(4 / 6)

    cm = metrics["confusion_matrix"]
    assert "labels" in cm
    assert "matrix" in cm
    assert cm["labels"] == ["0", "1", "2"]

    matrix = np.array(cm["matrix"])
    assert matrix.shape == (3, 3)
