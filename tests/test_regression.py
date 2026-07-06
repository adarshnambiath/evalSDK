"""Tests for regression evaluation."""

import math

import pandas as pd
import pytest

from experiment_sdk import ExperimentSession


def test_regression_evaluation(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    sample_ids = ["a", "b", "c"]
    ground_truth = [1.0, 2.0, 3.0]
    predictions = [1.1, 2.2, 2.8]

    evaluation = session.publish_evaluation(
        task="regression",
        sample_ids=sample_ids,
        ground_truth=ground_truth,
        predictions=predictions,
    )

    assert len(evaluation.evaluation_table) == 3
    expected_columns = {"sample_id", "ground_truth", "prediction", "residual", "absolute_error"}
    assert expected_columns.issubset(set(evaluation.evaluation_table.columns))

    # row 0: residual = 0.1, absolute_error = 0.1
    assert evaluation.evaluation_table.loc[0, "residual"] == pytest.approx(0.1)
    assert evaluation.evaluation_table.loc[0, "absolute_error"] == pytest.approx(0.1)

    session.finish()
    assert (tmp_path / "evaluation.parquet").exists()


def test_regression_perfect_predictions(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="regression",
        sample_ids=["x", "y"],
        ground_truth=[0.0, 5.0],
        predictions=[0.0, 5.0],
    )

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert (df["residual"] == 0.0).all()
    assert (df["absolute_error"] == 0.0).all()


def test_regression_metrics_computed(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="regression",
        sample_ids=["a", "b"],
        ground_truth=[1.0, 2.0],
        predictions=[1.0, 2.0],
    )

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert (df["absolute_error"] == 0.0).all()

    import json

    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    assert math.isclose(metrics["evaluation_0"]["mae"], 0.0)
    assert math.isclose(metrics["evaluation_0"]["rmse"], 0.0)
    assert math.isclose(metrics["evaluation_0"]["r2"], 1.0)
