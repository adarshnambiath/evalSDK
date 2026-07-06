"""Tests for classification evaluation."""

import pandas as pd
import pytest

from experiment_sdk import ExperimentSession
from experiment_sdk.constants import TaskType


def test_classification_basic(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    sample_ids = ["a", "b", "c", "d", "e"]
    ground_truth = [0, 1, 0, 1, 0]
    predictions = [0, 1, 0, 0, 1]
    probabilities = [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.3, 0.7],
    ]

    evaluation = session.publish_evaluation(
        task="classification",
        sample_ids=sample_ids,
        ground_truth=ground_truth,
        predictions=predictions,
        probabilities=probabilities,
    )

    assert evaluation.task == TaskType.CLASSIFICATION
    assert len(evaluation.evaluation_table) == 5

    expected_columns = {"sample_id", "ground_truth", "prediction", "correct", "confidence"}
    assert expected_columns.issubset(set(evaluation.evaluation_table.columns))

    # confidence corresponds to predicted class max probability
    assert evaluation.evaluation_table.loc[0, "confidence"] == pytest.approx(0.9)
    assert evaluation.evaluation_table.loc[1, "confidence"] == pytest.approx(0.8)

    session.finish()

    assert (tmp_path / "evaluation.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "artifacts.json").exists()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert len(df) == 5
    assert "evaluation_index" in df.columns
    assert list(df["evaluation_index"]) == [0] * 5


def test_classification_without_probabilities(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="classification",
        sample_ids=["x", "y", "z"],
        ground_truth=[1, 1, 0],
        predictions=[1, 0, 0],
    )

    evaluation = session._evaluations[0]
    assert "confidence" in evaluation.evaluation_table.columns
    assert evaluation.evaluation_table["confidence"].isna().all()

    session.finish()

    with open(tmp_path / "metrics.json") as f:
        import json

        metrics = json.load(f)

    assert "accuracy" in metrics["evaluation_0"]
    assert "confusion_matrix" in metrics["evaluation_0"]


def test_multiple_evaluations_in_session(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    session.publish_evaluation(
        task="regression",
        sample_ids=["c", "d"],
        ground_truth=[1.0, 2.0],
        predictions=[1.1, 1.9],
    )

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert len(df) == 4
    assert set(df["evaluation_index"]) == {0, 1}
