"""Tests for per-sample columns feature."""

import json

import pandas as pd
import pytest

from experiment_sdk import ExperimentSession
from experiment_sdk.exceptions import (
    LengthMismatchError,
    ReservedColumnError,
    ValidationError,
)


def test_extra_columns_classification(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    evaluation = session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b", "c"],
        ground_truth=[0, 1, 0],
        predictions=[0, 1, 1],
        probabilities=[
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        columns={
            "window_start": [0, 10, 20],
            "window_end": [10, 20, 30],
            "record_name": ["rec_1", "rec_2", "rec_3"],
        },
    )

    table = evaluation.evaluation_table
    assert "window_start" in table.columns
    assert "window_end" in table.columns
    assert "record_name" in table.columns
    assert list(table["window_start"]) == [0, 10, 20]
    assert list(table["window_end"]) == [10, 20, 30]
    assert list(table["record_name"]) == ["rec_1", "rec_2", "rec_3"]

    # Standard columns still present
    assert "sample_id" in table.columns
    assert "ground_truth" in table.columns

    session.finish()

    # Verify parquet contains extra columns
    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert "window_start" in df.columns
    assert "window_end" in df.columns
    assert "record_name" in df.columns


def test_extra_columns_regression(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    evaluation = session.publish_evaluation(
        task="regression",
        sample_ids=["a", "b", "c"],
        ground_truth=[1.0, 2.0, 3.0],
        predictions=[1.1, 2.2, 2.8],
        columns={
            "timestamp": [100, 200, 300],
            "source": ["alpha", "beta", "gamma"],
        },
    )

    table = evaluation.evaluation_table
    assert "timestamp" in table.columns
    assert "source" in table.columns
    assert list(table["timestamp"]) == [100, 200, 300]
    assert list(table["source"]) == ["alpha", "beta", "gamma"]

    # Standard columns still present
    assert "residual" in table.columns
    assert "absolute_error" in table.columns


def test_extra_columns_serialized_in_parquet(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="classification",
        sample_ids=["x", "y"],
        ground_truth=[0, 1],
        predictions=[0, 1],
        columns={
            "token_span": [(0, 5), (6, 10)],
            "bounding_box": [[10, 20, 30, 40], [50, 60, 70, 80]],
        },
    )

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert "token_span" in df.columns
    assert "bounding_box" in df.columns
    assert len(df) == 2

    # metrics.json remains unchanged in structure (flat, not nested under evaluation_0)
    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1" in metrics

    # artifacts.json remains unchanged
    with open(tmp_path / "artifacts.json") as f:
        artifacts = json.load(f)
    assert isinstance(artifacts, list)


def test_columns_length_mismatch():
    session = ExperimentSession(output_directory="/tmp/unused")

    with pytest.raises(LengthMismatchError):
        session.publish_evaluation(
            task="classification",
            sample_ids=["a", "b", "c"],
            ground_truth=[0, 1, 0],
            predictions=[0, 1, 1],
            columns={"window_start": [0, 10]},
        )


def test_reserved_column_name_raises():
    session = ExperimentSession(output_directory="/tmp/unused")

    reserved_names = [
        "sample_id",
        "ground_truth",
        "prediction",
        "confidence",
        "correct",
        "residual",
        "absolute_error",
        "evaluation_index",
    ]

    for name in reserved_names:
        with pytest.raises(ReservedColumnError):
            session.publish_evaluation(
                task="classification",
                sample_ids=["a", "b"],
                ground_truth=[0, 1],
                predictions=[0, 1],
                columns={name: [1, 2]},
            )


def test_column_value_not_list_like():
    session = ExperimentSession(output_directory="/tmp/unused")

    with pytest.raises(ValidationError):
        session.publish_evaluation(
            task="classification",
            sample_ids=["a", "b"],
            ground_truth=[0, 1],
            predictions=[0, 1],
            columns={"bad_column": 42},
        )


def test_columns_optional_does_not_break_existing(tmp_path):
    """Existing callers that do not pass 'columns' must work unchanged."""
    session = ExperimentSession(output_directory=str(tmp_path))

    evaluation = session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b", "c"],
        ground_truth=[0, 1, 0],
        predictions=[0, 1, 1],
    )

    assert "sample_id" in evaluation.evaluation_table.columns
    assert "ground_truth" in evaluation.evaluation_table.columns
    assert "correct" in evaluation.evaluation_table.columns

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert len(df) == 3


def test_columns_preserves_order(tmp_path):
    """Extra columns should maintain the supplied iteration order."""
    session = ExperimentSession(output_directory=str(tmp_path))

    evaluation = session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
        columns={
            "z_last": [1, 2],
            "a_first": [3, 4],
            "m_mid": [5, 6],
        },
    )

    cols = list(evaluation.evaluation_table.columns)
    std = {"sample_id", "ground_truth", "prediction", "correct", "confidence"}
    extra_start = next(i for i, c in enumerate(cols) if c not in std)
    extra_cols = cols[extra_start:]
    assert extra_cols == ["z_last", "a_first", "m_mid"]


def test_columns_empty_dict(tmp_path):
    """An empty dict for columns should be a no-op."""
    session = ExperimentSession(output_directory=str(tmp_path))

    evaluation = session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
        columns={},
    )

    session.finish()

    df = pd.read_parquet(tmp_path / "evaluation.parquet")
    assert "sample_id" in df.columns
    # sample_id, ground_truth, prediction, correct, confidence
    assert len(df.columns) == 5


def test_columns_metrics_unchanged(tmp_path):
    """Metrics computation must be unaffected by extra columns."""
    session = ExperimentSession(output_directory=str(tmp_path))

    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b", "c", "d", "e"],
        ground_truth=[0, 1, 0, 1, 0],
        predictions=[0, 1, 0, 0, 1],
        columns={"window_start": [0, 10, 20, 30, 40]},
    )

    session.finish()

    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    assert metrics["accuracy"] == 0.6
    assert metrics["f1"] == 0.5
