"""Tests for publish preview and confirmation mode."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from experiment_sdk import ExperimentSession
from experiment_sdk.artifacts import ArtifactRegistry
from experiment_sdk.preview import OutputEstimator


def test_preview_does_not_write_files(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    session.preview()

    assert not (tmp_path / "evaluation.parquet").exists()
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "artifacts.json").exists()


def test_preview_output_format(tmp_path, capsys):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    session.preview()

    captured = capsys.readouterr()
    output = captured.out

    assert "Experiment Publish Preview" in output
    assert "Output directory:" in output
    assert str(tmp_path) in output
    assert "Evaluations:" in output
    assert "1" in output
    assert "Rows:" in output
    assert "2" in output
    assert "Metrics:" in output
    assert "accuracy" in output
    assert "Artifacts:" in output
    assert "0" in output
    assert "Files to be created:" in output
    assert "evaluation.parquet" in output
    assert "metrics.json" in output
    assert "artifacts.json" in output
    assert "Estimated total:" in output


def test_preview_shows_registered_artifact(tmp_path, capsys):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a"],
        ground_truth=[0],
        predictions=[0],
    )
    session.publish_artifact(
        type="MODEL_CHECKPOINT",
        path="model.pt",
        name="my_model",
    )

    session.preview()

    captured = capsys.readouterr()
    output = captured.out

    assert "Artifacts:" in output
    assert "1" in output


def test_preview_metric_keys_displayed(tmp_path, capsys):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    session.preview()

    captured = capsys.readouterr()
    output = captured.out

    assert "accuracy" in output
    assert "f1" in output
    assert "precision" in output
    assert "recall" in output


def test_finish_default_writes_files(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    session.finish()

    assert (tmp_path / "evaluation.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "artifacts.json").exists()


def test_finish_confirm_no_cancels(tmp_path, capsys):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    with patch("builtins.input", return_value="n"):
        session.finish(confirm=True)

    captured = capsys.readouterr()
    assert "Serialization cancelled." in captured.out
    assert not (tmp_path / "evaluation.parquet").exists()
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "artifacts.json").exists()


def test_finish_confirm_empty_input_accepts(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a"],
        ground_truth=[0],
        predictions=[0],
    )

    with patch("builtins.input", return_value=""):
        session.finish(confirm=True)

    assert (tmp_path / "evaluation.parquet").exists()


def test_finish_confirm_case_insensitive_no(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a"],
        ground_truth=[0],
        predictions=[0],
    )

    with patch("builtins.input", return_value="No"):
        session.finish(confirm=True)

    assert not (tmp_path / "evaluation.parquet").exists()


def test_output_estimator_calculates_parquet_size(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b", "c"],
        ground_truth=[0, 1, 0],
        predictions=[0, 1, 0],
    )

    evaluation = session._evaluations[0]
    memory_usage = evaluation.evaluation_table.memory_usage(deep=True).sum()

    estimator = OutputEstimator(
        evaluations=session._evaluations,
        artifacts=session.artifact_registry,
        output_directory=session.output_directory,
    )
    plan = estimator.build_plan()

    assert plan["estimated_sizes"]["evaluation.parquet"] == memory_usage
    assert plan["total_size"] == sum(plan["estimated_sizes"].values())


def test_output_estimator_artifacts_size(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a"],
        ground_truth=[0],
        predictions=[0],
    )
    session.publish_artifact(
        type="MODEL_CHECKPOINT",
        path="model.pt",
        name="test",
    )

    estimator = OutputEstimator(
        evaluations=session._evaluations,
        artifacts=session.artifact_registry,
        output_directory=session.output_directory,
    )
    plan = estimator.build_plan()

    assert plan["artifacts"] == 1
    assert plan["estimated_sizes"]["artifacts.json"] > 0


def test_output_estimator_format_size(tmp_path):
    estimator = OutputEstimator(
        evaluations=[],
        artifacts=ArtifactRegistry(),
        output_directory=tmp_path,
    )

    assert estimator._format_size(500) == "500 B"
    assert estimator._format_size(2048) == "2.0 KB"
    assert estimator._format_size(2 * 1024 * 1024 + 500) == "2.0 MB"


def test_finish_confirm_yes_writes_files(tmp_path):
    session = ExperimentSession(output_directory=str(tmp_path))
    session.publish_evaluation(
        task="classification",
        sample_ids=["a", "b"],
        ground_truth=[0, 1],
        predictions=[0, 1],
    )

    with patch("builtins.input", return_value="y"):
        session.finish(confirm=True)

    assert (tmp_path / "evaluation.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "artifacts.json").exists()