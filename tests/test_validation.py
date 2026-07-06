"""Tests for input validation."""

import pytest

from experiment_sdk.constants import TaskType
from experiment_sdk.exceptions import (
    DuplicateSampleError,
    InvalidTaskError,
    LengthMismatchError,
    ValidationError,
)
from experiment_sdk.validation import validate_inputs


def test_valid_classification_inputs():
    validate_inputs(
        task=TaskType.CLASSIFICATION,
        sample_ids=["a", "b", "c"],
        ground_truth=[0, 1, 0],
        predictions=[0, 1, 1],
    )


def test_valid_regression_inputs():
    validate_inputs(
        task=TaskType.REGRESSION,
        sample_ids=["a", "b"],
        ground_truth=[1.0, 2.0],
        predictions=[1.1, 2.2],
    )


def test_invalid_task_raises():
    with pytest.raises(InvalidTaskError):
        validate_inputs(
            task="unknown",
            sample_ids=[1],
            ground_truth=[0],
            predictions=[1],
        )


def test_empty_arrays_raise():
    with pytest.raises(ValidationError):
        validate_inputs(
            task=TaskType.CLASSIFICATION,
            sample_ids=[],
            ground_truth=[],
            predictions=[],
        )


def test_length_mismatch_ground_truth():
    with pytest.raises(LengthMismatchError):
        validate_inputs(
            task=TaskType.CLASSIFICATION,
            sample_ids=[1, 2],
            ground_truth=[0, 1, 0],
            predictions=[0, 1, 1],
        )


def test_length_mismatch_predictions():
    with pytest.raises(LengthMismatchError):
        validate_inputs(
            task=TaskType.CLASSIFICATION,
            sample_ids=[1, 2, 3],
            ground_truth=[0, 1, 0],
            predictions=[0, 1],
        )


def test_duplicate_sample_ids_raise():
    with pytest.raises(DuplicateSampleError):
        validate_inputs(
            task=TaskType.CLASSIFICATION,
            sample_ids=[1, 1, 2],
            ground_truth=[0, 1, 0],
            predictions=[0, 1, 1],
        )


def test_probability_length_mismatch():
    with pytest.raises(LengthMismatchError):
        validate_inputs(
            task=TaskType.CLASSIFICATION,
            sample_ids=[1, 2],
            ground_truth=[0, 1],
            predictions=[0, 1],
            probabilities=[[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
        )


def test_valid_with_probabilities():
    validate_inputs(
        task=TaskType.CLASSIFICATION,
        sample_ids=[1, 2],
        ground_truth=[0, 1],
        predictions=[0, 1],
        probabilities=[[0.9, 0.1], [0.2, 0.8]],
    )
