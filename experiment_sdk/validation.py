"""Input validation for experiment evaluations."""

from typing import Optional, Sequence

import numpy as np

from .constants import TaskType
from .exceptions import (
    DuplicateSampleError,
    InvalidTaskError,
    LengthMismatchError,
    ReservedColumnError,
    ValidationError,
)


def validate_inputs(
    task: TaskType,
    sample_ids: Sequence,
    ground_truth: Sequence,
    predictions: Sequence,
    probabilities: Optional[Sequence] = None,
) -> None:
    """Validate that inputs meet the requirements for the given task.

    Args:
        task: The type of ML task.
        sample_ids: Unique identifiers for each sample.
        ground_truth: Ground-truth labels or values.
        predictions: Predicted labels or values.
        probabilities: Optional predicted probabilities.

    Raises:
        InvalidTaskError: If the task type is not supported.
        LengthMismatchError: If arrays do not have the same length.
        DuplicateSampleError: If sample_ids contains duplicates.
        ValidationError: If arrays are empty or probabilities length mismatches.
    """
    if task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
        raise InvalidTaskError(f"Unsupported task type: {task}")

    sample_ids = np.asarray(sample_ids)
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)

    n = len(sample_ids)
    if n == 0:
        raise ValidationError("Input arrays must not be empty.")

    if len(ground_truth) != n or len(predictions) != n:
        raise LengthMismatchError(
            f"Length mismatch: sample_ids={n}, "
            f"ground_truth={len(ground_truth)}, predictions={len(predictions)}"
        )

    if len(np.unique(sample_ids)) != n:
        raise DuplicateSampleError("sample_ids must be unique.")

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if len(probabilities) != n:
            raise LengthMismatchError(
                f"Length mismatch: probabilities has {len(probabilities)} rows "
                f"but sample_ids has {n} entries."
            )


_RESERVED_COLUMNS: frozenset = frozenset({
    "sample_id",
    "ground_truth",
    "prediction",
    "confidence",
    "correct",
    "residual",
    "absolute_error",
    "evaluation_index",
})


def validate_columns(
    columns: Optional[dict],
    sample_ids: Sequence,
) -> None:
    """Validate user-supplied per-sample columns.

    Args:
        columns: Optional dictionary mapping column names to per-sample values.
        sample_ids: The sample identifiers used to check length.

    Raises:
        ValidationError: If a column value is not a list-like.
        LengthMismatchError: If a column does not have the same length as sample_ids.
        ReservedColumnError: If a column name collides with an SDK-owned column.
    """
    if columns is None:
        return

    n = len(sample_ids)

    for name, values in columns.items():
        if not hasattr(values, "__len__"):
            raise ValidationError(
                f"Column '{name}' must be a list-like; got {type(values).__name__}."
            )
        if len(values) != n:
            raise LengthMismatchError(
                f"Column '{name}' has {len(values)} entries but "
                f"sample_ids has {n} entries."
            )
        if name in _RESERVED_COLUMNS:
            raise ReservedColumnError(
                f"Column name '{name}' is reserved by the SDK. "
                f"Choose a different name."
            )
