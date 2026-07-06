"""Input validation for experiment evaluations."""

from typing import Optional, Sequence

import numpy as np

from .constants import TaskType
from .exceptions import (
    DuplicateSampleError,
    InvalidTaskError,
    LengthMismatchError,
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
