"""Evaluation construction for individual experiment runs."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import TaskType
from .metrics import compute_metrics
from .schemas import EvaluationRecord
from .validation import validate_inputs


class Evaluation:
    """Validates inputs, builds the evaluation table, and computes metrics."""

    def __init__(
        self,
        task: TaskType,
        sample_ids: list,
        ground_truth: list,
        predictions: list,
        probabilities: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        validate_inputs(
            task=task,
            sample_ids=sample_ids,
            ground_truth=ground_truth,
            predictions=predictions,
            probabilities=probabilities,
        )

        self.sample_ids = np.asarray(sample_ids)
        self.ground_truth = np.asarray(ground_truth)
        self.predictions = np.asarray(predictions)
        self.probabilities = (
            np.asarray(probabilities) if probabilities is not None else None
        )
        self.metadata = metadata or {}
        self.task = task

        self.evaluation_table = self._build_table()
        self.metrics = compute_metrics(
            task=task,
            ground_truth=self.ground_truth,
            predictions=self.predictions,
            probabilities=self.probabilities,
        )

    def _build_table(self) -> pd.DataFrame:
        if self.task == TaskType.CLASSIFICATION:
            return self._build_classification_table()
        if self.task == TaskType.REGRESSION:
            return self._build_regression_table()
        raise ValueError(f"Unsupported task: {self.task}")

    def _build_classification_table(self) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                "sample_id": self.sample_ids,
                "ground_truth": self.ground_truth,
                "prediction": self.predictions,
                "correct": self.ground_truth == self.predictions,
            }
        )

        if self.probabilities is not None:
            table["confidence"] = np.max(self.probabilities, axis=1)
        else:
            table["confidence"] = np.nan

        return table

    def _build_regression_table(self) -> pd.DataFrame:
        residuals = self.predictions - self.ground_truth
        return pd.DataFrame(
            {
                "sample_id": self.sample_ids,
                "ground_truth": self.ground_truth,
                "prediction": self.predictions,
                "residual": residuals,
                "absolute_error": np.abs(residuals),
            }
        )

    def to_record(self) -> EvaluationRecord:
        """Return an EvaluationRecord snapshot."""
        return EvaluationRecord(
            task=self.task,
            evaluation_table=self.evaluation_table,
            metrics=self.metrics,
            metadata=self.metadata,
        )
