"""The primary public interface for experiment output management."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .artifacts import ArtifactRegistry
from .constants import ArtifactType, TaskType
from .evaluation import Evaluation
from .io import write_artifacts, write_evaluation_table, write_metrics
from .preview import OutputEstimator
from .schemas import EvaluationRecord
from .utils import ensure_list


class ExperimentSession:
    """Standardizes and serializes a single experiment run.

    Orchestrates validation, evaluation registration, artifact registration,
    and final serialization to disk.
    """

    def __init__(self, output_directory: str) -> None:
        """Initialize a new session.

        Args:
            output_directory: Directory where output files will be written.
        """
        self.output_directory = Path(output_directory)
        self._evaluations: List[Evaluation] = []
        self.artifact_registry = ArtifactRegistry()

    def publish_evaluation(
        self,
        task: str,
        sample_ids: list,
        ground_truth: list,
        predictions: list,
        probabilities: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        columns: Optional[Dict[str, list]] = None,
    ) -> Evaluation:
        """Validate inputs, build an evaluation, and store it.

        Args:
            task: ML task type, e.g. ``"classification"`` or ``"regression"``.
            sample_ids: Unique identifiers for each sample.
            ground_truth: Ground-truth labels or values.
            predictions: Predicted labels or values.
            probabilities: Optional predicted probabilities (classification).
            metadata: Optional metadata dictionary.
            columns: Optional dictionary of additional per-sample columns.
                Each value must be a list-like with the same length as
                ``sample_ids``. Column names must not collide with
                SDK-owned columns (``sample_id``, ``ground_truth``,
                ``prediction``, ``confidence``, ``correct``, ``residual``,
                ``absolute_error``, ``evaluation_index``).

        Returns:
            The created Evaluation instance.
        """
        evaluation = Evaluation(
            task=TaskType(task.lower()),
            sample_ids=ensure_list(sample_ids),
            ground_truth=ensure_list(ground_truth),
            predictions=ensure_list(predictions),
            probabilities=ensure_list(probabilities) if probabilities is not None else None,
            metadata=metadata,
            columns=columns,
        )
        self._evaluations.append(evaluation)
        return evaluation

    def publish_artifact(
        self,
        type: str,
        path: str,
        name: Optional[str] = None,
    ) -> None:
        """Register an artifact without copying or uploading it.

        Args:
            type: Artifact type, e.g. ``"MODEL_CHECKPOINT"``.
            path: User-provided path to the artifact file.
            name: Optional human-readable name.
        """
        self.artifact_registry.register(
            artifact_type=ArtifactType(type),
            path=path,
            name=name,
        )

    def preview(self) -> None:
        """Print a summary of outputs without writing anything."""

        if not self._evaluations:
            raise ValueError("No evaluation has been published.")

        if len(self._evaluations) > 1:
            raise ValueError(
                "ExperimentSession currently supports exactly one evaluation."
            )

        estimator = OutputEstimator(
            evaluations=self._evaluations,
            artifacts=self.artifact_registry,
            output_directory=self.output_directory,
        )
        plan = estimator.build_plan()
        estimator.print_preview(plan)

    def finish(self, confirm: bool = True) -> None:
        """Serialize the experiment outputs to disk.

        Args:
            confirm: If True, print a preview and ask for confirmation before
                writing any files. If False (default), write files immediately.
        """

        if not self._evaluations:
            raise ValueError("No evaluation has been published.")

        if len(self._evaluations) > 1:
            raise ValueError(
                "ExperimentSession currently supports exactly one evaluation."
            )

        estimator = OutputEstimator(
            evaluations=self._evaluations,
            artifacts=self.artifact_registry,
            output_directory=self.output_directory,
        )
        plan = estimator.build_plan()

        if confirm:
            estimator.print_preview(plan)
            response = input("Proceed with serialization? [Y/n] ").strip().lower()
            if response not in ("", "y", "yes"):
                print("Serialization cancelled.")
                return

        evaluation = self._evaluations[0]

        write_evaluation_table(
            self.output_directory / "evaluation.parquet",
            evaluation.evaluation_table,
        )

        write_metrics(
            self.output_directory / "metrics.json",
            evaluation.metrics,
        )

        write_artifacts(
            self.output_directory / "artifacts.json",
            self.artifact_registry.list(),
        )