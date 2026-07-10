"""Publish preview and output size estimation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .estimation import estimate_dataframe_parquet_size


class OutputEstimator:
    """Estimates output sizes and builds publish plans without writing files."""

    def __init__(
        self,
        evaluations: List[Any],
        artifacts: Any,
        output_directory: Path,
    ) -> None:
        self._evaluations = evaluations
        self._artifacts = artifacts
        self._output_directory = Path(output_directory)

    def _estimate_raw_parquet_size(self) -> int:
        total = 0
        for evaluation in self._evaluations:
            total += evaluation.evaluation_table.memory_usage(deep=True).sum()
        return int(total)

    def _estimate_parquet_size(self) -> int:
        total = 0
        for evaluation in self._evaluations:
            total += estimate_dataframe_parquet_size(
                evaluation.evaluation_table
            )
        return int(total)

    def _estimate_metrics_size(self) -> int:
        if not self._evaluations:
            return 0
        if len(self._evaluations) == 1:
            metrics = self._evaluations[0].metrics
        else:
            metrics = {
                f"evaluation_{i}": evaluation.metrics
                for i, evaluation in enumerate(self._evaluations)
            }
        return len(json.dumps(metrics, indent=2).encode("utf-8")) + 1

    def _estimate_artifacts_size(self) -> int:
        payload = [
            {
                "type": artifact.type.value,
                "path": artifact.path,
                "name": artifact.name,
                "timestamp": artifact.timestamp,
            }
            for artifact in self._artifacts.list()
        ]
        return len(json.dumps(payload, indent=2).encode("utf-8")) + 1

    def build_plan(self) -> Dict[str, Any]:
        """Build a publish plan without writing files.

        Returns:
            A dictionary summarizing the planned outputs and estimated sizes.
        """
        parquet_size = self._estimate_parquet_size()
        raw_parquet_size = self._estimate_raw_parquet_size()
        metrics_size = self._estimate_metrics_size()
        artifacts_size = self._estimate_artifacts_size()
        total_size = parquet_size + metrics_size + artifacts_size

        metric_keys: List[str] = []
        for evaluation in self._evaluations:
            metric_keys.extend(evaluation.metrics.keys())

        return {
            "evaluations": len(self._evaluations),
            "row_count": sum(
                len(evaluation.evaluation_table) for evaluation in self._evaluations
            ),
            "metrics": metric_keys,
            "artifacts": len(self._artifacts.list()),
            "output_directory": str(self._output_directory),
            "estimated_sizes": {
                "evaluation.parquet": parquet_size,
                "metrics.json": metrics_size,
                "artifacts.json": artifacts_size,
            },
            "raw_estimated_sizes": {
                "evaluation.parquet": raw_parquet_size,
            },
            "total_size": total_size,
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Return a human-readable file size string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def print_preview(self, plan: Optional[Dict[str, Any]] = None) -> None:
        """Print the publish preview to stdout.

        Args:
            plan: Optional pre-built plan dictionary. If omitted, one will
                be generated via ``build_plan()``.
        """
        plan = plan if plan is not None else self.build_plan()

        print("-" * 60)
        print()
        print("Experiment Publish Preview")
        print()
        print("Output directory:")
        print(plan["output_directory"])
        print()
        print("Evaluations:")
        print(plan["evaluations"])
        print()
        print("Rows:")
        print(f"{plan['row_count']:,}")
        print()
        print("Metrics:")
        for metric in plan["metrics"]:
            print(metric)
        print()
        print("Artifacts:")
        print(plan["artifacts"])
        print()
        print("Files to be created:")
        print()
        for filename, size in plan["estimated_sizes"].items():
            print(filename)
            if filename == "evaluation.parquet" and "evaluation.parquet" in plan.get("raw_estimated_sizes", {}):
                raw_size = plan["raw_estimated_sizes"]["evaluation.parquet"]
                print(f"Estimated (parquet): ~{self._format_size(size)}")
                print(f"Raw estimate:        ~{self._format_size(raw_size)}")
            else:
                print(f"~{self._format_size(size)}")
            print()
        print("Estimated total:")
        print(self._format_size(plan["total_size"]))
        print()
