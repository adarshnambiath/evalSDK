"""Serialization utilities for writing experiment outputs to disk."""

import json
from pathlib import Path
from typing import Any, List

import pandas as pd

from .schemas import ArtifactRecord


def write_evaluation_table(path: Path, table: pd.DataFrame) -> None:
    """Write an evaluation table to parquet.

    Args:
        path: Destination file path.
        table: pandas DataFrame to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def write_metrics(path: Path, metrics: Any) -> None:
    """Write metrics to JSON.

    Args:
        path: Destination file path.
        metrics: JSON-serializable metrics dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def write_artifacts(path: Path, artifacts: List[ArtifactRecord]) -> None:
    """Write artifact metadata to JSON.

    Args:
        path: Destination file path.
        artifacts: List of ArtifactRecord instances.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "type": artifact.type.value,
            "path": artifact.path,
            "name": artifact.name,
            "timestamp": artifact.timestamp,
        }
        for artifact in artifacts
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
