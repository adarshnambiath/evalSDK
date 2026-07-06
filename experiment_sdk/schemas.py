"""Data schemas for experiment evaluations and artifacts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .constants import ArtifactType, TaskType


@dataclass
class ArtifactRecord:
    """Metadata record for a registered artifact."""

    type: ArtifactType
    path: str
    name: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class EvaluationRecord:
    """Container for a single completed evaluation."""

    task: TaskType
    evaluation_table: pd.DataFrame
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
