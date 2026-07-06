"""Constants used throughout the SDK."""

from enum import Enum


class TaskType(str, Enum):
    """Supported ML experiment tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ArtifactType(str, Enum):
    """Supported artifact types."""

    MODEL_CHECKPOINT = "MODEL_CHECKPOINT"
    CONFIG = "CONFIG"
    LOGS = "LOGS"
    REPORT = "REPORT"
    VISUALIZATION = "VISUALIZATION"
    PREDICTIONS = "PREDICTIONS"
