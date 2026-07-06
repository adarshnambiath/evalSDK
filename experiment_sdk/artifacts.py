"""Artifact registration and management."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .constants import ArtifactType


@dataclass
class ArtifactRecord:
    """Metadata record for a registered artifact."""

    type: ArtifactType
    path: str
    name: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


class ArtifactRegistry:
    """In-memory artifact registry."""

    def __init__(self) -> None:
        self._artifacts: List[ArtifactRecord] = []

    def register(
        self,
        artifact_type: ArtifactType,
        path: str,
        name: Optional[str] = None,
    ) -> ArtifactRecord:
        """Register a new artifact.

        Args:
            artifact_type: The category of artifact.
            path: User-provided path to the artifact file.
            name: Optional human-readable name.

        Returns:
            The created ArtifactRecord.
        """
        record = ArtifactRecord(type=artifact_type, path=path, name=name)
        self._artifacts.append(record)
        return record

    def list(self) -> List[ArtifactRecord]:
        """Return a shallow copy of all registered artifacts."""
        return list(self._artifacts)
