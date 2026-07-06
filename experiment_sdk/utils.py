"""Internal utility functions."""

from typing import Any, List


def ensure_list(value: Any) -> List:
    """Coerce a value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)
