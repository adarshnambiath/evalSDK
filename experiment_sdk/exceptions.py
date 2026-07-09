"""Custom exceptions for the experiment SDK."""


class ExperimentSDKError(Exception):
    """Base exception for all experiment_sdk errors."""


class ValidationError(ExperimentSDKError):
    """Raised when input validation fails."""


class LengthMismatchError(ValidationError):
    """Raised when input arrays have different lengths."""


class DuplicateSampleError(ValidationError):
    """Raised when duplicate sample IDs are found."""


class InvalidTaskError(ValidationError):
    """Raised when an unsupported task type is provided."""


class MissingPredictionError(ValidationError):
    """Raised when required prediction data is missing."""


class ReservedColumnError(ValidationError):
    """Raised when a user-supplied column name collides with an SDK-owned column."""
