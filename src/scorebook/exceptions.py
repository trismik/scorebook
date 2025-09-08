"""
Custom exceptions for the Scorebook framework.

This module defines specific exception types used throughout the Scorebook
evaluation framework to provide clear error handling and debugging information.
"""


class ScoreBookError(Exception):
    """Base exception class for all Scorebook-related errors."""


class EvaluationError(ScoreBookError):
    """Raised when there are errors during model evaluation."""


class ParameterValidationError(ScoreBookError):
    """Raised when invalid parameters are provided to evaluation functions."""


class InferenceError(EvaluationError):
    """Raised when there are errors during model inference."""


class MetricComputationError(EvaluationError):
    """Raised when metric computation fails."""

    def __init__(self, metric_name: str, dataset_name: str, original_error: Exception):
        """Initialize metric computation error."""
        self.metric_name = metric_name
        self.dataset_name = dataset_name
        self.original_error = original_error
        super().__init__(
            f"Failed to compute metric '{metric_name}' for dataset "
            f"'{dataset_name}': {original_error}"
        )


class DataMismatchError(EvaluationError):
    """Raised when there's a mismatch between outputs and expected labels."""

    def __init__(self, outputs_count: int, labels_count: int, dataset_name: str):
        """Initialize data mismatch error."""
        self.outputs_count = outputs_count
        self.labels_count = labels_count
        self.dataset_name = dataset_name
        super().__init__(
            f"Output count ({outputs_count}) doesn't match label count ({labels_count}) "
            f"for dataset '{dataset_name}'"
        )


class ParallelExecutionError(ScoreBookError):
    """Raised when parallel execution requirements are not met."""
