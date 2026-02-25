"""
Custom exceptions for the Scorebook framework.

This module defines specific exception types used throughout the Scorebook
evaluation framework to provide clear error handling and debugging information.
"""


class ScoreBookError(Exception):
    """Base exception class for all Scorebook-related errors."""


class EvalDatasetError(ScoreBookError):
    """Base exception class for all EvalDataset errors."""


class DatasetConfigurationError(EvalDatasetError):
    """Raised when dataset configuration is invalid (e.g., mutually exclusive parameters)."""


class MissingFieldError(EvalDatasetError):
    """Raised when required field is missing from dataset."""

    def __init__(self, field_name: str, field_type: str, available_fields: list[str]):
        """Initialize missing field error with structured context."""
        self.field_name = field_name
        self.field_type = field_type  # "input" or "label"
        self.available_fields = available_fields
        super().__init__(
            f"{field_type.capitalize()} field '{field_name}' not found. "
            f"Available fields: {', '.join(available_fields)}"
        )


class DatasetLoadError(EvalDatasetError):
    """Raised when dataset fails to load from source (file or remote)."""


class DatasetParseError(EvalDatasetError):
    """Raised when dataset file cannot be parsed (CSV, JSON, YAML)."""


class DatasetNotInitializedError(EvalDatasetError):
    """Raised when operations are attempted on uninitialized dataset."""


class DatasetSampleError(EvalDatasetError):
    """Raised when sampling parameters are invalid."""

    def __init__(self, sample_size: int, dataset_size: int, dataset_name: str):
        """Initialize dataset sample error with structured context."""
        self.sample_size = sample_size
        self.dataset_size = dataset_size
        self.dataset_name = dataset_name
        super().__init__(
            f"Sample size {sample_size} exceeds dataset size {dataset_size} "
            f"for dataset '{dataset_name}'"
        )


class EvaluationError(ScoreBookError):
    """Raised when there are errors during model evaluation."""


class AllRunsFailedError(EvaluationError):
    """Raised when all evaluation runs in a multi-run sweep fail."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        """Initialize with a list of (run_description, exception) tuples."""
        self.errors = errors
        run_summaries = "\n".join(
            f"  - {desc}: {type(exc).__name__}: {exc}" for desc, exc in errors
        )
        super().__init__(f"All {len(errors)} evaluation runs failed:\n{run_summaries}")


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


class ScoreError(ScoreBookError):
    """Raised when there are errors during scoring."""


class DataMismatchError(ScoreError):
    """Raised when there's a mismatch between outputs and expected labels."""

    def __init__(self, outputs_count: int, labels_count: int, dataset_name: str = "Dataset"):
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
