"""Accuracy metric implementation for Scorebook."""

from typing import Any, List

from scorebook.metrics.metric_base import MetricBase


class Accuracy(MetricBase):
    """Accuracy metric for evaluating model predictions of any type.

    Accuracy = correct predictions / total predictions
    """

    name = "accuracy"

    def __init__(self) -> None:
        """Initialize the Accuracy metric."""
        super().__init__()

    @staticmethod
    def score(predictions: List[Any], references: List[Any]) -> float:
        """Calculate accuracy score between predictions and references.

        Accuracy is calculated as the number of correct predictions divided by the total number
        of predictions.

        Args:
            predictions: List of model predictions.
            references: List of ground truth reference values.

        Returns:
            Float value representing the accuracy score (between 0.0 and 1.0).

        Raises:
            ValueError: If predictions and references have different lengths.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        correct_predictions = sum(p == r for p, r in zip(predictions, references))
        return float(correct_predictions / len(predictions))
