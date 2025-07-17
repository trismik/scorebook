"""Precision metric implementation for Scorebook.

This module provides the Precision metric class for evaluating binary classification
predictions. Precision measures the ratio of true positive predictions to the total
number of positive predictions made by the model.
"""

from typing import Any, List

from scorebook.metrics.metric_base import MetricBase


class Precision(MetricBase):
    """Precision metric for binary classification.

    Precision = TP / (TP + FP)
    """

    name = "precision"

    def __init__(self) -> None:
        """Initialize the Precision metric."""
        super().__init__()

    @staticmethod
    def score(predictions: List[Any], references: List[Any]) -> float:
        """Calculate precision score between predictions and references.

        Precision is calculated as true positives / (true positives + false positives).
        True positives are cases where both prediction and reference are 1.
        False positives are cases where prediction is 1 but reference is not.

        Args:
            predictions: List of model predictions (expected to be 0 or 1).
            references: List of ground truth reference values (expected to be 0 or 1).

        Returns:
            Float value representing the precision score (between 0.0 and 1.0).
            Returns 0.0 if there are no positive predictions.

        Raises:
            ValueError: If predictions and references have different lengths.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        true_positives = sum(p == r == 1 for p, r in zip(predictions, references))
        false_positives = sum(p == 1 and r != 1 for p, r in zip(predictions, references))

        if true_positives + false_positives == 0:
            return 0.0

        return float(true_positives / (true_positives + false_positives))
