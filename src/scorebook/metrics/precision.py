"""Precision metric implementation for Scorebook."""

from typing import Any, List, Union

from scorebook.metrics.metric_base import MetricBase


class Precision(MetricBase):
    """Precision metric for evaluating model predictions.

    Precision is the ratio of true positive predictions to the total number of
    positive predictions made by the model (true positives + false positives).
    """

    name = "Precision"

    def __init__(self) -> None:
        """Initialize the Precision metric."""
        super().__init__()

    @staticmethod
    def score(predictions: List[Any], references: List[Any]) -> Union[float, dict[str, float]]:
        """Calculate precision score.

        Args:
            predictions: List of predicted values (0/1 or False/True)
            references: List of true values (0/1 or False/True)

        Returns:
            Float value representing the precision score

        Raises:
            ValueError: If inputs are invalid or empty
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and labels must have the same length")

        true_positives = sum(p == l for p, l in zip(predictions, references))
        return float(true_positives / len(predictions))
