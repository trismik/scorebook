"""Accuracy metric implementation for Scorebook."""

from typing import Any, List

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry


@MetricRegistry.register()
class Accuracy(MetricBase):
    """Accuracy metric for evaluating model predictions of any type.

    Accuracy = correct predictions / total predictions
    """

    @staticmethod
    def score(predictions: List[Any], references: List[Any], score_type: str = "aggregate") -> Any:
        """Calculate accuracy score between predictions and references.

        Args:
            predictions: List of model predictions.
            references: List of ground truth reference values.
            score_type: One of "aggregate" (overall accuracy), "item" (per-item correctness),
                       or "all" (both aggregate and item scores)

        Returns:
            If score_type is "aggregate": Float value representing the accuracy score
            If score_type is "item": List of booleans indicating correctness of each prediction
            If score_type is "all": Dictionary with both aggregate score and item scores

        Raises:
            ValueError: If predictions and references have different lengths or invalid score_type.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if score_type not in ["aggregate", "item", "all"]:
            raise ValueError("score_type must be 'aggregate', 'item', or 'all'")

        item_scores = [p == r for p, r in zip(predictions, references)]
        aggregate_score = float(sum(item_scores) / len(predictions))

        scores = {
            "item": item_scores,
            "aggregate": aggregate_score,
            "all": {"aggregate": aggregate_score, "items": item_scores},
        }
        return scores[score_type]
