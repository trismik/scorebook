"""Precision metric implementation for Scorebook."""

from typing import Any, List

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry
from scorebook.utils.mappers import to_binary_classification


@MetricRegistry.register()
class Precision(MetricBase):
    """Precision metric for binary classification.

    Precision = TP / (TP + FP)
    """

    @staticmethod
    def score(predictions: List[Any], references: List[Any], score_type: str = "aggregate") -> Any:
        """Calculate precision score between predictions and references.

        Args:
            predictions: List of model binary predictions.
            references: List of ground truth binary reference values.
            score_type: One of "aggregate", "item", or "all"

        Returns:
            If score_type is "aggregate": Float precision score
            If score_type is "item": List of classification results
                ("true_positive", "false_positive", "true_negative", "false_negative")
            If score_type is "all": Dictionary with both aggregate score and item results

        Raises:
            ValueError: If predictions and references have different lengths or invalid score_type.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if score_type not in ["aggregate", "item", "all"]:
            raise ValueError("score_type must be 'aggregate', 'item', or 'all'")

        item_scores = [
            to_binary_classification(pred, ref) for pred, ref in zip(predictions, references)
        ]

        # Calculate aggregate precision score
        true_positives = sum(1 for result in item_scores if result == "true_positive")
        false_positives = sum(1 for result in item_scores if result == "false_positive")

        aggregate_score = 0.0
        if true_positives + false_positives > 0:
            aggregate_score = true_positives / (true_positives + false_positives)

        scores = {
            "item": item_scores,
            "aggregate": aggregate_score,
            "all": {"aggregate": aggregate_score, "items": item_scores},
        }

        return scores[score_type]
