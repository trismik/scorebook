"""Precision metric implementation for Scorebook."""

from typing import Any, List, Optional

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry
from scorebook.types import EvaluatedItem
from scorebook.utils.mappers import to_binary_classification


@MetricRegistry.register()
class Precision(MetricBase):
    """Precision metric for binary classification.

    Precision = TP / (TP + FP)
    """

    @staticmethod
    def score(
        *,
        output: Optional[Any] = None,
        label: Optional[Any] = None,
        evaluated_items: Optional[List[EvaluatedItem]] = None,
    ) -> Any:
        """Calculate precision score between predictions and references.

        Args:
            output: Single prediction output
            label: Single ground truth label
            evaluated_items: List of evaluated items containing outputs and labels

        Returns:
            If scoring an individual item, returns the score as a string for the item
            If scoring a list of evaluated items, returns the aggregate score as a float

        Raises:
            ValueError: If neither or both parameter sets are provided
        """
        # Calculate aggregate precision score
        if evaluated_items:

            aggregate_score = 0.0
            true_positives = sum(
                1 for item in evaluated_items if item.scores["precision"] == "true_positive"
            )
            false_positives = sum(
                1 for item in evaluated_items if item.scores["precision"] == "false_positive"
            )

            if true_positives + false_positives > 0:
                aggregate_score = true_positives / (true_positives + false_positives)

            return aggregate_score

        # Classify as TP, FP, TN, FN
        else:
            return to_binary_classification(output, label)
