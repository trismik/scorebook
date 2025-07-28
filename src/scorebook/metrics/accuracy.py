"""Accuracy metric implementation for Scorebook."""

from typing import Any, List, Optional

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry
from scorebook.types import EvaluatedItem


@MetricRegistry.register()
class Accuracy(MetricBase):
    """Accuracy metric for evaluating model predictions of any type.

    Accuracy = correct predictions / total predictions
    """

    @staticmethod
    def score(
        *,
        output: Optional[Any] = None,
        label: Optional[Any] = None,
        evaluated_items: Optional[List[EvaluatedItem]] = None,
    ) -> Any:
        """Calculate accuracy score between predictions and references.

        Args:
            output: Single prediction output
            label: Single ground truth label
            evaluated_items: List of evaluated items containing outputs and labels

        Returns:
            If scoring an individual item, returns the score as a bool for the item
            If scoring a list of evaluated items, returns the aggregate score as a float

        Raises:
            ValueError: If neither or both parameter sets are provided
        """
        # return aggregate score
        if evaluated_items:
            return len([item for item in evaluated_items if item.scores["accuracy"]]) / len(
                evaluated_items
            )

        # return item score
        else:
            return output == label
