"""Accuracy metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class Accuracy(MetricBase):
    """Accuracy metric for evaluating model predictions of any type.

    Accuracy = correct predictions / total predictions
    """

    @staticmethod
    def score(outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate accuracy score between predictions and references.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            The aggregate accuracy score for all items (correct predictions / total predictions).
            The item scores for each output-label pair (true/false).
        """
        if not outputs:  # Handle empty lists
            return {"accuracy": 0.0}, []

        # Calculate item scores
        item_scores = [output == label for output, label in zip(outputs, labels)]

        # Calculate aggregate score
        correct_predictions = sum(item_scores)
        total_predictions = len(outputs)
        aggregate_scores = {"accuracy": correct_predictions / total_predictions}

        return aggregate_scores, item_scores
