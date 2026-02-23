"""Accuracy metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class Accuracy(MetricBase):
    """Accuracy metric for evaluating model predictions of any type.

    Accuracy = correct predictions / total predictions
    """

    @staticmethod
    def score(outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate accuracy score between predictions and references.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            The aggregate accuracy score for all items (correct predictions / total predictions).
            Per-item scores as dicts: [{"accuracy": True/False}, ...].
        """
        if not outputs:  # Handle empty lists
            return {"accuracy": 0.0}, []

        # Calculate item scores
        matches = [output == label for output, label in zip(outputs, labels)]
        item_scores = [{"accuracy": match} for match in matches]

        # Calculate aggregate score
        correct_predictions = sum(matches)
        total_predictions = len(outputs)
        aggregate_scores = {"accuracy": correct_predictions / total_predictions}

        return aggregate_scores, item_scores
