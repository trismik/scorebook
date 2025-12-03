"""Bertscore implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

import bert_score

from scorebook.metrics.core.metric_base import MetricBase
from scorebook.metrics.core.metric_registry import MetricRegistry


@MetricRegistry.register()
class BertScore(MetricBase):
    """Bert score metric for evaluating model predictions against reference text."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Bert score metric."""
        defaults = {"lang": "en", "verbose": False}
        self.kwargs = {**defaults, **kwargs}  # User kwargs override defaults

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate bert score between predictions and references.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            The aggregate precision, recall F1 for all items.
            The item level precision, recall, F1 scores for each output-label pair.
        """
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:  # Handle empty lists
            return {"precision": 0.0, "recall": 0.0, "F1": 0.0}, []

        # Calculate item scores
        p_scores, r_scores, f1_scores = bert_score.score(outputs, labels, **self.kwargs)

        item_scores = [
            {"precision": p, "recall": r, "F1": f1}
            for p, r, f1 in zip(p_scores.tolist(), r_scores.tolist(), f1_scores.tolist())
        ]
        aggregate_scores = {
            "precision": p_scores.mean().item(),
            "recall": r_scores.mean().item(),
            "F1": f1_scores.mean().item(),
        }

        return aggregate_scores, item_scores
