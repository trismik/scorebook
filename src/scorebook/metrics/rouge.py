"""ROUGE metric implementation for Scorebook."""

from typing import Any, Dict, List, Optional, Tuple

from rouge_score import rouge_scorer

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry


@MetricRegistry.register()
class ROUGE(MetricBase):
    """ROUGE metric for evaluating text generation quality.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap between generated text and reference text.
    Returns ROUGE-1 and ROUGE-L F1 scores.
    """

    def __init__(self, scorer: Optional[rouge_scorer.RougeScorer] = None) -> None:
        """Initialize the ROUGE metric.

        Args:
            scorer: Optional custom RougeScorer instance. If not provided,
                   creates a default scorer with rouge1 and rougeL metrics.
        """
        self.scorer = (
            scorer if scorer else rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        )

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate ROUGE scores between predictions and references.

        Args:
            outputs: A list of generated text outputs.
            labels: A list of reference text labels.

        Returns:
            A tuple containing:
            - aggregate_scores: Dict with average rouge1 and rougeL F1 scores
            - item_scores: List of dicts with rouge1 and rougeL F1 scores for each pair
        """
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:  # Handle empty lists
            return {"rouge1": 0.0, "rougeL": 0.0}, []

        # Calculate item scores
        item_scores = []
        for output, label in zip(outputs, labels):
            # Convert to strings if needed
            output_str = str(output) if output is not None else ""
            label_str = str(label) if label is not None else ""

            # Calculate ROUGE scores
            scores = self.scorer.score(output_str, label_str)

            # Extract F1 scores (fmeasure)
            item_scores.append(
                {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}
            )

        # Calculate aggregate scores (average of all items)
        avg_rouge1 = sum(item["rouge1"] for item in item_scores) / len(item_scores)
        avg_rougeL = sum(item["rougeL"] for item in item_scores) / len(item_scores)

        aggregate_scores = {"rouge1": avg_rouge1, "rougeL": avg_rougeL}

        return aggregate_scores, item_scores
