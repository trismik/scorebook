"""ROUGE-L metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from rouge_score import rouge_scorer

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class RougeL(MetricBase):
    """ROUGE-L metric for evaluating text generation quality.

    ROUGE-L measures longest common subsequence between generated text and
    reference text. Returns a single F1 score for rougeL.
    """

    def __init__(self, use_stemmer: bool = True, **kwargs: Any) -> None:
        """Initialize the ROUGE-L metric.

        Args:
            use_stemmer: Whether to apply Porter stemmer. Defaults to True.
            **kwargs: Additional arguments for RougeScorer.
        """
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer, **kwargs)

    def score(
        self, outputs: List[Any], labels: List[Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate ROUGE-L scores between predictions and references.

        Args:
            outputs: A list of generated text outputs.
            labels: A list of reference text labels.

        Returns:
            Tuple of (aggregate_scores, item_scores):
            - aggregate_scores: Dict with single key {"rougeL": <float>}
            - item_scores: List of dicts with single key {"rougeL": <float>}
        """
        if not outputs:  # Handle empty lists
            return {"rougeL": 0.0}, []

        # Calculate item scores
        item_scores = []
        for output, label in zip(outputs, labels):
            # Convert to strings if needed
            output_str = str(output) if output is not None else ""
            label_str = str(label) if label is not None else ""

            # Calculate ROUGE-L score
            scores = self.scorer.score(label_str, output_str)

            # Extract F1 score for rougeL
            item_score = {"rougeL": scores["rougeL"].fmeasure}
            item_scores.append(item_score)

        # Calculate aggregate score (average of all items)
        aggregate_scores = {
            "rougeL": sum(item["rougeL"] for item in item_scores) / len(item_scores)
        }

        return aggregate_scores, item_scores
