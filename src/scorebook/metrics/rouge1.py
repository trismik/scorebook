"""ROUGE-1 metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from rouge_score import rouge_scorer

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class Rouge1(MetricBase):
    """ROUGE-1 metric for evaluating text generation quality.

    ROUGE-1 measures unigram overlap between generated text and reference text.
    Returns a single F1 score for rouge1.
    """

    def __init__(self, use_stemmer: bool = True, **kwargs: Any) -> None:
        """Initialize the ROUGE-1 metric.

        Args:
            use_stemmer: Whether to apply Porter stemmer. Defaults to True.
            **kwargs: Additional arguments for RougeScorer.
        """
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=use_stemmer, **kwargs)

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate ROUGE-1 scores between predictions and references.

        Args:
            outputs: A list of generated text outputs.
            labels: A list of reference text labels.

        Returns:
            Tuple of (aggregate_scores, item_scores):
            - aggregate_scores: Dict with single key {"rouge1": <float>}
            - item_scores: List of dicts with single key {"rouge1": <float>}
        """
        if not outputs:  # Handle empty lists
            return {"rouge1": 0.0}, []

        # Calculate item scores
        item_scores = []
        for output, label in zip(outputs, labels):
            # Convert to strings if needed
            output_str = str(output) if output is not None else ""
            label_str = str(label) if label is not None else ""

            # Calculate ROUGE-1 score
            scores = self.scorer.score(output_str, label_str)

            # Extract F1 score for rouge1
            item_score = {"rouge1": scores["rouge1"].fmeasure}
            item_scores.append(item_score)

        # Calculate aggregate score (average of all items)
        aggregate_scores = {
            "rouge1": sum(item["rouge1"] for item in item_scores) / len(item_scores)
        }

        return aggregate_scores, item_scores
