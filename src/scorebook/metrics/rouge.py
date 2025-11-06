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

    def __init__(self, rouge_types: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Initialize the ROUGE metric.

        Args:
            rouge_types: List of ROUGE types to calculate (e.g., ["rouge1", "rouge2", "rougeL"]).
                        Defaults to ["rouge1", "rougeL"].
            **kwargs: Additional keyword arguments to pass to RougeScorer
                     (e.g., use_stemmer, split_summaries, tokenizer).
                     Defaults to use_stemmer=True if not provided.
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rougeL"]
        if "use_stemmer" not in kwargs:
            kwargs["use_stemmer"] = True
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, **kwargs)

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate ROUGE scores between predictions and references.

        Args:
            outputs: A list of generated text outputs.
            labels: A list of reference text labels.

        Returns:
            A tuple containing:
            - aggregate_scores: Dict with average F1 scores for each configured ROUGE type
            - item_scores: List of dicts with F1 scores for each configured ROUGE type
        """
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:  # Handle empty lists
            return {rouge_type: 0.0 for rouge_type in self.rouge_types}, []

        # Calculate item scores
        item_scores = []
        for output, label in zip(outputs, labels):
            # Convert to strings if needed
            output_str = str(output) if output is not None else ""
            label_str = str(label) if label is not None else ""

            # Calculate ROUGE scores
            scores = self.scorer.score(output_str, label_str)

            # Extract F1 scores (fmeasure) for all configured rouge types
            item_score = {
                rouge_type: scores[rouge_type].fmeasure for rouge_type in self.rouge_types
            }
            item_scores.append(item_score)

        # Calculate aggregate scores (average of all items for each rouge type)
        aggregate_scores = {
            rouge_type: sum(item[rouge_type] for item in item_scores) / len(item_scores)
            for rouge_type in self.rouge_types
        }

        return aggregate_scores, item_scores
