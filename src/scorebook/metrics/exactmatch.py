"""Exact Match metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class ExactMatch(MetricBase):
    """Exact Match metric for evaluating string predictions.

    Compares strings for exact equality with optional preprocessing.

    Args:
        case_insensitive: If True, convert strings to lowercase before comparison.
            Defaults to True.
        strip: If True, strip leading and trailing whitespace before comparison.
            Defaults to True.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "exact_match"

    def __init__(self, case_insensitive: bool = True, strip: bool = True) -> None:
        """Initialize ExactMatch metric with preprocessing options.

        Args:
            case_insensitive: If True, convert strings to lowercase before comparison.
                Defaults to True.
            strip: If True, strip leading and trailing whitespace before comparison.
                Defaults to True.
        """
        self.case_insensitive = case_insensitive
        self.strip = strip

    def _preprocess(self, value: Any) -> Any:
        """Apply preprocessing to a value if it's a string.

        Args:
            value: The value to preprocess.

        Returns:
            The preprocessed value (string) or original value (non-string).
        """
        if not isinstance(value, str):
            return value

        result = value
        if self.strip:
            result = result.strip()
        if self.case_insensitive:
            result = result.lower()
        return result

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate exact match score between predictions and references.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            The aggregate exact match score for all items (matches / total).
            The item scores for each output-label pair (true/false).
        """
        if not outputs:
            return {"exact_match": 0.0}, []

        # Calculate item scores with preprocessing
        item_scores = [
            self._preprocess(output) == self._preprocess(label)
            for output, label in zip(outputs, labels)
        ]

        # Calculate aggregate score
        matches = sum(item_scores)
        total = len(outputs)
        aggregate_scores = {"exact_match": matches / total}

        return aggregate_scores, item_scores
