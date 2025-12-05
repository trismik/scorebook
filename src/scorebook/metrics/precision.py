"""Precision metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple, Union

from sklearn.metrics import precision_score

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class Precision(MetricBase):
    """Precision score metric for evaluating model predictions using scikit-learn.

    Precision = TP / (TP + FP)

    This metric can handle both binary and multi-class classification tasks.

    Args:
        average: The averaging method(s) for multi-class classification.
            Can be a single string or list of strings:
            - 'macro': Unweighted mean across labels
            - 'micro': Global calculation counting total TP, FP
            - 'weighted': Weighted mean by support
            - 'all': All three methods simultaneously
            - List of methods: Calculate multiple methods
            Defaults to 'macro'.
    """

    def __init__(self, average: Union[str, List[str]] = "macro", **kwargs: Any) -> None:
        """Initialize Precision metric with specified averaging method(s).

        Args:
            average: Averaging method(s) - string or list of strings.
                Options: 'macro', 'micro', 'weighted', 'all', or a list of methods.
                Defaults to 'macro'.
            **kwargs: Additional keyword arguments passed to sklearn's precision_score.

        Raises:
            ValueError: If average contains invalid methods or combines 'all' with others.
        """
        # Normalize to list for validation
        averages = [average] if isinstance(average, str) else average

        # Validate
        valid = {"macro", "micro", "weighted", "all"}
        if not all(a in valid for a in averages):
            raise ValueError(f"Invalid average method(s). Must be from {valid}.")
        if len(averages) > 1 and "all" in averages:
            raise ValueError("'all' cannot be combined with other methods.")

        self.average = average
        self.kwargs = kwargs

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate Precision score between predictions and references using scikit-learn.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            Tuple containing:
                - aggregate_scores (Dict[str, float]): Dictionary with Precision scores
                  keyed by averaging method (e.g., {"Precision (macro)": 0.85} or
                  {"Precision (macro)": 0.85, "Precision (micro)": 0.82}).
                - item_scores (List[bool]): True/False list indicating correct
                  predictions.

        """

        # Normalize to list of methods to calculate
        if isinstance(self.average, str):
            methods = ["macro", "micro", "weighted"] if self.average == "all" else [self.average]
        else:
            methods = self.average

        # Handle empty lists
        if not outputs:
            return {f"Precision ({method})": 0.0 for method in methods}, []

        # Calculate Precision score using scikit-learn with configured averaging method
        # Default zero_division=0 unless overridden in kwargs
        kwargs = {"zero_division": 0, **self.kwargs}

        # Calculate item scores (correctness of each prediction)
        item_scores = [output == label for output, label in zip(outputs, labels)]

        # Calculate Precision for each method
        aggregate_scores = {
            f"Precision ({method})": precision_score(labels, outputs, average=method, **kwargs)
            for method in methods
        }

        return aggregate_scores, item_scores
