"""F1 metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from sklearn.metrics import f1_score, precision_score, recall_score

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry


@MetricRegistry.register()
class F1(MetricBase):
    """F1 score metric for evaluating model predictions using scikit-learn.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    where:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)

    This metric can handle both binary and multi-class classification tasks.

    Args:
        average: The averaging method for multi-class classification.
            - 'macro': Calculate metrics for each label and find their
              unweighted mean.
            - 'micro': Calculate metrics globally by counting total TP, FP, FN.
            - 'weighted': Calculate metrics for each label and find their
              average weighted by support.
            Defaults to 'macro'.
    """

    def __init__(self, average: str = "macro", **kwargs: Any) -> None:
        """Initialize F1 metric with specified averaging method.

        Args:
            average: Averaging method ('macro', 'micro', or 'weighted').
                Defaults to 'macro'.
            **kwargs: Additional keyword arguments passed to scikit-learn's
                f1_score, precision_score, and recall_score functions.

        Raises:
            ValueError: If average is not one of 'macro', 'micro', or 'weighted'.
        """
        valid_averages = ["macro", "micro", "weighted"]
        if average not in valid_averages:
            raise ValueError(
                f"Invalid average method: '{average}'. " f"Must be one of {valid_averages}."
            )
        self.average = average
        self.kwargs = kwargs

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate F1 score between predictions and references using scikit-learn.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            Tuple containing:
                - aggregate_scores (Dict[str, float]): Dictionary with F1, precision, and recall.
                - item_scores (List[bool]): True/False list indicating correct predictions.

        """
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:  # Handle empty lists
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}, []

        # Calculate metrics using scikit-learn with configured averaging method
        # Default zero_division=0 unless overridden in kwargs
        kwargs = {"zero_division": 0, **self.kwargs}
        f1 = f1_score(labels, outputs, average=self.average, **kwargs)
        precision = precision_score(labels, outputs, average=self.average, **kwargs)
        recall = recall_score(labels, outputs, average=self.average, **kwargs)

        # Calculate item scores (correctness of each prediction)
        item_scores = [output == label for output, label in zip(outputs, labels)]

        # Create aggregate scores dictionary
        aggregate_scores = {"f1": f1, "precision": precision, "recall": recall}

        return aggregate_scores, item_scores
