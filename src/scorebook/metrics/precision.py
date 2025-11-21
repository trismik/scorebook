"""Precision metric implementation for Scorebook."""

from typing import Any, Dict, List, Tuple

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class Precision(MetricBase):
    """Precision metric for binary classification.

    Precision = TP / (TP + FP)
    """

    @staticmethod
    def score(outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Not implemented."""
        raise NotImplementedError("Precision not implemented")
