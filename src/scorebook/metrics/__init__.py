"""
Metrics for evaluating model predictions.

This module provides a collection of evaluation metrics for comparing model outputs
against ground truth labels. Available metrics include standard classification and
generation metrics like accuracy, precision, recall, F1-score, etc.

Metrics can be accessed by name through the `get_metrics()` function or used
directly by instantiating specific metric classes. All metrics implement a
common interface for scoring predictions against references.
"""

from scorebook.metrics.accuracy import Accuracy
from scorebook.metrics.get_metrics import get_metrics
from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.precision import Precision

__all__ = ["MetricBase", "Precision", "Accuracy", "get_metrics"]
