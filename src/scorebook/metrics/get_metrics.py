"""
Registry module for evaluation metrics.

This module maintains a centralized registry of available evaluation metrics
that can be used to assess model performance. It provides a single access point
to retrieve all implemented metric classes.
"""

from typing import List, Type

from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.precision import Precision


def get_metrics() -> List[Type[MetricBase]]:
    """
    Get all available evaluation metric classes.

    Returns:
        List[Type[MetricBase]]: A list of metric classes that inherit from MetricBase.
    """

    return [Precision]
