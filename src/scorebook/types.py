"""
Type definitions for evaluation results in the Scorebook framework.

This module defines the data structures used to represent evaluation results,
including individual prediction outcomes and aggregated dataset metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalResult:
    """
    Container for a single evaluation result.

    Attributes:
        dataset_item: Original item from the dataset that was evaluated
        output: Model's prediction/output for this item
        label: Ground truth label/reference for this item
    """

    dataset_item: Dict[str, Any]
    output: Any
    label: Any


@dataclass
class DatasetResults:
    """
    Container for evaluation results across an entire dataset.

    Attributes:
        items: List of individual evaluation results for each dataset item
        metrics: Dictionary mapping metric names to their computed scores
    """

    items: List[EvalResult]
    metrics: Dict[str, float]
