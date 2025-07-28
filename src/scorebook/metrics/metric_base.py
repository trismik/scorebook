"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from scorebook.types import EvaluatedItem


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    @property
    def name(self) -> str:
        """Return the metric name based on the class name."""
        return self.__class__.__name__.lower()

    @staticmethod
    @abstractmethod
    def score(
        *,
        output: Optional[Any] = None,
        label: Optional[Any] = None,
        evaluated_items: Optional[List[EvaluatedItem]] = None,
    ) -> Any:
        """Score either a single item or a list of evaluated items.

        Must provide either (output, label) OR evaluated_items, but not both.

        Args:
            output: Single prediction output
            label: Single ground truth label
            evaluated_items: List of evaluated items containing outputs and labels

        Raises:
            ValueError: If neither or both parameter sets are provided
        """
