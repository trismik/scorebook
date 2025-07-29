"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    @property
    def name(self) -> str:
        """Return the metric name based on the class name."""
        return self.__class__.__name__.lower()

    @staticmethod
    @abstractmethod
    def score(outputs: List[Any], labels: List[Any]) -> Tuple[Any, List[Any]]:
        """Calculate the metric score for a list of outputs and labels.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            An aggregate metric score for all items.
            Individual scores for each item.
        """
        raise NotImplementedError("MetricBase is an abstract class")
