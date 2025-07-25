"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    @property
    def name(self) -> str:
        """Return the metric name based on the class name."""
        return self.__class__.__name__.lower()

    @staticmethod
    @abstractmethod
    def score(predictions: List[Any], references: List[Any], score_type: str = "aggregate") -> Any:
        """Evaluate predictions against references.

        Args:
            predictions: Model predictions to evaluate.
            references: Ground truth references to compare against.
            score_type: aggregate, item, all

        Returns:
            Metric score as a float.

        Raises:
            ValueError: If the inputs are invalid.
        """
        raise NotImplementedError
