"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    name: str

    def __init__(self) -> None:
        """Initialize the metric."""
        if not hasattr(self, "name"):
            raise ValueError("Metric classes must define a 'name' class attribute")

    @staticmethod
    @abstractmethod
    def score(predictions: List[Any], references: List[Any]) -> Union[float, dict[str, float]]:
        """Evaluate predictions against references.

        Args:
            predictions: Model predictions to evaluate.
            references: Ground truth references to compare against.

        Returns:
            Metric score as a float.

        Raises:
            ValueError: If the inputs are invalid.
        """
        raise NotImplementedError
