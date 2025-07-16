"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    name: Optional[str] = None

    def __init__(self, name: str):
        """Initialize the metric."""
        if self.name is None:
            raise ValueError("Metric classes must define a 'name' class attribute")

    @abstractmethod
    def score(
        self, predictions: List[Any], references: List[Any]
    ) -> Union[float, dict[str, float]]:
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
