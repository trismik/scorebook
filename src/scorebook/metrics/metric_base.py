"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class MetricBase(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, name: str):
        """
        Create a new metric instance.

        :param name: the name of the metric
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            The name of the metric.
        """
        return self.name

    @abstractmethod
    def evaluate(
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
