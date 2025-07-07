"""Dataset implementation for scorebook."""

from typing import Any, List


class Dataset:
    """Dataset implementation for scorebook."""

    def __init__(self, name: str, data: Any, metrics: List[Any]):
        """
        Create a new dataset instance.

        :param name: the name of the dataset.
        :param data: the data to use.
        :param metrics: the metrics to use.
        """
        self.name: str = name
        self.data: Any = data
        self.metrics: List[Any] = metrics
