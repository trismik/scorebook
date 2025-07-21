"""
Registry module for evaluation metrics.

This module maintains a centralized registry of available evaluation metrics
that can be used to assess model performance. It provides a single access point
to retrieve all implemented metric classes.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from scorebook.metrics.metric_base import MetricBase


class MetricRegistry:
    """A registry for evaluation metrics.

    This class provides a central registry for all evaluation metrics in the system.
    It allows metrics to be registered with unique names and retrieved either by
    name or by class. The registry ensures that metrics are properly initialized
    and accessible throughout the application.

    The registry supports:
    - Registering new metric classes with optional custom names
    - Retrieving metric instances by name or class
    - Listing all available metrics

    Usage:
        @MetricRegistry.register("custom_name")
        class MyMetric(MetricBase):
            ...

        # Get by name
        metric = MetricRegistry.get("custom_name")

        # Get by class
        metric = MetricRegistry.get(MyMetric)

        # List available metrics
        metrics = MetricRegistry.list_metrics()
    """

    _registry: Dict[str, Type[MetricBase]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[Type[MetricBase]], Type[MetricBase]]:
        """
        Register a metric class in the registry.

        Args:
            name: Optional name for the metric. Defaults to the class name in lowercase.

        Returns:
            A decorator that registers the class and returns it.

        Raises:
            ValueError: If a metric with the given name is already registered.
        """

        def decorator(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
            key = name or metric_cls.__name__.lower()
            if key in cls._registry:
                raise ValueError(f"Metric '{key}' is already registered")
            cls._registry[key] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def get(cls, name_or_class: Union[str, Type[MetricBase]], **kwargs: Any) -> MetricBase:
        """
        Get an instance of a registered metric by name or class.

        Args:
            name_or_class: The metric name (string) or class (subclass of BaseMetric).
            **kwargs: Additional arguments to pass to the metric's constructor.

        Returns:
            An instance of the requested metric.

        Raises:
            ValueError: If the metric name is not registered.
        """
        # If input is a class that's a subclass of BaseMetric, instantiate it directly
        if isinstance(name_or_class, type) and issubclass(name_or_class, MetricBase):
            return name_or_class(**kwargs)

        # If input is a string, look up the class in the registry
        if isinstance(name_or_class, str):
            key = name_or_class.lower()
            if key not in cls._registry:
                raise ValueError(f"Metric '{name_or_class}' not registered.")
            return cls._registry[key](**kwargs)

        raise ValueError(
            f"Invalid metric type: {type(name_or_class)}."
            f"Must be string name or BaseMetric subclass"
        )

    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        List all registered metrics.

        Returns:
            A list of metric names.
        """
        return list(cls._registry.keys())
