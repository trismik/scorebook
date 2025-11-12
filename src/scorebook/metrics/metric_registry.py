"""
Registry module for evaluation metrics.

This module maintains a centralized registry of available evaluation metrics
that can be used to assess model performance. It provides a single access point
to retrieve all implemented metric classes.
"""

import importlib
from typing import Any, Callable, Dict, List, Type, Union

from scorebook.metrics.metric_base import MetricBase


class MetricRegistry:
    """A registry for evaluation metrics.

    This class provides a central registry for all evaluation metrics in the system.
    Metrics are lazily loaded on demand - when you request a metric by name, it will
    be automatically imported and registered if it's in the built-in metrics list.

    The registry supports:
    - Registering new metric classes with the @register decorator
    - Retrieving metric instances by name or class (with lazy loading)
    - Listing all available metrics

    Usage:
        # Define a metric (auto-registers via decorator)
        @MetricRegistry.register()
        class MyMetric(MetricBase):
            ...

        # Get by name (auto-loads if needed)
        metric = MetricRegistry.get("mymetric")

        # Get by class (no registry needed)
        metric = MetricRegistry.get(MyMetric)

        # List available metrics
        metrics = MetricRegistry.list_metrics()

    Security:
        Lazy loading only imports metrics from a predefined whitelist to prevent
        arbitrary code execution. See _BUILT_IN_METRICS for the list of valid metrics.
    """

    _registry: Dict[str, Type[MetricBase]] = {}

    _BUILT_IN_METRICS: Dict[str, str] = {
        "accuracy": "accuracy",
        "precision": "precision",
    }

    @classmethod
    def register(cls) -> Callable[[Type[MetricBase]], Type[MetricBase]]:
        """
        Register a metric class in the registry.

        Returns:
            A decorator that registers the class and returns it.

        Raises:
            ValueError: If a metric with the given name is already registered.
        """

        def decorator(metric_cls: Type[MetricBase]) -> Type[MetricBase]:

            key = metric_cls.__name__.lower()
            if key in cls._registry:
                raise ValueError(f"Metric '{key}' is already registered")
            cls._registry[key] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def _lazy_load_metric(cls, metric_name: str) -> bool:
        """Attempt to lazily load a metric module if it's in the built-in metrics list."""

        if metric_name not in cls._BUILT_IN_METRICS:
            return False

        if metric_name in cls._registry:
            return True

        module_name = cls._BUILT_IN_METRICS[metric_name]

        try:
            importlib.import_module(f"scorebook.metrics.{module_name}")
            return True
        except ImportError as e:
            raise ImportError(
                f"Failed to load metric '{metric_name}' from module "
                f"'scorebook.metrics.{module_name}': {e}"
            ) from e

    @classmethod
    def get(cls, name_or_class: Union[str, Type[MetricBase]], **kwargs: Any) -> MetricBase:
        """
        Get an instance of a registered metric by name or class.

        Args:
            name_or_class: The metric name (string) or class (subclass of MetricBase).
            **kwargs: Additional arguments to pass to the metric's constructor.

        Returns:
            An instance of the requested metric.

        Raises:
            ValueError: If the metric name is not registered or not in built-in metrics.
            ImportError: If lazy loading fails due to import errors.
        """
        # If input is a class that's a subclass of MetricBase, instantiate it directly
        if isinstance(name_or_class, type) and issubclass(name_or_class, MetricBase):
            return name_or_class(**kwargs)

        # If input is a string, look up the class in the registry
        if isinstance(name_or_class, str):
            key = name_or_class.lower()

            # Try lazy loading if not already registered
            if key not in cls._registry:
                # Attempt to lazy load the metric
                if not cls._lazy_load_metric(key):
                    # Not in whitelist - raise error
                    available_metrics = ", ".join(sorted(cls._BUILT_IN_METRICS.keys()))
                    raise ValueError(
                        f"Metric '{name_or_class}' is not a known metric. "
                        f"Available metrics: {available_metrics}"
                    )

            # After lazy loading attempt, check registry
            if key not in cls._registry:

                raise ValueError(
                    f"Metric '{name_or_class}' was loaded but failed to register. "
                    f"This is likely a bug in the metric implementation."
                )

            return cls._registry[key](**kwargs)

        raise ValueError(
            f"Invalid metric type: {type(name_or_class)}. "
            f"Must be string name or MetricBase subclass"
        )

    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        List all registered metrics.

        Returns:
            A list of metric names.
        """
        return list(cls._registry.keys())
