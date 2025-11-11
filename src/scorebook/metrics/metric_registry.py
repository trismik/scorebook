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

    # Whitelist of built-in metrics (maps lowercase name to module name)
    _BUILT_IN_METRICS: Dict[str, str] = {
        "accuracy": "accuracy",
        "precision": "precision",
        # Add new metrics here as they're implemented
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
        """
        Attempt to lazily load a metric module if it's in the built-in metrics list.

        This method provides safe lazy loading by only importing metrics from a
        predefined whitelist. It will not import arbitrary modules.

        Args:
            metric_name: The lowercase metric name (e.g., "accuracy")

        Returns:
            True if the metric was successfully loaded (or already loaded)
            False if the metric is not in the built-in metrics list

        Raises:
            ImportError: If the metric module exists in the whitelist but fails to import
        """
        # Check if metric is in our whitelist
        if metric_name not in cls._BUILT_IN_METRICS:
            return False

        # Check if already loaded
        if metric_name in cls._registry:
            return True

        # Get the module name from whitelist
        module_name = cls._BUILT_IN_METRICS[metric_name]

        # Attempt to import the metric module
        # This will trigger the @register decorator
        try:
            importlib.import_module(f"scorebook.metrics.{module_name}")
            return True
        except ImportError as e:
            # Metric is in whitelist but module doesn't exist or has import errors
            # Re-raise with context
            raise ImportError(
                f"Failed to load metric '{metric_name}' from module "
                f"'scorebook.metrics.{module_name}': {e}"
            ) from e

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
            ValueError: If the metric name is not registered or not in built-in metrics.
            ImportError: If lazy loading fails due to import errors.
        """
        # If input is a class that's a subclass of BaseMetric, instantiate it directly
        if isinstance(name_or_class, type) and issubclass(name_or_class, MetricBase):
            return name_or_class(**kwargs)

        # If input is a string, look up the class in the registry
        if isinstance(name_or_class, str):
            key = name_or_class.lower()

            # Try lazy loading if not already registered
            if key not in cls._registry:
                # Attempt to lazy load the metric
                if not cls._lazy_load_metric(key):
                    # Not in whitelist - provide helpful error
                    available_metrics = ", ".join(sorted(cls._BUILT_IN_METRICS.keys()))
                    raise ValueError(
                        f"Metric '{name_or_class}' is not a known metric. "
                        f"Available metrics: {available_metrics}"
                    )
                # Lazy load succeeded, check registry again

            # After lazy loading attempt, check registry
            if key not in cls._registry:
                # This shouldn't happen if lazy load succeeded
                # Indicates the module imported but didn't register
                raise ValueError(
                    f"Metric '{name_or_class}' was loaded but failed to register. "
                    f"This is likely a bug in the metric implementation."
                )

            return cls._registry[key](**kwargs)

        raise ValueError(
            f"Invalid metric type: {type(name_or_class)}. "
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
