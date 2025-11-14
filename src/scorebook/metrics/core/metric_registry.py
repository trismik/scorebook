"""
Registry module for evaluation metrics.

This module maintains a centralized registry of available evaluation metrics
that can be used to assess model performance. It provides a single access point
to retrieve all implemented metric classes.
"""

import importlib
from typing import Any, Callable, Dict, List, Type, Union

from scorebook.metrics.core.metric_base import MetricBase


class MetricRegistry:
    """A registry for evaluation metrics.

    This class provides a central registry for all evaluation metrics in the system.
    Metrics are lazily loaded on demand - when you request a metric by name, it will
    be automatically imported from the metrics directory using a naming convention.

    Naming Convention:
        Metric names are converted to module names by:
        - Converting to lowercase
        - Replacing spaces with underscores
        - Looking for: scorebook.metrics.{converted_name}

        Examples:
            "Accuracy" -> scorebook.metrics.accuracy
            "F1 Score" -> scorebook.metrics.f1_score
            "MyCustomMetric" -> scorebook.metrics.mycustommetric

    The registry supports:
    - Registering new metric classes with the @register decorator
    - Retrieving metric instances by name or class (with lazy loading)
    - Listing all registered metrics

    Usage:
        # Define a metric (auto-registers via decorator)
        @MetricRegistry.register()
        class MyMetric(MetricBase):
            ...

        # Get by name (auto-loads if needed)
        metric = MetricRegistry.get("mymetric")
        metric = MetricRegistry.get("My Metric")  # Also works

        # Get by class (no registry needed)
        metric = MetricRegistry.get(MyMetric)

        # List registered metrics
        metrics = MetricRegistry.list_metrics()

    Security:
        Lazy loading is safe because:
        - Import path is always prefixed with "scorebook.metrics."
        - Only modules with @register() decorator can be used
        - Python's import system validates module names
    """

    _registry: Dict[str, Type[MetricBase]] = {}

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
        """Attempt to lazily load a metric module using naming convention.

        Args:
            metric_name: The metric name (already lowercased with spaces -> underscores)

        Returns:
            True if the metric was successfully loaded
            False if the metric module doesn't exist

        Raises:
            ImportError: If the module exists but has import errors
        """
        # Check if already registered
        if metric_name in cls._registry:
            return True

        # Convert metric name to module name (spaces -> underscores, lowercase)
        module_name = metric_name.replace(" ", "_").lower()

        try:
            importlib.import_module(f"scorebook.metrics.{module_name}")
            return True
        except ModuleNotFoundError:
            # Module doesn't exist - this is expected for invalid metric names
            return False
        except ImportError as e:
            # Module exists but has import errors - re-raise with context
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
            ValueError: If the metric cannot be found or loaded.
            ImportError: If lazy loading fails due to import errors.
        """
        # If input is a class that's a subclass of MetricBase, instantiate it directly
        if isinstance(name_or_class, type) and issubclass(name_or_class, MetricBase):
            return name_or_class(**kwargs)

        # If input is a string, look up the class in the registry
        if isinstance(name_or_class, str):
            # Normalize: lowercase and replace spaces with underscores
            key = name_or_class.lower().replace(" ", "_")

            # Try lazy loading if not already registered
            if key not in cls._registry:
                # Attempt to lazy load the metric
                if not cls._lazy_load_metric(key):
                    # Module doesn't exist - provide helpful error
                    error_msg = (
                        f"Metric '{name_or_class}' could not be found. "
                        f"Attempted to import from 'scorebook.metrics.{key}'."
                    )
                    if cls._registry:
                        registered = ", ".join(sorted(cls._registry.keys()))
                        error_msg += f" Currently registered metrics: {registered}"
                    else:
                        error_msg += " No metrics are currently registered."
                    raise ValueError(error_msg)

            # After lazy loading attempt, check registry
            if key not in cls._registry:
                raise ValueError(
                    f"Metric '{name_or_class}' module was loaded but failed to register. "
                    f"Ensure the metric class has the @MetricRegistry.register() decorator."
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


def scorebook_metric(cls: Type[MetricBase]) -> Type[MetricBase]:
    """Register a custom metric with Scorebook.

    The metric will be registered using the lowercase version of the class name.
    For example, a class named "MyCustomMetric" will be registered as "mycustommetric"
    and can be accessed via MetricRegistry.get("mycustommetric") or score(metrics="mycustommetric").

    Args:
        cls: A metric class that inherits from MetricBase

    Returns:
        The same class, now registered with Scorebook

    Example:
        ```python
        from scorebook import scorebook_metric
        from scorebook.metrics import MetricBase

        @scorebook_metric
        class MyCustomMetric(MetricBase):
            def score(self, outputs, labels):
                # Your metric implementation
                return aggregate_scores, item_scores
        ```
    Raises:
        ValueError: If a metric with the same name is already registered
    """
    return MetricRegistry.register()(cls)
