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
        All metric names are normalized by:
        - Converting to lowercase
        - Removing all underscores and spaces

        Module files must follow this normalized naming (lowercase, no underscores/spaces):
            Examples:
                Class "Accuracy" → module "accuracy.py"
                Class "F1Score" → module "f1score.py"
                Class "MeanSquaredError" → module "meansquarederror.py"

        User input is also normalized, so all variations work:
            "f1_score", "F1Score", "f1 score" → all resolve to "f1score"

    Collision Detection:
        Class names that normalize to the same key will raise an error:
            "F1Score" and "F1_Score" both → "f1score" (COLLISION)
            "MetricName" and "Metric_Name" both → "metricname" (COLLISION)

    Security:
        Lazy loading is restricted to modules in the "scorebook.metrics." namespace.
        Python's import system validates module names.
    """

    _registry: Dict[str, Type[MetricBase]] = {}

    @classmethod
    def register(cls) -> Callable[[Type[MetricBase]], Type[MetricBase]]:
        """Register a metric class in the registry.

        Returns:
            A decorator that registers the class and returns it.

        Raises:
            ValueError: If a metric with the given name is already registered.
        """

        def decorator(metric_cls: Type[MetricBase]) -> Type[MetricBase]:

            # Normalize: lowercase and strip underscores and spaces
            key = metric_cls.__name__.lower().replace("_", "").replace(" ", "")
            if key in cls._registry:
                raise ValueError(
                    f"Metric '{key}' is already registered. "
                    f"Class names '{metric_cls.__name__}' and "
                    f"'{cls._registry[key].__name__}' both normalize to '{key}'."
                )
            cls._registry[key] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def _lazy_load_metric(cls, normalized_key: str) -> None:
        """Attempt to lazily load a metric module using naming convention.

        Module files must be named using the normalized key (lowercase, no underscores/spaces).

        Args:
            normalized_key: The normalized metric name (lowercase, no underscores/spaces): "f1score"

        Raises:
            ValueError: If the module doesn't exist or fails to register
            ImportError: If the module exists but has import errors
        """
        # Check if already registered
        if normalized_key in cls._registry:
            return

        # Try to import the module using the normalized key
        try:
            importlib.import_module(f"scorebook.metrics.{normalized_key}")
        except ModuleNotFoundError:
            # Module doesn't exist - provide helpful error
            error_msg = (
                f"Metric '{normalized_key}' could not be found. "
                f"Attempted to import from 'scorebook.metrics.{normalized_key}'."
            )
            if cls._registry:
                registered = ", ".join(sorted(cls._registry.keys()))
                error_msg += f" Currently registered metrics: {registered}"
            else:
                error_msg += " No metrics are currently registered."
            raise ValueError(error_msg)
        except ImportError as e:
            # Module exists but has import errors - re-raise with context
            raise ImportError(
                f"Failed to load metric '{normalized_key}' from module "
                f"'scorebook.metrics.{normalized_key}': {e}"
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
            # Normalize: lowercase and strip all underscores and spaces
            normalized_key = name_or_class.lower().replace("_", "").replace(" ", "")

            # Try lazy loading if not already registered
            if normalized_key not in cls._registry:
                cls._lazy_load_metric(normalized_key)

            # After lazy loading attempt, check registry
            if normalized_key not in cls._registry:
                raise ValueError(
                    f"Metric '{name_or_class}' module was loaded but failed to register. "
                    f"Ensure the metric class has the @scorebook_metric decorator."
                )

            return cls._registry[normalized_key](**kwargs)

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
        ValueError: If a metric with the same normalized name is already registered
    """
    return MetricRegistry.register()(cls)
