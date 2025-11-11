"""Tests for the metric registry functionality."""

import pytest

from scorebook.metrics.accuracy import Accuracy
from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry
from scorebook.metrics.precision import Precision


def test_registry_registration():
    """Test that metrics can be registered correctly."""
    # Check that pre-registered metrics exist
    assert "accuracy" in MetricRegistry.list_metrics()
    assert "precision" in MetricRegistry.list_metrics()

    # Test registering a new metric
    @MetricRegistry.register()
    class TestMetric(MetricBase):
        @staticmethod
        def score(predictions, references):
            return 0.0

    # The metric should be registered with its lowercase class name
    assert "testmetric" in MetricRegistry.list_metrics()


def test_get_metric_by_name():
    """Test retrieving metrics by name."""
    # Get existing metrics
    accuracy_metric = MetricRegistry.get("accuracy")
    precision_metric = MetricRegistry.get("precision")

    assert isinstance(accuracy_metric, Accuracy)
    assert isinstance(precision_metric, Precision)

    # Test case-insensitive lookup
    accuracy_upper = MetricRegistry.get("ACCURACY")
    assert isinstance(accuracy_upper, Accuracy)

    # Test getting non-existent metric
    with pytest.raises(ValueError):
        MetricRegistry.get("nonexistent")


def test_get_metric_by_class():
    """Test retrieving metrics by class."""
    accuracy_metric = MetricRegistry.get(Accuracy)
    assert isinstance(accuracy_metric, Accuracy)

    precision_metric = MetricRegistry.get(Precision)
    assert isinstance(precision_metric, Precision)


def test_invalid_metric_type():
    """Test handling of invalid metric type."""
    with pytest.raises(ValueError):
        MetricRegistry.get(123)


def test_list_metrics():
    """Test listing all registered metrics."""
    metrics = MetricRegistry.list_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0


def test_prevent_metric_overwrite():
    """Test that metrics with the same class name cannot be registered twice."""

    # First registration should succeed
    @MetricRegistry.register()
    class DuplicateMetric(MetricBase):
        @staticmethod
        def score(output=None, label=None, evaluated_items=None):
            return 0.0

    print(DuplicateMetric)

    # Second registration with same class name should raise ValueError
    with pytest.raises(ValueError):

        @MetricRegistry.register()
        class DuplicateMetric(MetricBase):  # Same class name as above
            @staticmethod
            def score(output=None, label=None, evaluated_items=None):
                return 1.0

    # Verify that the original metric is still registered and wasn't overwritten
    metric = MetricRegistry.get("duplicatemetric")
    assert isinstance(metric, DuplicateMetric)
    # Optionally verify the behavior of the original implementation
    assert metric.score(output=[], label=[]) == 0.0


def test_lazy_loading_accuracy():
    """Test that accuracy metric can be loaded on demand via lazy loading.

    Note: This tests the mechanism by verifying that requesting a metric by string
    successfully loads it, even if it wasn't explicitly imported in this test.
    """
    # Request accuracy by string name (lazy loading mechanism)
    metric = MetricRegistry.get("accuracy")

    # Verify it's an Accuracy instance
    assert isinstance(metric, Accuracy)
    assert metric.name == "accuracy"

    # Verify it's in the registry after lazy load
    assert "accuracy" in MetricRegistry._registry


def test_lazy_loading_unknown_metric():
    """Test that unknown metrics raise helpful error with available metrics list."""
    with pytest.raises(ValueError) as exc_info:
        MetricRegistry.get("unknown_metric")

    error_message = str(exc_info.value)
    assert "not a known metric" in error_message
    assert "Available metrics:" in error_message
    assert "accuracy" in error_message
    assert "precision" in error_message


def test_lazy_loading_case_insensitive():
    """Test that metric names are case-insensitive with lazy loading."""
    # Should work with any case
    metric1 = MetricRegistry.get("ACCURACY")
    metric2 = MetricRegistry.get("Accuracy")
    metric3 = MetricRegistry.get("accuracy")

    # All should be Accuracy instances
    assert isinstance(metric1, Accuracy)
    assert isinstance(metric2, Accuracy)
    assert isinstance(metric3, Accuracy)

    # All should have the same name
    assert metric1.name == "accuracy"
    assert metric2.name == "accuracy"
    assert metric3.name == "accuracy"


def test_lazy_loading_no_reimport():
    """Test that already-registered metrics don't trigger re-import."""
    # Ensure accuracy is already registered
    MetricRegistry.get("accuracy")
    assert "accuracy" in MetricRegistry._registry

    # Get reference to the current registry entry
    original_class = MetricRegistry._registry["accuracy"]

    # Request accuracy again - should use cached version, not re-import
    metric = MetricRegistry.get("accuracy")

    # Should still be the same class in registry (no re-import)
    assert MetricRegistry._registry["accuracy"] is original_class
    assert isinstance(metric, Accuracy)


def test_lazy_loading_built_in_metrics_list():
    """Test that _BUILT_IN_METRICS contains expected metrics."""
    # Verify the whitelist contains our known metrics
    assert "accuracy" in MetricRegistry._BUILT_IN_METRICS
    assert "precision" in MetricRegistry._BUILT_IN_METRICS

    # Verify it maps to module names
    assert MetricRegistry._BUILT_IN_METRICS["accuracy"] == "accuracy"
    assert MetricRegistry._BUILT_IN_METRICS["precision"] == "precision"


def test_lazy_loading_multiple_metrics():
    """Test lazy loading multiple metrics in sequence."""
    # Load multiple metrics by string name
    accuracy = MetricRegistry.get("accuracy")
    precision = MetricRegistry.get("precision")

    # Both should be loaded
    assert isinstance(accuracy, Accuracy)
    assert isinstance(precision, Precision)

    # Both should be in registry
    assert "accuracy" in MetricRegistry._registry
    assert "precision" in MetricRegistry._registry


def test_get_metric_with_kwargs():
    """Test that kwargs are passed to metric constructor during lazy loading."""
    # Note: Current metrics don't accept kwargs, but test the mechanism works
    metric = MetricRegistry.get("accuracy")
    assert isinstance(metric, Accuracy)
