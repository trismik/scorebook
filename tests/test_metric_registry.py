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
    with pytest.raises(ValueError, match="Invalid metric type"):
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
        def score(predictions, references):
            return 0.0

    print(DuplicateMetric)

    # Second registration with same class name should raise ValueError
    with pytest.raises(ValueError, match="Metric 'duplicatemetric' is already registered"):

        @MetricRegistry.register()
        class DuplicateMetric(MetricBase):  # Same class name as above
            @staticmethod
            def score(predictions, references):
                return 1.0

    # Verify that the original metric is still registered and wasn't overwritten
    metric = MetricRegistry.get("duplicatemetric")
    assert isinstance(metric, DuplicateMetric)
    # Optionally verify the behavior of the original implementation
    assert metric.score([], []) == 0.0
