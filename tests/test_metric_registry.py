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
    @MetricRegistry.register("test_metric")
    class TestMetric(MetricBase):
        name = "test_metric"

        def score(self, predictions, references):
            return 0.0

    assert "test_metric" in MetricRegistry.list_metrics()


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
    with pytest.raises(ValueError, match="Metric 'nonexistent' not registered"):
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
    assert "accuracy" in metrics
    assert "precision" in metrics


def test_prevent_metric_overwrite():
    """Test that metrics with the same name cannot be registered twice."""

    # First registration should succeed
    @MetricRegistry.register("duplicate_metric")
    class FirstMetric(MetricBase):
        name = "duplicate_metric"

        def score(self, predictions, references):
            return 0.0

    # Second registration with same name should raise ValueError
    with pytest.raises(ValueError, match="Metric 'duplicate_metric' is already registered"):

        @MetricRegistry.register("duplicate_metric")
        class SecondMetric(MetricBase):
            name = "duplicate_metric"

            def score(self, predictions, references):
                return 1.0

    # Verify that the original metric is still registered and wasn't overwritten
    metric = MetricRegistry.get("duplicate_metric")
    assert isinstance(metric, FirstMetric)
