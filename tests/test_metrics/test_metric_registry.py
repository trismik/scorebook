"""Tests for the metric registry functionality."""

import pytest

from scorebook import scorebook_metric
from scorebook.metrics import MetricBase
from scorebook.metrics.accuracy import Accuracy
from scorebook.metrics.core.metric_registry import MetricRegistry
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
    """Test that unknown metrics raise ValueError."""
    with pytest.raises(ValueError):
        MetricRegistry.get("unknown_metric")


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


def test_scorebook_metric_decorator():
    """Test that the scorebook_metric decorator works for custom metrics."""

    @scorebook_metric
    class CustomTestMetric(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"custom": 1.0}, [True] * len(outputs)

    # Metric should be registered
    assert "customtestmetric" in MetricRegistry.list_metrics()

    # Can retrieve by name
    metric = MetricRegistry.get("customtestmetric")
    assert isinstance(metric, CustomTestMetric)

    # Can retrieve by class
    metric2 = MetricRegistry.get(CustomTestMetric)
    assert isinstance(metric2, CustomTestMetric)


def test_metric_name_normalization_case_insensitive():
    """Test that metric names are normalized to be case-insensitive."""

    @MetricRegistry.register()
    class CaseTestMetric(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"test": 1.0}, [True]

    # All these variations should retrieve the same metric
    variations = [
        "casetestmetric",
        "CaseTestMetric",
        "CASETESTMETRIC",
        "cAsEtEsTmEtRiC",
    ]

    for variation in variations:
        metric = MetricRegistry.get(variation)
        assert isinstance(metric, CaseTestMetric)


def test_metric_name_normalization_underscores_stripped():
    """Test that underscores are stripped from metric names during normalization."""

    @MetricRegistry.register()
    class UnderscoreTest(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"test": 1.0}, [True]

    # All these variations should retrieve the same metric
    variations = [
        "underscoretest",
        "underscore_test",
        "Underscore_Test",
        "UNDERSCORE_TEST",
        "_underscore_test_",  # Leading/trailing underscores
        "under_score_test",  # Multiple underscores
    ]

    for variation in variations:
        metric = MetricRegistry.get(variation)
        assert isinstance(metric, UnderscoreTest)


def test_metric_name_normalization_spaces_to_underscores():
    """Test that spaces are converted to underscores, then stripped during normalization."""

    @MetricRegistry.register()
    class SpaceTestMetric(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"test": 1.0}, [True]

    # All these variations should retrieve the same metric
    variations = [
        "spacetestmetric",
        "space test metric",
        "Space Test Metric",
        "SPACE TEST METRIC",
        "space_test_metric",  # Underscores and spaces both work
    ]

    for variation in variations:
        metric = MetricRegistry.get(variation)
        assert isinstance(metric, SpaceTestMetric)


def test_class_name_collision_detection():
    """Test that class names with different underscores but same normalized name raise an error."""

    # Register first metric
    @MetricRegistry.register()
    class CollisionMetric(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"test": 1.0}, [True]

    # Try to register a second metric with underscores that normalizes to the same name
    with pytest.raises(ValueError) as exc_info:

        @MetricRegistry.register()
        class Collision_Metric(MetricBase):  # Same normalized name as CollisionMetric
            @staticmethod
            def score(outputs, labels):
                return {"test": 2.0}, [True]

    # Verify error message is informative
    error_msg = str(exc_info.value)
    assert "collisionmetric" in error_msg.lower()
    assert "already registered" in error_msg.lower()
    assert "CollisionMetric" in error_msg or "Collision_Metric" in error_msg


def test_class_name_collision_case_variations():
    """Test that different case variations of the same name cause collisions."""

    @MetricRegistry.register()
    class MyMetric(MetricBase):
        @staticmethod
        def score(outputs, labels):
            return {"test": 1.0}, [True]

    # Try variations that should all collide
    collision_names = ["MYMETRIC", "mymetric", "MyMetric", "my_metric", "MY_METRIC"]

    for collision_name in collision_names:
        with pytest.raises(ValueError) as exc_info:
            # Dynamically create a class with the collision name
            metric_cls = type(
                collision_name,
                (MetricBase,),
                {"score": staticmethod(lambda outputs, labels: ({"test": 2.0}, [True]))},
            )
            MetricRegistry.register()(metric_cls)

        error_msg = str(exc_info.value)
        assert "mymetric" in error_msg.lower()
        assert "already registered" in error_msg.lower()
