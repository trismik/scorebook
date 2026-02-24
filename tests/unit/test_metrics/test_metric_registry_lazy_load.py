"""Tests for lazy loading metrics without explicit imports.

This test module deliberately does NOT import Accuracy metric
to verify that the lazy loading mechanism in MetricRegistry works correctly.
"""

import pytest

from scorebook.metrics.core.metric_registry import MetricRegistry


def test_lazy_load_accuracy():
    """Test that accuracy metric can be loaded via lazy loading without explicit import."""
    # No import of Accuracy class - should be loaded via lazy loading
    metric = MetricRegistry.get("accuracy")

    # Verify it was loaded and is an instance of the correct class
    assert metric is not None
    assert metric.name == "accuracy"
    assert hasattr(metric, "score")

    # Verify it's actually in the registry now
    assert "accuracy" in MetricRegistry._registry

    # Test that it can actually score
    predictions = ["cat", "dog", "bird"]
    references = ["cat", "dog", "bird"]
    scores, item_results = metric.score(predictions, references)

    # Should return perfect accuracy
    assert scores["accuracy"] == 1.0
    assert all(item["accuracy"] for item in item_results)  # All items should be correct


def test_lazy_load_case_variations():
    """Test that lazy loading works with different case variations."""
    # All these should work and return the same metric type
    accuracy1 = MetricRegistry.get("accuracy")
    accuracy2 = MetricRegistry.get("ACCURACY")
    accuracy3 = MetricRegistry.get("Accuracy")

    # All should be the same type
    assert type(accuracy1) is type(accuracy2)
    assert type(accuracy2) is type(accuracy3)
    assert accuracy1.name == accuracy2.name == accuracy3.name == "accuracy"


def test_lazy_load_nonexistent_metric():
    """Test that attempting to load a nonexistent metric raises ValueError."""
    with pytest.raises(ValueError):
        MetricRegistry.get("nonexistent_metric")
