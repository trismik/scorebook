"""Unit tests for the score() function."""

from typing import Any, Dict, List, Tuple

import pytest

from scorebook import score
from scorebook.exceptions import ParameterValidationError
from scorebook.metrics.accuracy import Accuracy
from scorebook.metrics.core.metric_base import MetricBase


# Mock metrics for collision testing
class MockMetricA(MetricBase):
    """Mock metric that returns 'score' key (for collision testing)."""

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        return {"score": 0.8, "unique_a": 0.9}, [{"score": 0.8, "unique_a": 0.9}] * len(outputs)


class MockMetricB(MetricBase):
    """Mock metric that also returns 'score' key (for collision testing)."""

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        return {"score": 0.7, "unique_b": 0.6}, [{"score": 0.7, "unique_b": 0.6}] * len(outputs)


class MockMetricC(MetricBase):
    """Mock metric with unique keys (no collision)."""

    def score(self, outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        return {"metric_c_score": 0.5}, [{"metric_c_score": 0.5}] * len(outputs)


def test_score_basic():
    """Test basic scoring with single metric."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
        {"output": "Shakespeare", "label": "William Shakespeare"},
    ]

    results = score(
        items=items,
        metrics=Accuracy,
        upload_results=False,
    )

    # Should return dict with both aggregates and items
    assert isinstance(results, dict)
    assert "aggregate_results" in results
    assert "item_results" in results
    assert len(results["aggregate_results"]) == 1
    assert "accuracy" in results["aggregate_results"][0]
    assert "dataset" in results["aggregate_results"][0]
    assert len(results["item_results"]) == 3


def test_score_multiple_metrics():
    """Test scoring with multiple metrics."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
        {"output": "Shakespeare", "label": "Shakespeare"},
    ]

    # Test with single metric for now (Precision not implemented yet)
    results = score(
        items=items,
        metrics=[Accuracy],
        upload_results=False,
    )

    # Should have results with the metric
    assert "accuracy" in results["aggregate_results"][0]


def test_score_with_input_key():
    """Test scoring with optional input key present."""
    items = [
        {"input": "What is the capital of France?", "output": "Paris", "label": "Paris"},
        {"input": "What is 2+2?", "output": "4", "label": "4"},
        {"input": "Who wrote Romeo and Juliet?", "output": "Shakespeare", "label": "Shakespeare"},
    ]

    results = score(
        items=items,
        metrics=Accuracy,
        upload_results=False,
    )

    # Should work the same whether input is present or not
    assert "accuracy" in results["aggregate_results"][0]
    # Items should have input field
    assert "input" in results["item_results"][0]


def test_score_mixed_input_keys():
    """Test scoring where some items have input key, some don't."""
    items = [
        {"input": "What is the capital of France?", "output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},  # No input key
        {"input": "Who wrote Romeo and Juliet?", "output": "Shakespeare", "label": "Shakespeare"},
    ]

    results = score(
        items=items,
        metrics=Accuracy,
        upload_results=False,
    )

    # Should handle mixed presence of input key
    assert "accuracy" in results["aggregate_results"][0]
    assert "input" in results["item_results"][0]
    assert "input" not in results["item_results"][1]


def test_score_with_hyperparameters():
    """Test scoring with hyperparameter metadata."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
    ]

    # Provide hyperparameter metadata as a single dict
    hyperparams = {"temperature": 0.7, "model": "gpt-4"}

    results = score(
        items=items,
        metrics=Accuracy,
        hyperparameters=hyperparams,
        upload_results=False,
    )

    # Should have hyperparameters in result
    assert results["aggregate_results"][0]["temperature"] == 0.7
    assert results["aggregate_results"][0]["model"] == "gpt-4"


def test_score_with_dataset_name():
    """Test scoring with custom dataset name."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = score(
        items=items,
        metrics=Accuracy,
        dataset_name="my_custom_dataset",
        upload_results=False,
    )

    # Dataset name should appear in results
    assert results["aggregate_results"][0]["dataset"] == "my_custom_dataset"


def test_score_with_string_metric():
    """Test scoring with metric specified as string."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = score(
        items=items,
        metrics="accuracy",  # String name instead of class
        upload_results=False,
    )

    assert "accuracy" in results["aggregate_results"][0]


def test_score_empty_items():
    """Test that empty items list raises error."""
    with pytest.raises(ParameterValidationError, match="items list cannot be empty"):
        score(
            items=[],
            metrics=Accuracy,
            upload_results=False,
        )


def test_score_invalid_items_not_list():
    """Test that non-list items raises error."""
    with pytest.raises(ParameterValidationError, match="items must be a list"):
        score(
            items="not a list",  # type: ignore
            metrics=Accuracy,
            upload_results=False,
        )


def test_score_invalid_items_not_dict():
    """Test that items containing non-dicts raises error."""
    with pytest.raises(ParameterValidationError, match="not a dict"):
        score(
            items=["not a dict", "also not a dict"],  # type: ignore
            metrics=Accuracy,
            upload_results=False,
        )


def test_score_missing_output_key():
    """Test that missing 'output' key raises error."""
    items = [
        {"label": "Paris"},  # Missing 'output'
    ]

    with pytest.raises(ParameterValidationError, match="missing required 'output' key"):
        score(
            items=items,
            metrics=Accuracy,
            upload_results=False,
        )


def test_score_missing_label_key():
    """Test that missing 'label' key raises error."""
    items = [
        {"output": "Paris"},  # Missing 'label'
    ]

    with pytest.raises(ParameterValidationError, match="missing required 'label' key"):
        score(
            items=items,
            metrics=Accuracy,
            upload_results=False,
        )


def test_score_invalid_metric():
    """Test that invalid metric type raises error."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    with pytest.raises(ParameterValidationError, match="Invalid metric type"):
        score(
            items=items,
            metrics=123,  # type: ignore  # Invalid metric type
            upload_results=False,
        )


def test_score_upload_requires_ids():
    """Test that upload requires experiment_id and project_id."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    with pytest.raises(
        ParameterValidationError,
        match="experiment_id and project_id are required",
    ):
        score(
            items=items,
            metrics=Accuracy,
            upload_results=True,
            # Missing experiment_id and project_id
        )


def test_score_with_metadata():
    """Test scoring with metadata."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = score(
        items=items,
        metrics=Accuracy,
        metadata={"version": "1.0", "note": "test run"},
        upload_results=False,
    )

    # Metadata doesn't appear directly in results, but shouldn't cause errors
    assert "accuracy" in results["aggregate_results"][0]


def test_score_hyperparameters_must_be_dict():
    """Test that hyperparameters must be a dict, not a list."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    # Hyperparameters as list should raise error
    with pytest.raises(ParameterValidationError, match="hyperparameters must be a dict"):
        score(
            items=items,
            metrics=Accuracy,
            hyperparameters=[{"temperature": 0.5}, {"temperature": 0.7}],  # type: ignore
            upload_results=False,
        )


# --- Metric key collision tests ---


def test_score_multiple_metrics_with_colliding_keys():
    """Test that colliding keys get suffixed with metric name."""
    items = [
        {"output": "a", "label": "a"},
        {"output": "b", "label": "b"},
    ]

    results = score(
        items=items,
        metrics=[MockMetricA(), MockMetricB()],
        upload_results=False,
    )

    aggregate = results["aggregate_results"][0]

    # Colliding 'score' key should be suffixed with metric names
    assert "score_mockmetrica" in aggregate
    assert "score_mockmetricb" in aggregate
    assert aggregate["score_mockmetrica"] == 0.8
    assert aggregate["score_mockmetricb"] == 0.7

    # Unique keys should NOT have suffix
    assert "unique_a" in aggregate
    assert "unique_b" in aggregate
    assert aggregate["unique_a"] == 0.9
    assert aggregate["unique_b"] == 0.6

    # Item results should also have suffixed colliding keys
    item = results["item_results"][0]
    assert "score_mockmetrica" in item
    assert "score_mockmetricb" in item
    assert "unique_a" in item
    assert "unique_b" in item


def test_score_multiple_metrics_no_collision():
    """Test that unique keys remain unsuffixed when no collision."""
    items = [
        {"output": "a", "label": "a"},
    ]

    results = score(
        items=items,
        metrics=[MockMetricA(), MockMetricC()],
        upload_results=False,
    )

    aggregate = results["aggregate_results"][0]

    # No collision for these keys, so no suffix
    assert "score" in aggregate
    assert "unique_a" in aggregate
    assert "metric_c_score" in aggregate

    # Should NOT have suffixed versions
    assert "score_mockmetrica" not in aggregate
    assert "score_mockmetricc" not in aggregate


def test_score_multiple_metrics_item_level_collision():
    """Test collision handling specifically for item-level dict scores."""
    items = [
        {"output": "x", "label": "x"},
        {"output": "y", "label": "y"},
    ]

    results = score(
        items=items,
        metrics=[MockMetricA(), MockMetricB()],
        upload_results=False,
    )

    # Check all items have properly suffixed keys
    for item in results["item_results"]:
        assert "score_mockmetrica" in item
        assert "score_mockmetricb" in item
        assert item["score_mockmetrica"] == 0.8
        assert item["score_mockmetricb"] == 0.7
        # Unique keys remain unsuffixed
        assert "unique_a" in item
        assert "unique_b" in item


def test_score_three_metrics_partial_collision():
    """Test with three metrics where only two have colliding keys."""
    items = [
        {"output": "test", "label": "test"},
    ]

    results = score(
        items=items,
        metrics=[MockMetricA(), MockMetricB(), MockMetricC()],
        upload_results=False,
    )

    aggregate = results["aggregate_results"][0]

    # 'score' collides between A and B, so both get suffixed
    assert "score_mockmetrica" in aggregate
    assert "score_mockmetricb" in aggregate
    # 'score' without suffix should NOT exist (it was a collision)
    assert "score" not in aggregate

    # Unique keys from all three metrics should NOT have suffix
    assert "unique_a" in aggregate
    assert "unique_b" in aggregate
    assert "metric_c_score" in aggregate
