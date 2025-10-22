"""Unit tests for the score() function."""

import pytest

from scorebook import score
from scorebook.exceptions import ParameterValidationError
from scorebook.metrics import Accuracy


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
        dataset="my_custom_dataset",
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
