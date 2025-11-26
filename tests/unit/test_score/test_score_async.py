"""Unit tests for the await score_async() function."""

import pytest

from scorebook import score_async
from scorebook.exceptions import ParameterValidationError
from scorebook.metrics.accuracy import Accuracy


@pytest.mark.asyncio
async def test_score_async_basic():
    """Test async basic scoring with single metric."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
        {"output": "Shakespeare", "label": "William Shakespeare"},
    ]

    results = await score_async(
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


@pytest.mark.asyncio
async def test_score_async_multiple_metrics():
    """Test async scoring with multiple metrics."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
        {"output": "Shakespeare", "label": "Shakespeare"},
    ]

    # Test with single metric for now (Precision not implemented yet)
    results = await score_async(
        items=items,
        metrics=[Accuracy],
        upload_results=False,
    )

    # Should have results with the metric
    assert "accuracy" in results["aggregate_results"][0]


@pytest.mark.asyncio
async def test_score_async_with_input_key():
    """Test async scoring with optional input key present."""
    items = [
        {"input": "What is the capital of France?", "output": "Paris", "label": "Paris"},
        {"input": "What is 2+2?", "output": "4", "label": "4"},
        {"input": "Who wrote Romeo and Juliet?", "output": "Shakespeare", "label": "Shakespeare"},
    ]

    results = await score_async(
        items=items,
        metrics=Accuracy,
        upload_results=False,
    )

    # Should work the same whether input is present or not
    assert "accuracy" in results["aggregate_results"][0]
    # Items should have input field
    assert "input" in results["item_results"][0]


@pytest.mark.asyncio
async def test_score_async_mixed_input_keys():
    """Test async scoring where some items have input key, some don't."""
    items = [
        {"input": "What is the capital of France?", "output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},  # No input key
        {"input": "Who wrote Romeo and Juliet?", "output": "Shakespeare", "label": "Shakespeare"},
    ]

    results = await score_async(
        items=items,
        metrics=Accuracy,
        upload_results=False,
    )

    # Should handle mixed presence of input key
    assert "accuracy" in results["aggregate_results"][0]
    assert "input" in results["item_results"][0]
    assert "input" not in results["item_results"][1]


@pytest.mark.asyncio
async def test_score_async_with_hyperparameters():
    """Test async scoring with hyperparameter metadata."""
    items = [
        {"output": "Paris", "label": "Paris"},
        {"output": "4", "label": "4"},
    ]

    # Provide hyperparameter metadata as a single dict
    hyperparams = {"temperature": 0.7, "model": "gpt-4"}

    results = await score_async(
        items=items,
        metrics=Accuracy,
        hyperparameters=hyperparams,
        upload_results=False,
    )

    # Should have hyperparameters in result
    assert results["aggregate_results"][0]["temperature"] == 0.7
    assert results["aggregate_results"][0]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_score_async_with_dataset_name():
    """Test async scoring with custom dataset name."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = await score_async(
        items=items,
        metrics=Accuracy,
        dataset_name="my_custom_dataset",
        upload_results=False,
    )

    # Dataset name should appear in results
    assert results["aggregate_results"][0]["dataset"] == "my_custom_dataset"


@pytest.mark.asyncio
async def test_score_async_with_string_metric():
    """Test async scoring with metric specified as string."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = await score_async(
        items=items,
        metrics="accuracy",  # String name instead of class
        upload_results=False,
    )

    assert "accuracy" in results["aggregate_results"][0]


@pytest.mark.asyncio
async def test_score_async_empty_items():
    """Test async that empty items list raises error."""
    with pytest.raises(ParameterValidationError, match="items list cannot be empty"):
        await score_async(
            items=[],
            metrics=Accuracy,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_invalid_items_not_list():
    """Test async that non-list items raises error."""
    with pytest.raises(ParameterValidationError, match="items must be a list"):
        await score_async(
            items="not a list",  # type: ignore
            metrics=Accuracy,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_invalid_items_not_dict():
    """Test async that items containing non-dicts raises error."""
    with pytest.raises(ParameterValidationError, match="not a dict"):
        await score_async(
            items=["not a dict", "also not a dict"],  # type: ignore
            metrics=Accuracy,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_missing_output_key():
    """Test async that missing 'output' key raises error."""
    items = [
        {"label": "Paris"},  # Missing 'output'
    ]

    with pytest.raises(ParameterValidationError, match="missing required 'output' key"):
        await score_async(
            items=items,
            metrics=Accuracy,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_missing_label_key():
    """Test async that missing 'label' key raises error."""
    items = [
        {"output": "Paris"},  # Missing 'label'
    ]

    with pytest.raises(ParameterValidationError, match="missing required 'label' key"):
        await score_async(
            items=items,
            metrics=Accuracy,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_invalid_metric():
    """Test async that invalid metric type raises error."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    with pytest.raises(ParameterValidationError, match="Invalid metric type"):
        await score_async(
            items=items,
            metrics=123,  # type: ignore  # Invalid metric type
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_score_async_upload_requires_ids():
    """Test async that upload requires experiment_id and project_id."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    with pytest.raises(
        ParameterValidationError,
        match="experiment_id and project_id are required",
    ):
        await score_async(
            items=items,
            metrics=Accuracy,
            upload_results=True,
            # Missing experiment_id and project_id
        )


@pytest.mark.asyncio
async def test_score_async_with_metadata():
    """Test async scoring with metadata."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    results = await score_async(
        items=items,
        metrics=Accuracy,
        metadata={"version": "1.0", "note": "test run"},
        upload_results=False,
    )

    # Metadata doesn't appear directly in results, but shouldn't cause errors
    assert "accuracy" in results["aggregate_results"][0]


@pytest.mark.asyncio
async def test_score_async_hyperparameters_must_be_dict():
    """Test async that hyperparameters must be a dict, not a list."""
    items = [
        {"output": "Paris", "label": "Paris"},
    ]

    # Hyperparameters as list should raise error
    with pytest.raises(ParameterValidationError, match="hyperparameters must be a dict"):
        await score_async(
            items=items,
            metrics=Accuracy,
            hyperparameters=[{"temperature": 0.5}, {"temperature": 0.7}],  # type: ignore
            upload_results=False,
        )
