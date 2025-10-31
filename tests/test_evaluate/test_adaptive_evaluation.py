"""Tests for adaptive evaluation functionality.

This test module focuses on adaptive evaluation within scorebook,
mocking the trismik client to avoid integration tests.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from scorebook.evaluate import evaluate_async
from scorebook.exceptions import ParameterValidationError, ScoreBookError
from scorebook.types import AdaptiveEvalRunResult, EvalResult
from tests.fixtures.mock_trismik import MockTrismikAsyncClient, MockTrismikSyncClient


def create_simple_async_inference():
    """Create simple async inference function for testing."""

    async def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Return 'A' for all inputs."""
        await asyncio.sleep(0.001)  # Simulate async work
        return ["A" for _ in inputs]

    return inference


@pytest.fixture
def mock_trismik_client():
    """Fixture that provides a mock trismik client."""
    return MockTrismikAsyncClient()


@pytest.fixture
def mock_trismik_client_with_custom_splits():
    """Fixture for trismik client with custom split configurations."""

    def _create_client(dataset_splits: Dict[str, List[str]]):
        return MockTrismikAsyncClient(dataset_splits)

    return _create_client


def test_adaptive_dataset_parsing_no_split():
    """Test parsing adaptive dataset string without split."""
    from scorebook.evaluate.evaluate_helpers import prepare_datasets

    datasets = prepare_datasets("my_dataset:adaptive")

    assert len(datasets) == 1
    from scorebook.types import AdaptiveEvalDataset

    assert isinstance(datasets[0], AdaptiveEvalDataset)
    assert datasets[0].name == "my_dataset:adaptive"
    assert datasets[0].split is None


def test_adaptive_dataset_parsing_with_split():
    """Test parsing adaptive dataset string with split."""
    from scorebook.evaluate.evaluate_helpers import prepare_datasets

    datasets = prepare_datasets("my_dataset:adaptive:validation")

    assert len(datasets) == 1
    from scorebook.types import AdaptiveEvalDataset

    assert isinstance(datasets[0], AdaptiveEvalDataset)
    assert datasets[0].name == "my_dataset:adaptive"
    assert datasets[0].split == "validation"


def test_adaptive_dataset_parsing_multiple_splits():
    """Test parsing multiple adaptive datasets with different splits."""
    from scorebook.evaluate.evaluate_helpers import prepare_datasets

    datasets = prepare_datasets(
        [
            "dataset_a:adaptive:validation",
            "dataset_b:adaptive:test",
            "dataset_c:adaptive",  # No split
        ]
    )

    assert len(datasets) == 3

    # Check first dataset
    assert datasets[0].name == "dataset_a:adaptive"
    assert datasets[0].split == "validation"

    # Check second dataset
    assert datasets[1].name == "dataset_b:adaptive"
    assert datasets[1].split == "test"

    # Check third dataset
    assert datasets[2].name == "dataset_c:adaptive"
    assert datasets[2].split is None


def test_adaptive_dataset_parsing_invalid_format():
    """Test that invalid adaptive dataset format raises error."""
    from scorebook.evaluate.evaluate_helpers import prepare_datasets

    with pytest.raises(ParameterValidationError) as exc_info:
        prepare_datasets("my_dataset:adaptive:split:extra")

    assert "Invalid adaptive dataset format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_resolve_split_user_specified():
    """Test that user-specified split is validated and used."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_async

    client = MockTrismikAsyncClient({"test:adaptive": ["validation", "test", "train"]})

    # User specifies 'test' split
    result = await resolve_split_async("test:adaptive", "test", client)
    assert result == "test"


@pytest.mark.asyncio
async def test_resolve_split_fallback_to_validation():
    """Test that split falls back to 'validation' when not specified."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_async

    client = MockTrismikAsyncClient({"test:adaptive": ["validation", "test"]})

    # No split specified - should use validation
    result = await resolve_split_async("test:adaptive", None, client)
    assert result == "validation"


@pytest.mark.asyncio
async def test_resolve_split_fallback_to_test():
    """Test that split falls back to 'test' when validation doesn't exist."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_async

    client = MockTrismikAsyncClient({"test:adaptive": ["test", "train"]})  # No validation

    # No split specified - should use test
    result = await resolve_split_async("test:adaptive", None, client)
    assert result == "test"


@pytest.mark.asyncio
async def test_resolve_split_user_specified_invalid():
    """Test that invalid user-specified split raises error."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_async

    client = MockTrismikAsyncClient({"test:adaptive": ["validation", "test"]})

    # User specifies invalid split
    with pytest.raises(ScoreBookError) as exc_info:
        await resolve_split_async("test:adaptive", "invalid_split", client)

    assert "not found for dataset" in str(exc_info.value)
    assert "Available splits" in str(exc_info.value)


@pytest.mark.asyncio
async def test_resolve_split_no_valid_splits():
    """Test that error is raised when no validation or test split exists."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_async

    client = MockTrismikAsyncClient({"test:adaptive": ["train"]})  # Only train split

    # No split specified and no valid splits
    with pytest.raises(ScoreBookError) as exc_info:
        await resolve_split_async("test:adaptive", None, client)

    assert "No suitable split found" in str(exc_info.value)
    assert "Expected 'validation' or 'test'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adaptive_evaluation_with_explicit_split():
    """Test adaptive evaluation with explicitly specified split."""
    inference = create_simple_async_inference()
    mock_client = MockTrismikAsyncClient({"test_dataset:adaptive": ["validation", "test"]})

    # Patch both the creation and the actual evaluate_async imports
    with (
        patch(
            "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
            return_value=mock_client,
        ),
        patch(
            "scorebook.evaluate._async.evaluate_async.create_trismik_async_client",
            return_value=mock_client,
        ),
    ):
        results = await evaluate_async(
            inference,
            datasets="test_dataset:adaptive:test",  # Explicit 'test' split
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=False,
        )

    # Check results
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 1

    # Check that it's an adaptive run result
    run_result = results.run_results[0]
    assert isinstance(run_result, AdaptiveEvalRunResult)
    assert run_result.run_completed is True

    # Verify that the correct split was used
    assert len(mock_client.run_calls) == 1
    assert mock_client.run_calls[0]["split"] == "test"
    assert mock_client.run_calls[0]["test_id"] == "test_dataset:adaptive"


@pytest.mark.asyncio
async def test_adaptive_evaluation_with_fallback_split():
    """Test adaptive evaluation using fallback split (validation)."""
    inference = create_simple_async_inference()
    mock_client = MockTrismikAsyncClient({"test_dataset:adaptive": ["validation", "test"]})

    with (
        patch(
            "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
            return_value=mock_client,
        ),
        patch(
            "scorebook.evaluate._async.evaluate_async.create_trismik_async_client",
            return_value=mock_client,
        ),
    ):
        results = await evaluate_async(
            inference,
            datasets="test_dataset:adaptive",  # No split specified
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=False,
        )

    # Check results
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 1

    # Verify that the fallback split (validation) was used
    assert len(mock_client.run_calls) == 1
    assert mock_client.run_calls[0]["split"] == "validation"


@pytest.mark.asyncio
async def test_adaptive_evaluation_multiple_datasets_different_splits():
    """Test adaptive evaluation with multiple datasets having different splits."""
    inference = create_simple_async_inference()
    mock_client = MockTrismikAsyncClient(
        {
            "dataset_a:adaptive": ["validation", "test"],
            "dataset_b:adaptive": ["validation", "test"],
        }
    )

    with (
        patch(
            "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
            return_value=mock_client,
        ),
        patch(
            "scorebook.evaluate._async.evaluate_async.create_trismik_async_client",
            return_value=mock_client,
        ),
    ):
        results = await evaluate_async(
            inference,
            datasets=[
                "dataset_a:adaptive:validation",
                "dataset_b:adaptive:test",
            ],
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=False,
        )

    # Check results
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 2

    # Verify correct splits were used
    assert len(mock_client.run_calls) == 2
    assert mock_client.run_calls[0]["test_id"] == "dataset_a:adaptive"
    assert mock_client.run_calls[0]["split"] == "validation"
    assert mock_client.run_calls[1]["test_id"] == "dataset_b:adaptive"
    assert mock_client.run_calls[1]["split"] == "test"


@pytest.mark.asyncio
async def test_adaptive_evaluation_results_structure():
    """Test that adaptive evaluation results have correct structure."""
    inference = create_simple_async_inference()
    mock_client = MockTrismikAsyncClient({"test_dataset:adaptive": ["validation", "test"]})

    with (
        patch(
            "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
            return_value=mock_client,
        ),
        patch(
            "scorebook.evaluate._async.evaluate_async.create_trismik_async_client",
            return_value=mock_client,
        ),
    ):
        results = await evaluate_async(
            inference,
            datasets="test_dataset:adaptive:validation",
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=True,
            return_aggregates=True,
        )

    # Check that results are in dict format
    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert "dataset" in result
    assert result["dataset"] == "test_dataset:adaptive"
    assert "experiment_id" in result
    assert result["experiment_id"] == "test_exp"
    assert "project_id" in result
    assert result["project_id"] == "test_proj"

    # Check that scores were included
    assert "theta" in result or "overall" in result


@pytest.mark.asyncio
async def test_adaptive_evaluation_missing_experiment_id():
    """Test that adaptive evaluation requires experiment_id."""
    inference = create_simple_async_inference()

    with pytest.raises(ScoreBookError) as exc_info:
        await evaluate_async(
            inference,
            datasets="test_dataset:adaptive",
            # Missing experiment_id
            project_id="test_proj",
            return_dict=False,
        )

    assert "experiment_id and project_id are required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adaptive_evaluation_missing_project_id():
    """Test that adaptive evaluation requires project_id."""
    inference = create_simple_async_inference()

    with pytest.raises(ScoreBookError) as exc_info:
        await evaluate_async(
            inference,
            datasets="test_dataset:adaptive",
            experiment_id="test_exp",
            # Missing project_id
            return_dict=False,
        )

    assert "experiment_id and project_id are required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adaptive_evaluation_with_hyperparameters():
    """Test adaptive evaluation with hyperparameters."""
    inference = create_simple_async_inference()
    mock_client = MockTrismikAsyncClient({"test_dataset:adaptive": ["validation", "test"]})

    with (
        patch(
            "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
            return_value=mock_client,
        ),
        patch(
            "scorebook.evaluate._async.evaluate_async.create_trismik_async_client",
            return_value=mock_client,
        ),
    ):
        results = await evaluate_async(
            inference,
            datasets="test_dataset:adaptive:validation",
            hyperparameters={"temperature": 0.7},
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=False,
        )

    # Check that run completed
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 1
    assert results.run_results[0].run_completed is True

    # Check that hyperparameters are in the run spec
    assert results.run_results[0].run_spec.hyperparameter_config == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_adaptive_evaluation_failure_handling():
    """Test that adaptive evaluation handles failures gracefully."""
    inference = create_simple_async_inference()

    # Create a mock client that raises an error
    mock_client = MockTrismikAsyncClient()

    async def failing_run(*args, **kwargs):
        raise Exception("Mock trismik error")

    mock_client.run = failing_run

    with patch(
        "scorebook.evaluate.evaluate_helpers.create_trismik_async_client",
        return_value=mock_client,
    ):
        results = await evaluate_async(
            inference,
            datasets="test_dataset:adaptive",
            experiment_id="test_exp",
            project_id="test_proj",
            return_dict=False,
        )

    # Check that run failed but didn't raise
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 1
    assert results.run_results[0].run_completed is False


def test_sync_resolve_split_fallback():
    """Test sync version of resolve_split with fallback."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_sync

    client = MockTrismikSyncClient({"test:adaptive": ["validation", "test"]})

    # No split specified - should use validation
    result = resolve_split_sync("test:adaptive", None, client)
    assert result == "validation"


def test_sync_resolve_split_fallback_to_test():
    """Test sync version of resolve_split falls back to test."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_sync

    client = MockTrismikSyncClient({"test:adaptive": ["test", "train"]})  # No validation

    # No split specified - should use test
    result = resolve_split_sync("test:adaptive", None, client)
    assert result == "test"


def test_sync_resolve_split_user_specified():
    """Test sync version with user-specified split."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_sync

    client = MockTrismikSyncClient({"test:adaptive": ["validation", "test", "custom"]})

    # User specifies 'custom' split
    result = resolve_split_sync("test:adaptive", "custom", client)
    assert result == "custom"


def test_sync_resolve_split_invalid():
    """Test sync version with invalid split."""
    from scorebook.evaluate.evaluate_helpers import resolve_split_sync

    client = MockTrismikSyncClient({"test:adaptive": ["validation", "test"]})

    # User specifies invalid split
    with pytest.raises(ScoreBookError) as exc_info:
        resolve_split_sync("test:adaptive", "invalid", client)

    assert "not found for dataset" in str(exc_info.value)
