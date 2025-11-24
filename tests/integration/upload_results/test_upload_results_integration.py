"""
Integration tests for trismik result upload functionality.

These tests make real API calls to the trismik service and do not use mocks.
They require a valid TRISMIK_API_KEY environment variable to run.
"""

import os
import uuid

import pytest

from scorebook.dashboard.create_project import create_project
from scorebook.dashboard.upload_results import upload_result, upload_result_async


@pytest.fixture
def test_api_key() -> str:
    """Get test API key from environment."""
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        pytest.skip("TRISMIK_API_KEY environment variable not set")
    return api_key


@pytest.fixture(autouse=True)
def ensure_logged_in(test_api_key, monkeypatch):
    """Ensure we're logged in for all tests."""
    monkeypatch.setenv("TRISMIK_API_KEY", test_api_key)


@pytest.fixture
def test_project():
    """Create a test project for uploads."""
    project_name = f"test-upload-project-{uuid.uuid4().hex[:8]}"
    project = create_project(name=project_name, description="Integration test project")
    return project


@pytest.fixture
def sample_run_result():
    """Create a sample run result for testing."""
    return {
        "aggregate_results": [
            {
                "accuracy": 0.85,
                "f1_score": 0.82,
                "dataset": "test_dataset",
                "run_completed": True,
            }
        ],
        "item_results": [
            {
                "id": "item_001",
                "input": "What is 2+2?",
                "output": "4",
                "label": "4",
                "accuracy": 1.0,
                "f1_score": 0.9,
                "dataset": "test_dataset",
            },
            {
                "id": "item_002",
                "input": "What is the capital of France?",
                "output": "Paris",
                "label": "Paris",
                "accuracy": 1.0,
                "f1_score": 0.85,
                "dataset": "test_dataset",
            },
        ],
    }


@pytest.fixture
def sample_run_result_with_booleans():
    """Create a sample run result with boolean metrics for normalization testing."""
    return {
        "aggregate_results": [
            {"exact_match": True, "is_correct": False, "dataset": "test_dataset"}
        ],
        "item_results": [
            {
                "id": "item_001",
                "input": "test input",
                "output": "test output",
                "label": "test label",
                "is_correct": True,
                "exact_match": True,
                "dataset": "test_dataset",
            },
        ],
    }


def test_upload_result_basic(sample_run_result, test_project):
    """Test uploading a basic result."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"

    run_id = upload_result(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
    )

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_upload_result_with_dataset_name(sample_run_result, test_project):
    """Test uploading a result with explicit dataset name."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"
    dataset_name = "custom_dataset"

    run_id = upload_result(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
        dataset_name=dataset_name,
    )

    assert run_id is not None


def test_upload_result_with_hyperparameters(sample_run_result, test_project):
    """Test uploading a result with hyperparameters."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"
    hyperparameters = {"temperature": 0.7, "max_tokens": 100}

    run_id = upload_result(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
        hyperparameters=hyperparameters,
    )

    assert run_id is not None


def test_upload_result_with_metadata(sample_run_result, test_project):
    """Test uploading a result with metadata."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"
    metadata = {"model": "gpt-4", "notes": "Integration test"}

    run_id = upload_result(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
        metadata=metadata,
    )

    assert run_id is not None


def test_upload_result_with_all_parameters(sample_run_result, test_project):
    """Test uploading a result with all optional parameters."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"

    run_id = upload_result(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
        dataset_name="complete_dataset",
        hyperparameters={"temperature": 0.8},
        metadata={"model": "test-model"},
        model_name="full-test-model",
    )

    assert run_id is not None
    assert isinstance(run_id, str)


def test_upload_result_boolean_normalization(sample_run_result_with_booleans, test_project):
    """Test that boolean metrics are properly normalized to floats."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"

    # Should not raise an error even though metrics contain booleans
    run_id = upload_result(
        run_result=sample_run_result_with_booleans,
        experiment_id=experiment_id,
        project_id=test_project.id,
    )

    assert run_id is not None


def test_upload_multiple_results_same_experiment(sample_run_result, test_project):
    """Test uploading multiple results to the same experiment."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"

    run_ids = []
    for i in range(3):
        run_id = upload_result(
            run_result=sample_run_result,
            experiment_id=experiment_id,
            project_id=test_project.id,
            model_name=f"model-variant-{i}",
        )
        run_ids.append(run_id)

    # Verify all uploads succeeded with unique IDs
    assert len(run_ids) == 3
    assert all(isinstance(rid, str) for rid in run_ids)
    assert len(set(run_ids)) == 3


@pytest.mark.asyncio
async def test_upload_result_async(sample_run_result, test_project):
    """Test uploading a result asynchronously."""
    experiment_id = f"test-experiment-async-{uuid.uuid4().hex[:8]}"

    run_id = await upload_result_async(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
    )

    assert run_id is not None
    assert isinstance(run_id, str)


@pytest.mark.asyncio
async def test_upload_result_async_with_parameters(sample_run_result, test_project):
    """Test uploading asynchronously with parameters."""
    experiment_id = f"test-experiment-async-{uuid.uuid4().hex[:8]}"

    run_id = await upload_result_async(
        run_result=sample_run_result,
        experiment_id=experiment_id,
        project_id=test_project.id,
        dataset_name="async_dataset",
        hyperparameters={"temp": 0.9},
        model_name="async-model",
    )

    assert run_id is not None


@pytest.mark.asyncio
async def test_upload_multiple_results_async_concurrent(sample_run_result, test_project):
    """Test uploading multiple results concurrently to different experiments."""
    import asyncio

    # Use different experiment IDs to avoid server-side race conditions
    tasks = [
        upload_result_async(
            run_result=sample_run_result,
            experiment_id=f"test-experiment-concurrent-{i}-{uuid.uuid4().hex[:8]}",
            project_id=test_project.id,
            model_name=f"concurrent-model-{i}",
        )
        for i in range(3)
    ]

    run_ids = await asyncio.gather(*tasks)

    # Verify all uploads succeeded with unique IDs
    assert len(run_ids) == 3
    assert all(isinstance(rid, str) for rid in run_ids)
    assert len(set(run_ids)) == 3


def test_upload_result_with_invalid_project_id(sample_run_result):
    """Test that uploading with an invalid project ID raises an error."""
    experiment_id = f"test-experiment-{uuid.uuid4().hex[:8]}"
    invalid_project_id = "invalid_project_id_12345"

    with pytest.raises(Exception):
        upload_result(
            run_result=sample_run_result,
            experiment_id=experiment_id,
            project_id=invalid_project_id,
        )
