"""
Integration tests for trismik adaptive testing functionality.

These tests make real API calls to the trismik service and do not use mocks.
They require a valid TRISMIK_API_KEY environment variable to run.
"""

import os
import random
import uuid
from typing import Any, Dict, List, Union

import pytest

from scorebook import evaluate, evaluate_async
from scorebook.dashboard.create_project import create_project


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
    """Create a test project for adaptive evaluations."""
    project_name = f"test-adaptive-project-{uuid.uuid4().hex[:8]}"
    project = create_project(name=project_name, description="Adaptive testing integration tests")
    return project


@pytest.fixture
def test_project_sync():
    """Create a test project synchronously for async tests."""
    project_name = f"test-adaptive-project-async-{uuid.uuid4().hex[:8]}"
    project = create_project(
        name=project_name, description="Async adaptive testing integration tests"
    )
    return project


def random_letter_inference(
    items: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[str, List[str]]:
    """Return a random letter A-D for inference."""
    if isinstance(items, dict):
        return random.choice(["A", "B", "C", "D"])

    return [random.choice(["A", "B", "C", "D"]) for _ in items]


async def random_letter_inference_async(
    items: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[str, List[str]]:
    """Return a random letter A-D for async inference."""
    if isinstance(items, dict):
        return random.choice(["A", "B", "C", "D"])

    return [random.choice(["A", "B", "C", "D"]) for _ in items]


def test_evaluate_adaptive(test_project):
    """Test synchronous adaptive evaluation."""
    experiment_id = f"test-adaptive-{uuid.uuid4().hex[:8]}"

    try:
        results = evaluate(
            inference=random_letter_inference,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
        )

        # Verify results structure
        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0

        # Verify adaptive run result is a dict with expected fields
        result = results[0]
        assert isinstance(result, dict)
        assert "dataset" in result
        assert result["dataset"] == "trismik/headQA:adaptive"

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


def test_evaluate_adaptive_with_split(test_project):
    """Test adaptive evaluation with explicit split."""
    experiment_id = f"test-adaptive-split-{uuid.uuid4().hex[:8]}"

    try:
        results = evaluate(
            inference=random_letter_inference,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
        )

        assert results is not None
        assert len(results) > 0

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset or split not available: {e}")
        raise


def test_evaluate_adaptive_with_hyperparameters(test_project):
    """Test adaptive evaluation with hyperparameters."""
    experiment_id = f"test-adaptive-hp-{uuid.uuid4().hex[:8]}"

    def inference_with_params(
        items: Union[Dict[str, Any], List[Dict[str, Any]]], temperature: float = 0.7
    ) -> Union[str, List[str]]:
        """Inference with temperature parameter."""
        return random_letter_inference(items)

    try:
        results = evaluate(
            inference=inference_with_params,
            datasets="trismik/headQA:adaptive",
            split="test",
            hyperparameters={"temperature": 0.9},
            experiment_id=experiment_id,
            project_id=test_project.id,
        )

        assert results is not None
        assert len(results) > 0

    except Exception as e:
        if "not found" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


@pytest.mark.asyncio
async def test_evaluate_async_adaptive(test_project_sync):
    """Test asynchronous adaptive evaluation."""
    project = test_project_sync
    experiment_id = f"test-adaptive-async-{uuid.uuid4().hex[:8]}"

    try:
        results = await evaluate_async(
            inference=random_letter_inference_async,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=project.id,
        )

        assert results is not None
        assert len(results) > 0
        assert isinstance(results[0], dict)

    except Exception as e:
        if "not found" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


@pytest.mark.asyncio
async def test_evaluate_async_adaptive_with_split(test_project_sync):
    """Test async adaptive evaluation with split."""
    project = test_project_sync
    experiment_id = f"test-adaptive-async-split-{uuid.uuid4().hex[:8]}"

    try:
        results = await evaluate_async(
            inference=random_letter_inference_async,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=project.id,
        )

        assert results is not None
        assert len(results) > 0

    except Exception as e:
        if "not found" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise
