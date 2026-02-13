"""
Integration tests for adaptive replay functionality.

These tests make real API calls to the trismik service and do not use mocks.
They require a valid TRISMIK_API_KEY environment variable or .env file to run.
"""

import os
import random
import uuid
from typing import Any, Dict, List, Union

import pytest
from dotenv import load_dotenv

from scorebook import evaluate, evaluate_async, replay, replay_async
from scorebook.dashboard.create_project import create_project

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def test_api_key() -> str:
    """Get test API key from environment or .env file."""
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        pytest.fail(
            "TRISMIK_API_KEY not found. Set it as an environment variable or in a .env file."
        )
    return api_key


@pytest.fixture(autouse=True)
def ensure_logged_in(test_api_key, monkeypatch):
    """Ensure we're logged in for all tests."""
    monkeypatch.setenv("TRISMIK_API_KEY", test_api_key)


@pytest.fixture
def test_project():
    """Create a test project for replay tests."""
    project_name = f"test-replay-project-{uuid.uuid4().hex[:8]}"
    project = create_project(name=project_name, description="Replay integration tests")
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


def test_replay_sync(test_project):
    """Test synchronous replay functionality."""
    experiment_id = f"test-replay-sync-{uuid.uuid4().hex[:8]}"

    try:
        # First, run an original adaptive evaluation
        original_results = evaluate(
            inference=random_letter_inference,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
            return_dict=False,
        )

        # Get the run ID from the original evaluation
        assert len(original_results.run_results) > 0, "Original run should have results"
        original_run_id = original_results.run_results[0].run_id
        assert original_run_id is not None, "Original run should return a run_id"

        # Replay with the same inference function
        replay_result = replay(
            inference=random_letter_inference,
            previous_run_id=original_run_id,
            experiment_id=experiment_id,
            project_id=test_project.id,
            metadata={"model": "test-model-v2", "test": "replay-sync"},
        )

        # Verify replay result
        assert replay_result is not None
        assert isinstance(replay_result, dict)

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


@pytest.mark.asyncio
async def test_replay_async_basic(test_project):
    """Test basic async replay functionality."""
    experiment_id = f"test-replay-async-{uuid.uuid4().hex[:8]}"

    try:
        # First, run an original adaptive evaluation
        original_results = await evaluate_async(
            inference=random_letter_inference_async,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
            return_dict=False,
        )

        # Get the run ID from the original evaluation
        assert len(original_results.run_results) > 0, "Original run should have results"
        original_run_id = original_results.run_results[0].run_id
        assert original_run_id is not None, "Original run should return a run_id"

        # Replay with a different "model"
        replay_result = await replay_async(
            inference=random_letter_inference_async,
            previous_run_id=original_run_id,
            experiment_id=experiment_id,
            project_id=test_project.id,
            metadata={"model": "test-model-v2", "test": "replay-async"},
        )

        # Verify replay result
        assert replay_result is not None
        assert isinstance(replay_result, dict)

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


@pytest.mark.asyncio
async def test_replay_async_with_return_object(test_project):
    """Test async replay with return_dict=False."""
    experiment_id = f"test-replay-obj-{uuid.uuid4().hex[:8]}"

    try:
        # First, run an original adaptive evaluation
        original_results = await evaluate_async(
            inference=random_letter_inference_async,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
            return_dict=False,
        )

        original_run_id = original_results.run_results[0].run_id
        assert original_run_id is not None

        # Replay with return_dict=False to get AdaptiveReplayRunResult object
        replay_result = await replay_async(
            inference=random_letter_inference_async,
            previous_run_id=original_run_id,
            experiment_id=experiment_id,
            project_id=test_project.id,
            return_dict=False,
        )

        # Verify we get an AdaptiveReplayRunResult object
        assert replay_result is not None
        assert hasattr(replay_result, "run_spec")
        assert hasattr(replay_result, "run_completed")
        assert hasattr(replay_result, "scores")
        assert hasattr(replay_result, "run_id")
        assert hasattr(replay_result, "replay_of_run")
        assert replay_result.run_completed is True

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise


@pytest.mark.asyncio
async def test_replay_async_with_hyperparameters(test_project):
    """Test async replay with hyperparameters."""
    experiment_id = f"test-replay-hp-{uuid.uuid4().hex[:8]}"

    async def inference_with_params(
        items: Union[Dict[str, Any], List[Dict[str, Any]]], temperature: float = 0.7
    ) -> Union[str, List[str]]:
        """Inference with temperature parameter."""
        return await random_letter_inference_async(items)

    try:
        # First, run an original adaptive evaluation
        original_results = await evaluate_async(
            inference=inference_with_params,
            datasets="trismik/headQA:adaptive",
            split="test",
            experiment_id=experiment_id,
            project_id=test_project.id,
            return_dict=False,
        )

        original_run_id = original_results.run_results[0].run_id
        assert original_run_id is not None

        # Replay with different hyperparameters
        replay_result = await replay_async(
            inference=inference_with_params,
            previous_run_id=original_run_id,
            experiment_id=experiment_id,
            project_id=test_project.id,
            hyperparameters={"temperature": 0.9},
        )

        assert replay_result is not None
        assert isinstance(replay_result, dict)

    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            pytest.skip(f"Adaptive dataset not available: {e}")
        raise
