"""Mock Trismik clients and types for testing without network access.

This module provides mock implementations of Trismik clients and types
that can be used in tests to avoid network dependencies.
"""

from typing import Any, Dict, List


class MockAdaptiveTestScore:
    """Mock for trismik.types.AdaptiveTestScore."""

    def __init__(self, theta: float = 0.75, std_error: float = 0.15):
        """Initialize mock adaptive test score."""
        self.theta = theta
        self.std_error = std_error


class MockTrismikRunResults:
    """Mock for trismik.types.TrismikRunResults."""

    def __init__(self, run_id: str = "mock_run_123", theta: float = 0.75, std_error: float = 0.15):
        """Initialize mock trismik run results."""
        self.run_id = run_id
        self.score = MockAdaptiveTestScore(theta, std_error)
        self.responses = []
        self.scores = {"overall": MockAdaptiveTestScore(theta, std_error)}


class MockTrismikDatasetInfo:
    """Mock for trismik.types.TrismikDatasetInfo."""

    def __init__(
        self,
        test_id: str,
        splits: List[str],
        is_adaptive: bool = True,
        name: str = None,
    ):
        """Initialize mock dataset info."""
        self.id = test_id
        self.name = name or test_id
        self.isAdaptive = is_adaptive
        self.splits = splits
        self.datacard = None


class MockTrismikAsyncClient:
    """Mock for TrismikAsyncClient."""

    def __init__(self, dataset_splits: Dict[str, List[str]] = None):
        """Initialize mock client.

        Args:
            dataset_splits: Dict mapping test_id to available splits.
                          Defaults to {"test_dataset:adaptive": ["validation", "test"]}
        """
        self.dataset_splits = dataset_splits or {"test_dataset:adaptive": ["validation", "test"]}
        self.run_calls = []  # Track calls to run()

    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""

    async def get_dataset_info(self, test_id: str) -> MockTrismikDatasetInfo:
        """Mock get_dataset_info."""
        splits = self.dataset_splits.get(test_id, ["validation", "test"])
        return MockTrismikDatasetInfo(test_id, splits)

    async def run(
        self,
        test_id: str,
        split: str,
        project_id: str,
        experiment: str,
        run_metadata: Any,
        item_processor: Any,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> MockTrismikRunResults:
        """Mock run method."""
        # Track the call
        self.run_calls.append(
            {
                "test_id": test_id,
                "split": split,
                "project_id": project_id,
                "experiment": experiment,
            }
        )

        # Return mock results
        return MockTrismikRunResults(run_id=f"run_{test_id}_{split}", theta=0.75, std_error=0.15)


class MockTrismikSyncClient:
    """Mock for TrismikClient (sync version)."""

    def __init__(self, dataset_splits: Dict[str, List[str]] = None):
        """Initialize mock client.

        Args:
            dataset_splits: Dict mapping test_id to available splits.
                          Defaults to {"test_dataset:adaptive": ["validation", "test"]}
        """
        self.dataset_splits = dataset_splits or {"test_dataset:adaptive": ["validation", "test"]}
        self.run_calls = []  # Track calls to run()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args):
        """Exit context."""

    def get_dataset_info(self, test_id: str) -> MockTrismikDatasetInfo:
        """Mock get_dataset_info."""
        splits = self.dataset_splits.get(test_id, ["validation", "test"])
        return MockTrismikDatasetInfo(test_id, splits)

    def run(
        self,
        test_id: str,
        split: str,
        project_id: str,
        experiment: str,
        run_metadata: Any,
        item_processor: Any,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> MockTrismikRunResults:
        """Mock run method."""
        # Track the call
        self.run_calls.append(
            {
                "test_id": test_id,
                "split": split,
                "project_id": project_id,
                "experiment": experiment,
            }
        )

        # Return mock results
        return MockTrismikRunResults(run_id=f"run_{test_id}_{split}", theta=0.75, std_error=0.15)
