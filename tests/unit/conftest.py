"""Pytest configuration and fixtures for scorebook tests.

This file is automatically loaded by pytest and provides:
- Automatic mocking of HuggingFace dataset downloads
- Shared fixtures for all tests
- Test markers and configuration
"""

import pytest

from tests.unit.fixtures.mock_hf_datasets import MockHFDatasets


@pytest.fixture(autouse=True)
def mock_huggingface_datasets(monkeypatch, request):
    """Automatically mock HuggingFace dataset loading for all tests.

    This prevents tests from making network requests to HuggingFace Hub,
    making tests faster and more reliable.

    Tests can opt-out by using the @pytest.mark.integration marker.

    Args:
        monkeypatch: Pytest's monkeypatch fixture for patching
        request: Pytest's request fixture to access test markers
    """
    # Check if this test is marked as an integration test
    if "integration" in request.keywords:
        # Don't mock for integration tests - they should use real network calls
        return

    # Patch datasets.load_dataset to use our mock
    def mock_load_dataset(path: str, *args, **kwargs):
        """Mock implementation of datasets.load_dataset()."""
        return MockHFDatasets.get_mock_dataset(path, *args, **kwargs)

    # Patch the import in eval_dataset.py where load_dataset is used
    monkeypatch.setattr("scorebook.eval_datasets.eval_dataset.load_dataset", mock_load_dataset)
