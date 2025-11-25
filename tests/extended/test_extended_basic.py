"""Verify extended test directory is properly configured."""

import pytest


@pytest.mark.extended
def test_extended_tests_discoverable() -> None:
    """Verify that extended tests are properly discovered by pytest."""
    assert True
