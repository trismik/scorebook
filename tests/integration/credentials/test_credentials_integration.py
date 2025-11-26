"""
Integration tests for trismik credentials and authentication.

These tests make real API calls to the trismik service and do not use mocks.
They require a valid TRISMIK_API_KEY environment variable or .env file to run.
"""

import os

import pytest
from dotenv import load_dotenv

from scorebook.dashboard.credentials import get_token, login, logout, validate_token, whoami

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


@pytest.fixture
def temp_token_path(monkeypatch, tmp_path):
    """Use a temporary directory for token storage."""
    token_path = tmp_path / "test_token"
    monkeypatch.setenv("TRISMIK_TOKEN_PATH", str(token_path))
    yield token_path
    if token_path.exists():
        token_path.unlink()


def test_login_with_valid_key(test_api_key, temp_token_path, monkeypatch):
    """Test that login stores a valid API key."""
    # Remove env var so we only check file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    login(test_api_key)

    # Verify token file was created with correct permissions
    assert temp_token_path.exists()
    assert oct(temp_token_path.stat().st_mode)[-3:] == "600"

    # Verify we can retrieve the token
    assert get_token() == test_api_key


def test_login_with_invalid_key(temp_token_path):
    """Test that login fails with an invalid API key."""
    with pytest.raises(ValueError, match="Invalid API key"):
        login("invalid_key_12345")

    # Verify no token file was created
    assert not temp_token_path.exists()


def test_logout_removes_token(test_api_key, temp_token_path, monkeypatch):
    """Test that logout removes the stored token."""
    # Remove env var so we only check file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    login(test_api_key)
    assert temp_token_path.exists()

    result = logout()
    assert result is True
    assert not temp_token_path.exists()
    assert get_token() is None


def test_logout_when_not_logged_in(temp_token_path):
    """Test that logout returns False when no token exists."""
    result = logout()
    assert result is False


def test_get_token_from_env_variable(test_api_key, temp_token_path, monkeypatch):
    """Test that TRISMIK_API_KEY environment variable is used."""
    monkeypatch.setenv("TRISMIK_API_KEY", test_api_key)

    token = get_token()
    assert token == test_api_key


def test_get_token_from_file(test_api_key, temp_token_path, monkeypatch):
    """Test retrieving token from stored file."""
    # Remove env var so we only check file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    login(test_api_key)

    token = get_token()
    assert token == test_api_key


def test_get_token_env_priority_over_file(test_api_key, temp_token_path, monkeypatch):
    """Test that environment variable takes priority over stored file."""
    # Remove env var first to login to file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    login(test_api_key)

    # Now set a different env var
    env_key = "env_priority_key"
    monkeypatch.setenv("TRISMIK_API_KEY", env_key)

    # Environment variable should take priority over file
    assert get_token() == env_key


def test_get_token_returns_none_when_no_token(temp_token_path, monkeypatch):
    """Test that get_token returns None when no token is available."""
    # Remove env var so get_token returns None
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    assert get_token() is None


def test_validate_token_with_valid_key(test_api_key):
    """Test validating a valid API key."""
    assert validate_token(test_api_key) is True


def test_validate_token_with_invalid_key():
    """Test validating an invalid API key."""
    assert validate_token("invalid_key_12345") is False


def test_whoami_when_logged_in(test_api_key, temp_token_path, monkeypatch):
    """Test whoami returns token when logged in."""
    # Remove env var so we only check file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    login(test_api_key)

    assert whoami() == test_api_key


def test_whoami_when_not_logged_in(temp_token_path, monkeypatch):
    """Test whoami returns None when not logged in."""
    # Remove env var so whoami returns None
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    assert whoami() is None


def test_full_login_logout_cycle(test_api_key, temp_token_path, monkeypatch):
    """Test a complete login/logout cycle."""
    # Remove env var so we only check file
    monkeypatch.delenv("TRISMIK_API_KEY", raising=False)

    # Start logged out
    assert get_token() is None
    assert whoami() is None

    # Login
    login(test_api_key)
    assert get_token() == test_api_key
    assert whoami() == test_api_key
    assert temp_token_path.exists()

    # Validate token works
    assert validate_token(test_api_key) is True

    # Logout
    assert logout() is True
    assert get_token() is None
    assert whoami() is None
    assert not temp_token_path.exists()
