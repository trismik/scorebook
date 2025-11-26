"""Tests for trismik login functionality."""

# Import the actual module using importlib to avoid __init__.py import conflicts
import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import functions from the module
from scorebook.dashboard.credentials import (
    get_scorebook_config_dir,
    get_stored_token,
    get_token,
    get_token_path,
    login,
    logout,
    save_token,
    validate_token,
    whoami,
)

login_module = importlib.import_module("scorebook.dashboard.credentials")


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(login_module, "get_scorebook_config_dir", return_value=temp_dir):
            yield temp_dir


@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    original_env = os.environ.copy()
    # Remove any trismik-related env vars
    for key in ["TRISMIK_API_KEY", "TRISMIK_TOKEN_PATH", "TRISMIK_HOME"]:
        os.environ.pop(key, None)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class TestConfigPaths:
    """Test configuration path functions."""

    def test_get_scorebook_config_dir_default(self):
        """Test default config directory path."""
        expected = os.path.join(os.path.expanduser("~"), ".scorebook")
        assert get_scorebook_config_dir() == expected

    def test_get_token_path_default(self, temp_config_dir):
        """Test default token path."""
        expected = os.path.join(temp_config_dir, "config")
        assert get_token_path() == expected

    def test_get_token_path_custom_env(self, clean_env):
        """Test token path with custom environment variable."""
        custom_path = "/custom/path/token"
        os.environ["TRISMIK_TOKEN_PATH"] = custom_path
        assert get_token_path() == custom_path


class TestTokenStorage:
    """Test token storage and retrieval."""

    def test_save_and_get_token(self, temp_config_dir):
        """Test saving and retrieving a token."""
        test_token = "test-api-key-12345"

        # Save token
        save_token(test_token)

        # Verify file exists
        token_path = os.path.join(temp_config_dir, "config")
        assert os.path.exists(token_path)

        # Verify file permissions are restrictive
        file_stat = os.stat(token_path)
        assert oct(file_stat.st_mode)[-3:] == "600"

        # Retrieve token
        retrieved_token = get_stored_token()
        assert retrieved_token == test_token

    def test_get_stored_token_no_file(self, temp_config_dir):
        """Test getting token when file doesn't exist."""
        assert get_stored_token() is None

    def test_get_stored_token_empty_file(self, temp_config_dir):
        """Test getting token from empty file."""
        token_path = os.path.join(temp_config_dir, "config")
        Path(token_path).write_text("")
        assert get_stored_token() is None

    def test_save_token_strips_whitespace(self, temp_config_dir):
        """Test that saved tokens are stripped of whitespace."""
        test_token = "  test-token-with-spaces  \n"
        save_token(test_token)

        retrieved_token = get_stored_token()
        assert retrieved_token == "test-token-with-spaces"


class TestTokenRetrieval:
    """Test token retrieval with priority handling."""

    def test_get_token_env_var_priority(self, temp_config_dir, clean_env):
        """Test that environment variable takes priority over stored token."""
        # Save a token to file
        save_token("stored-token")

        # Set environment variable
        os.environ["TRISMIK_API_KEY"] = "env-token"  # pragma: allowlist secret

        # Environment variable should take priority
        assert get_token() == "env-token"

    def test_get_token_fallback_to_stored(self, temp_config_dir, clean_env):
        """Test fallback to stored token when no env var."""
        save_token("stored-token")
        assert get_token() == "stored-token"

    def test_get_token_no_token_available(self, temp_config_dir, clean_env):
        """Test when no token is available."""
        assert get_token() is None

    def test_get_token_env_var_with_whitespace(self, clean_env):
        """Test environment variable token is stripped."""
        os.environ["TRISMIK_API_KEY"] = "  env-token-with-spaces  "  # pragma: allowlist secret
        assert get_token() == "env-token-with-spaces"


class TestValidateToken:
    """Test token validation."""

    def test_validate_token_valid(self):
        """Test validation of valid token."""
        # Mock the TrismikClient to simulate successful validation
        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            assert validate_token("valid-token") is True
            mock_client.assert_called_once()
            mock_instance.me.assert_called_once()
            mock_instance.close.assert_called_once()

    def test_validate_token_api_error(self):
        """Test validation when API call fails."""
        # Mock the TrismikClient to simulate API error
        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.side_effect = Exception("API Error")

            assert validate_token("invalid-token") is False

    def test_validate_token_empty(self):
        """Test validation of empty token."""
        assert validate_token("") is False
        assert validate_token("   ") is False

    def test_validate_token_none(self):
        """Test validation of None token."""
        assert validate_token(None) is False


class TestLogin:
    """Test login functionality."""

    def test_login_with_token_parameter(self, temp_config_dir, clean_env):
        """Test login with token provided as parameter."""
        test_token = "test-login-token"

        # Mock the TrismikClient to simulate successful validation
        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            login(test_token)

            # Verify token was saved
            assert get_stored_token() == test_token

    def test_login_empty_token_raises_error(self, clean_env):
        """Test that empty token raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            login("")

    def test_login_invalid_token_raises_error(self, clean_env):
        """Test that invalid token raises ValueError."""
        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.side_effect = Exception("API Error")

            with pytest.raises(ValueError, match="Invalid API key provided"):
                login("invalid-token")

    def test_login_without_parameter_uses_env_var(self, temp_config_dir, clean_env):
        """Test login without parameter reads from environment variable."""
        test_token = "env-var-token"
        os.environ["TRISMIK_API_KEY"] = test_token  # pragma: allowlist secret

        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            login()

            # Verify token was saved
            assert get_stored_token() == test_token

    def test_login_without_parameter_uses_dotenv(self, temp_config_dir, clean_env):
        """Test login without parameter reads from .env file via dotenv."""
        test_token = "dotenv-token"

        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            # Mock load_dotenv to set the env var
            with patch("scorebook.dashboard.credentials.load_dotenv") as mock_load_dotenv:

                def set_env_var() -> bool:
                    os.environ["TRISMIK_API_KEY"] = test_token  # pragma: allowlist secret
                    return True

                mock_load_dotenv.side_effect = set_env_var

                login()

                mock_load_dotenv.assert_called_once()
                # Verify token was saved
                assert get_stored_token() == test_token

    def test_login_without_parameter_no_key_raises_error(self, clean_env):
        """Test login without parameter raises error when no key available."""
        with patch("scorebook.dashboard.credentials.load_dotenv"):
            with pytest.raises(ValueError, match="API key cannot be empty"):
                login()

    def test_login_warns_when_env_var_and_explicit_key(self, temp_config_dir, clean_env):
        """Test login warns when env var is set but explicit key is passed."""
        explicit_token = "explicit-token"
        os.environ["TRISMIK_API_KEY"] = "env-token"  # pragma: allowlist secret

        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            with pytest.warns(UserWarning, match="TRISMIK_API_KEY environment variable"):
                login(explicit_token)

            # Token should still be saved
            assert get_stored_token() == explicit_token


class TestLogout:
    """Test logout functionality."""

    def test_logout_with_existing_token(self, temp_config_dir):
        """Test logout when token exists."""
        # Save a token first
        save_token("test-token")

        result = logout()

        # Verify token was removed and function returned True
        assert get_stored_token() is None
        assert result is True

    def test_logout_no_existing_token(self, temp_config_dir):
        """Test logout when no token exists."""
        result = logout()

        # Verify function returned False
        assert result is False


class TestWhoami:
    """Test whoami functionality."""

    def test_whoami_with_token(self, temp_config_dir):
        """Test whoami when token exists."""
        test_token = "test-token-12345678"
        save_token(test_token)

        result = whoami()

        assert result == test_token

    def test_whoami_no_token(self, temp_config_dir):
        """Test whoami when no token exists."""
        result = whoami()

        assert result is None


class TestIntegration:
    """Integration tests."""

    def test_full_login_logout_cycle(self, temp_config_dir, clean_env):
        """Test complete login/logout cycle."""
        test_token = "integration-test-token"

        # Mock the TrismikClient for login validation
        with patch("scorebook.dashboard.credentials.TrismikClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.me.return_value = {"user": "test"}

            # Login
            login(test_token)

            # Verify token is available
            assert get_token() == test_token

            # Logout
            logout()

            # Verify token is gone
            assert get_token() is None

    def test_token_priority_env_over_stored(self, temp_config_dir, clean_env):
        """Test that environment variable takes priority over stored token."""
        # Save token to file
        save_token("stored-token")

        # Set environment variable
        os.environ["TRISMIK_API_KEY"] = "env-token"  # pragma: allowlist secret

        # Environment variable should win
        assert get_token() == "env-token"

        # Remove env var
        del os.environ["TRISMIK_API_KEY"]

        # Should fallback to stored token
        assert get_token() == "stored-token"
