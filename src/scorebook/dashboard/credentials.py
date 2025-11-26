"""Authentication and token management for Trismik API."""

import logging
import os
import pathlib
from typing import Optional

from dotenv import load_dotenv
from trismik import TrismikClient

from scorebook.settings import TRISMIK_SERVICE_URL

logger = logging.getLogger(__name__)


def get_scorebook_config_dir() -> str:
    """Get the scorebook config directory."""
    return os.path.join(os.path.expanduser("~"), ".scorebook")


def get_token_path() -> str:
    """Get the path where the trismik token is stored."""
    if "TRISMIK_TOKEN_PATH" in os.environ:
        return os.environ["TRISMIK_TOKEN_PATH"]
    return os.path.join(get_scorebook_config_dir(), "config")


def save_token(token: str) -> None:
    """Save the token to the local cache directory."""
    token_path = get_token_path()

    # Create a directory if it doesn't exist
    os.makedirs(os.path.dirname(token_path), exist_ok=True)

    # Write token to file
    pathlib.Path(token_path).write_text(token.strip())

    # Set restrictive permissions (owner read/write only)
    os.chmod(token_path, 0o600)


def get_stored_token() -> Optional[str]:
    """Retrieve the stored token from the cache directory."""
    token_path = get_token_path()

    if not os.path.exists(token_path):
        return None

    try:
        token = pathlib.Path(token_path).read_text().strip()
        return token if token else None
    except (OSError, IOError) as e:
        logger.warning(f"Failed to read token from {token_path}: {e}")
        return None


def get_token() -> Optional[str]:
    """Get the trismik API token in order of priority.

    Priority order:
    1. TRISMIK_API_KEY environment variable
    2. Stored token file
    """
    # Check environment variable first
    env_token = os.environ.get("TRISMIK_API_KEY")
    if env_token:
        return env_token.strip()

    # Fallback to stored token
    return get_stored_token()


def validate_token(token: str) -> bool:
    """Validate the token by making a test API call to trismik.

    Args:
        token: The API token to validate.

    Returns:
        bool: True if the token is valid, False otherwise.
    """
    if not token or not token.strip():
        return False

    try:
        # Create a client with the token and verify it works
        client = TrismikClient(service_url=TRISMIK_SERVICE_URL, api_key=token)
        client.me()
        client.close()
        return True
    except Exception as e:
        logger.debug(f"Token validation failed: {e}")
        return False


def login(trismik_api_key: Optional[str] = None) -> None:
    """Login to trismik by saving API key locally.

    If no API key is provided, the function will attempt to read it from the
    TRISMIK_API_KEY environment variable or .env file (using python-dotenv).
    Environment variables take precedence over .env file values.

    Args:
        trismik_api_key: The API key to use. If not provided, reads from
            environment or .env file.
    Raises:
        ValueError: If API key is empty, not found, or invalid.
    """
    if trismik_api_key is None:
        # Try to load from .env file first (this will not override existing env vars)
        load_dotenv()
        trismik_api_key = os.environ.get("TRISMIK_API_KEY")

    if not trismik_api_key:
        raise ValueError(
            "API key cannot be empty. Either pass it as a parameter or "
            "set the TRISMIK_API_KEY environment variable or .env file."
        )

    # Validate token
    if not validate_token(trismik_api_key):
        raise ValueError("Invalid API key provided")

    # Save token
    save_token(trismik_api_key)


def logout() -> bool:
    """Remove the stored token.

    Returns:
        bool: True if a token was removed, False if no token was found.
    """
    token_path = get_token_path()

    if os.path.exists(token_path):
        os.remove(token_path)
        return True
    else:
        return False


def whoami() -> Optional[str]:
    """Return information about the current user/token.

    Returns:
        str: The stored token if logged in, None if not logged in.
    """
    return get_stored_token()
