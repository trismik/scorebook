"""Configuration settings for Scorebook."""

import os

TRISMIK_API_URL = "https://api.trismik.com/adaptive-testing"
TRISMIK_STAGE_API_URL = "https://api-stage.trismik.com/adaptive-testing"

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv(verbose=False)
except ImportError:  # pragma: no cover
    pass

# Settings from environment
USE_TRISMIK_STAGE = os.environ.get("USE_TRISMIK_STAGE", "false").lower() == "true"
TRISMIK_SERVICE_URL = os.environ.get(
    "TRISMIK_SERVICE_URL",
    TRISMIK_STAGE_API_URL if USE_TRISMIK_STAGE else TRISMIK_API_URL,
)
TRISMIK_API_KEY_ENV_VAR = (  # pragma: allowlist secret
    "TRISMIK_API_KEY_STAGE" if USE_TRISMIK_STAGE else "TRISMIK_API_KEY"
)
SHOW_PROGRESS_BARS = os.environ.get("SCOREBOOK_SHOW_PROGRESS_BARS", "true").lower() == "true"
