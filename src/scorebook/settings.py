"""Configuration settings for Scorebook."""

import os

# Optional: Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv(verbose=False)
except ImportError:  # pragma: no cover
    pass  # python-dotenv not installed, skip .env file loading

# Trismik API settings
TRISMIK_API_BASE_URL = "https://api.trismik.com"
TRISMIK_ADAPTIVE_TESTING_URL = f"{TRISMIK_API_BASE_URL}/adaptive-testing"

# Allow override via environment variable
TRISMIK_SERVICE_URL = os.environ.get("TRISMIK_SERVICE_URL", TRISMIK_ADAPTIVE_TESTING_URL)

# Progress bar configuration
SHOW_PROGRESS_BARS = os.environ.get("SCOREBOOK_SHOW_PROGRESS_BARS", "true").lower() == "true"
