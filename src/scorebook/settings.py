"""Configuration settings for Scorebook."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(verbose=False)

# Trismik API settings
TRISMIK_API_BASE_URL = "https://api.trismik.com"
TRISMIK_ADAPTIVE_TESTING_URL = f"{TRISMIK_API_BASE_URL}/adaptive-testing"

# Allow override via environment variable
TRISMIK_SERVICE_URL = os.environ.get("TRISMIK_SERVICE_URL", TRISMIK_ADAPTIVE_TESTING_URL)
