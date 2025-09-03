"""
Scorebook package.

A Python project for scorebook functionality.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

# Automatically load TRISMIK_API_KEY at import time
import os

from scorebook.evaluator import evaluate
from scorebook.trismik.login import get_token, login
from scorebook.types.eval_dataset import EvalDataset
from scorebook.utils.build_prompt import build_prompt

# Set TRISMIK_API_KEY environment variable if not already set
_token = get_token()
if _token and "TRISMIK_API_KEY" not in os.environ:
    os.environ["TRISMIK_API_KEY"] = _token


__all__ = ["EvalDataset", "evaluate", "build_prompt", "login"]
