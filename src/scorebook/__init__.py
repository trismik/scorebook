"""
Scorebook package.

A Python project for scorebook functionality.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)
