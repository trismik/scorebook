"""
Evaluation module for Scorebook.

This module provides both synchronous and asynchronous evaluation functions.
The async version serves as the source of truth, with the sync version
automatically generated using unasync.
"""

# Import from async module
from ._async.evaluate_async import evaluate_async

# Import from generated sync module
from ._sync.evaluate import evaluate

__all__ = ["evaluate", "evaluate_async"]
