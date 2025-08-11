"""Async utilities for handling callable objects and coroutines."""

import asyncio
from typing import Callable


def is_awaitable(obj: Callable) -> bool:
    """
    Check if a callable object is awaitable.

    This handles both coroutine functions and callable instances (like classes
    with __call__ methods) that may return coroutines.

    Args:
        obj: The callable object to check

    Returns:
        True if the object is awaitable, False otherwise
    """
    if asyncio.iscoroutinefunction(obj):
        return True

    # Check if it's a callable instance with an awaitable __call__ method
    if hasattr(obj, "__call__") and asyncio.iscoroutinefunction(obj.__call__):
        return True

    return False
