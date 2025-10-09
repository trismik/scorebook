"""Async utilities for handling callable objects and coroutines."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional, TypeVar

T = TypeVar("T")


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


@asynccontextmanager
async def async_nullcontext(value: Optional[T] = None) -> AsyncIterator[Optional[T]]:
    """Async version of contextlib.nullcontext for Python 3.9 compatibility.

    contextlib.nullcontext() is sync-only and cannot be used with async with on Python 3.9.
    This provides an async equivalent that can be used with async context managers.

    Args:
        value: Optional value to yield from the context manager

    Yields:
        The provided value
    """
    yield value
