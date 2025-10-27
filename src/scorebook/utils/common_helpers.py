"""Common helper functions shared across scorebook modules."""

import logging
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)


def resolve_upload_results(upload_results: Union[Literal["auto"], bool]) -> bool:
    """Resolve the upload_results parameter based on trismik login status.

    Args:
        upload_results: Can be True, False, or "auto". When "auto", resolves to True
            if user is logged in to Trismik, False otherwise.

    Returns:
        bool: Whether to upload results to Trismik
    """
    if upload_results == "auto":
        from scorebook.trismik.credentials import get_token

        upload_results = get_token() is not None
        logger.debug("Auto upload results resolved to: %s", upload_results)

    return upload_results


def resolve_show_progress(show_progress: Optional[bool]) -> bool:
    """Resolve whether to show progress bars.

    Args:
        show_progress: Explicit setting (None uses default from settings)

    Returns:
        bool: Whether to show progress bars
    """
    if show_progress is None:
        from scorebook.settings import SHOW_PROGRESS_BARS

        return bool(SHOW_PROGRESS_BARS)
    return show_progress
