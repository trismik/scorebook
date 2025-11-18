"""Create projects in Trismik's experimentation platform."""

import logging
from typing import Optional

from trismik.types import TrismikProject

from scorebook.evaluate.evaluate_helpers import (
    create_trismik_async_client,
    create_trismik_sync_client,
)

logger = logging.getLogger(__name__)


def create_project(
    name: str,
    team_id: Optional[str] = None,
    description: Optional[str] = None,
) -> TrismikProject:
    """Create a new project in Trismik's experimentation platform (synchronous).

    This function creates a new project that can be used to organize experiments
    and evaluation runs in the Trismik platform.

    Args:
        name: Name of the project
        team_id: Optional ID of the team to create the project in. If not provided,
            the project will be created in the user's default team.
        description: Optional description of the project

    Returns:
        TrismikProject: Created project object containing project details including
            id, name, description, accountId, createdAt, and updatedAt fields

    Raises:
        TrismikValidationError: If the request fails validation
        TrismikApiError: If the API request fails
    """
    # Create Trismik client
    trismik_client = create_trismik_sync_client()

    # Create project via Trismik API
    project = trismik_client.create_project(
        name=name,
        team_id=team_id,
        description=description,
    )

    logger.info(f"Project '{name}' created successfully with ID: {project.id}")

    return project


async def create_project_async(
    name: str,
    team_id: Optional[str] = None,
    description: Optional[str] = None,
) -> TrismikProject:
    """Create a new project in Trismik's experimentation platform (asynchronous).

    This function creates a new project that can be used to organize experiments
    and evaluation runs in the Trismik platform.

    Args:
        name: Name of the project
        team_id: Optional ID of the team to create the project in. If not provided,
            the project will be created in the user's default team.
        description: Optional description of the project

    Returns:
        TrismikProject: Created project object containing project details including
            id, name, description, accountId, createdAt, and updatedAt fields

    Raises:
        TrismikValidationError: If the request fails validation
        TrismikApiError: If the API request fails
    """
    # Create Trismik async client
    trismik_client = create_trismik_async_client()

    # Create project via Trismik API (async)
    project = await trismik_client.create_project(
        name=name,
        team_id=team_id,
        description=description,
    )

    logger.info(f"Project '{name}' created successfully with ID: {project.id}")

    return project
