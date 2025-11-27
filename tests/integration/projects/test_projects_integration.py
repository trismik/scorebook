"""
Integration tests for trismik project management.

These tests make real API calls to the trismik service and do not use mocks.
They require a valid TRISMIK_API_KEY environment variable or .env file to run.
"""

import os
import uuid

import pytest
from dotenv import load_dotenv

from scorebook.dashboard.create_project import create_project, create_project_async

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def test_api_key() -> str:
    """Get test API key from environment or .env file."""
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        pytest.fail(
            "TRISMIK_API_KEY not found. Set it as an environment variable or in a .env file."
        )
    return api_key


@pytest.fixture(autouse=True)
def ensure_logged_in(test_api_key, monkeypatch):
    """Ensure we're logged in for all tests."""
    monkeypatch.setenv("TRISMIK_API_KEY", test_api_key)


def test_create_project_basic():
    """Test creating a project with just a name."""
    project_name = f"test-project-{uuid.uuid4().hex[:8]}"
    project = create_project(name=project_name)

    assert project is not None
    assert project.id is not None
    assert project.name == project_name
    assert project.accountId is not None
    assert project.createdAt is not None
    assert project.updatedAt is not None


def test_create_project_with_description():
    """Test creating a project with a description."""
    project_name = f"test-project-{uuid.uuid4().hex[:8]}"
    description = "Integration test project"

    project = create_project(name=project_name, description=description)

    assert project.name == project_name
    assert project.description == description
    assert project.id is not None


def test_create_multiple_projects():
    """Test creating multiple projects."""
    projects = []

    for i in range(3):
        project_name = f"test-multi-{uuid.uuid4().hex[:8]}"
        project = create_project(name=project_name)
        projects.append(project)

    # Verify all projects were created with unique IDs
    assert len(projects) == 3
    assert all(p.id is not None for p in projects)
    project_ids = [p.id for p in projects]
    assert len(set(project_ids)) == 3


@pytest.mark.asyncio
async def test_create_project_async():
    """Test creating a project asynchronously."""
    project_name = f"test-async-{uuid.uuid4().hex[:8]}"
    project = await create_project_async(name=project_name)

    assert project is not None
    assert project.id is not None
    assert project.name == project_name


@pytest.mark.asyncio
async def test_create_project_async_with_description():
    """Test creating a project asynchronously with description."""
    project_name = f"test-async-{uuid.uuid4().hex[:8]}"
    description = "Async test project"

    project = await create_project_async(name=project_name, description=description)

    assert project.name == project_name
    assert project.description == description


@pytest.mark.asyncio
async def test_create_multiple_projects_async_concurrent():
    """Test creating multiple projects concurrently."""
    import asyncio

    project_names = [f"test-concurrent-{uuid.uuid4().hex[:8]}" for _ in range(3)]
    tasks = [create_project_async(name=name) for name in project_names]
    projects = await asyncio.gather(*tasks)

    # Verify all projects were created with unique IDs
    assert len(projects) == 3
    assert all(p.id is not None for p in projects)
    project_ids = [p.id for p in projects]
    assert len(set(project_ids)) == 3


def test_create_project_with_empty_name():
    """Test that creating a project with empty name raises an error."""
    with pytest.raises(Exception):
        create_project(name="")
