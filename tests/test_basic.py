"""Basic tests for the scorebook package."""

from scorebook import __version__


def test_version():
    """Test that the version matches pyproject.toml."""
    import toml

    # Read version from pyproject.toml
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)
    expected_version = pyproject["tool"]["poetry"]["version"]

    # Verify package version matches
    assert __version__ == expected_version
