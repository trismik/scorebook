"""Basic tests for the scorebook package."""

from scorebook import __version__


def test_version():
    """Test that the version is importable and valid."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0
