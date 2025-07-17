"""Input/output helper functions for Scorebook."""

from pathlib import Path
from typing import Optional


def validate_path(file_path: str, expected_suffix: Optional[str] = None) -> Path:
    """Validate that a file path exists and optionally check its suffix.

    Args:
        file_path: Path to the file as string or Path object
        expected_suffix: Optional file extension to validate (e.g. ".json", ".csv")

    Returns:
        Path object for the validated file path

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file has the wrong extension
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if expected_suffix and path.suffix.lower() != expected_suffix.lower():
        raise ValueError(f"File must have {expected_suffix} extension, got: {path.suffix}")

    return path
