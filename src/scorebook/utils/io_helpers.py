"""Input/output helper functions for Scorebook."""

from pathlib import Path
from typing import Optional, Tuple, Union


def validate_path(
    file_path: str, expected_suffix: Optional[Union[str, Tuple[str, ...]]] = None
) -> Path:
    """Validate that a file path exists and optionally check its suffix.

    Args:
        file_path: Path to the file as string or Path object
        expected_suffix: Optional file extension(s) to validate.
            Can be a single string (e.g. ".json") or tuple of strings (e.g. (".yaml", ".yml"))

    Returns:
        Path object for the validated file path

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file has the wrong extension
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if expected_suffix:
        # Convert single suffix to tuple for uniform handling
        allowed_suffixes = (
            (expected_suffix,) if isinstance(expected_suffix, str) else expected_suffix
        )
        allowed_suffixes_lower = tuple(s.lower() for s in allowed_suffixes)

        if path.suffix.lower() not in allowed_suffixes_lower:
            suffix_list = ", ".join(f"'{s}'" for s in allowed_suffixes)
            raise ValueError(
                f"File must have one of ({suffix_list}) extensions, got: '{path.suffix}'"
            )

    return path
