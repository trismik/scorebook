"""CSV dataset loader for Scorebook."""

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset


def from_csv(
    file_path: str,
    *,
    open_kwargs: Optional[Dict[str, Any]] = None,
    reader_kwargs: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """Load a dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.
        open_kwargs: Dict of kwargs passed to `open()`.
        reader_kwargs: Dict of kwargs passed to `csv.DictReader`.

    Returns:
        A Hugging Face Dataset.
    """
    open_kwargs = open_kwargs or {}
    reader_kwargs = reader_kwargs or {}

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, newline="", encoding="utf-8", **open_kwargs) as csvfile:
            reader = csv.DictReader(csvfile, **reader_kwargs)
            data = [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error reading CSV file {file_path}: {e}") from e

    if not data:
        raise ValueError(f"CSV file {file_path} is empty or contains only headers.")

    return Dataset.from_list(data)
