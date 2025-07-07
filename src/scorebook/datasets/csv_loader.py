"""CSV dataset loader for Scorebook."""

import csv
from pathlib import Path
from typing import Any

from datasets import Dataset


def from_csv(
    file_path: str, encoding: str = "utf-8", newline: str = "", **reader_kwargs: Any
) -> Dataset:
    """Load a dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.
        encoding: Encoding of the CSV file.
        newline: Newline character of the CSV file.
        reader_kwargs: Dict of kwargs passed to `csv.DictReader`.

    Returns:
        A Hugging Face Dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the CSV file cannot be parsed or is empty.
    """
    reader_kwargs = reader_kwargs or {}

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, encoding=encoding, newline=newline) as csvfile:
            reader = csv.DictReader(csvfile, **reader_kwargs)
            data = [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e

    if not data:
        raise ValueError(f"CSV file {file_path} is empty or contains only headers.")

    return Dataset.from_list(data)
