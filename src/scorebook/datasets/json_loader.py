"""JSON dataset loader for Scorebook."""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset


def from_json(file_path: str, split: Optional[str] = None) -> Dataset:
    """Load a dataset from a JSON file.

    The JSON file must follow one of two formats:

    1. **List of items (no splits)**:
        A JSON array where each element is a dictionary representing one item.
        Example:

        [
            {"input": "What is 2+2?", "label": "4"},
            {"input": "Capital of France?", "label": "Paris"}
        ]


    2. **Dataset Dict structure**:
        A JSON object where each key is a split name (e.g. "train", "test", "validation"),
        and the value is a list of items.
        The dataset within the "train" split is returned by default.
        Example:

        {
            "train": [
                {"input": "What is 2+2?", "label": "4"}
            ],
            "test": [
                {"input": "Capital of France?", "label": "Paris"}
            ]
        }

    Args:
        file_path: Path to the JSON file.
        split: Optional key for split if the JSON contains a dict of splits.

    Returns:
        A Hugging Face Dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contents are not valid JSON or not in expected format.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

    if isinstance(data, dict):
        if split is None:
            raise ValueError(f"Split name must be provided for split-style JSON: {file_path}")
        split_data = data.get(split)
        if split_data is None:
            raise ValueError(f"Split '{split}' not found in JSON file: {file_path}")
        if not isinstance(split_data, list):
            raise ValueError(f"Split '{split}' is not a list of examples in: {file_path}")
        return Dataset.from_list(split_data)
    elif isinstance(data, list):
        return Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported JSON structure in {file_path}. Expected list or dict.")
