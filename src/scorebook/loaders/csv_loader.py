from datasets import Dataset
from pathlib import Path
import pandas as pd

def from_csv(file_path: str) -> Dataset:
    """Load a dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A Hugging Face Dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error reading CSV file {file_path}: {e}") from e

    if df.empty:
        raise ValueError(f"CSV file {file_path} is empty.")

    return Dataset.from_pandas(df)