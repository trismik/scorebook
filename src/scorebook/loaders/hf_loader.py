"""Huggingface dataset loader for Scorebook."""

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset


def from_huggingface(dataset: str, split: Optional[str] = None) -> Dataset:
    """Load a dataset from the Hugging Face Hub.

    Args:
        dataset: Dataset name or path on the Hugging Face Hub.
        split: Optional split name (e.g., 'train', 'test').

    Returns:
        A Hugging Face Dataset.

    Raises:
        ValueError: If the dataset or requested split is invalid.
    """
    try:
        if split:
            ds = load_dataset(dataset, split=split)
        else:
            ds = load_dataset(dataset)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset}' from Hugging Face: {e}") from e

    if isinstance(ds, Dataset):
        return ds
    elif isinstance(ds, DatasetDict):
        if "train" in ds:
            return ds["train"]
        raise ValueError(f"Split not specified and no 'train' split found in dataset '{dataset}'.")
    else:
        raise ValueError(f"Unexpected dataset type for '{dataset}': {type(ds)}")
