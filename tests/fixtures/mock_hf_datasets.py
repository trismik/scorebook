"""Mock HuggingFace datasets for testing without network access.

This module provides mock implementations of HuggingFace datasets
that can be used in tests to avoid network dependencies.
"""

from typing import Any, Optional

from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict


class MockHFDatasets:
    """Factory for creating mock HuggingFace datasets."""

    @staticmethod
    def create_imdb_dataset() -> HuggingFaceDataset:
        """Create a mock IMDB dataset matching the real structure.

        The IMDB dataset contains movie reviews with sentiment labels.
        Structure: {"text": str, "label": int}
        Labels: 0 = negative, 1 = positive
        """
        data = {
            "text": [
                "This movie was fantastic! I loved every minute of it.",
                "Terrible film, waste of time and money.",
                "An okay movie, nothing special but watchable.",
                "Absolutely brilliant! Best film I've seen this year.",
                "Boring and predictable, do not recommend.",
            ],
            "label": [1, 0, 1, 1, 0],  # 1 = positive, 0 = negative
        }
        return HuggingFaceDataset.from_dict(data)

    @staticmethod
    def create_mmlu_pro_dataset() -> HuggingFaceDataset:
        """Create a mock MMLU-Pro dataset matching the real structure.

        MMLU-Pro is a multiple-choice question dataset.
        Structure: {"question": str, "options": List[str], "answer": str}
        """
        data = {
            "question": [
                "What is the capital of France?",
                "What is 2 + 2?",
                "Who wrote Romeo and Juliet?",
            ],
            "options": [
                ["London", "Paris", "Berlin", "Madrid"],
                ["3", "4", "5", "6"],
                ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
            ],
            "answer": ["Paris", "4", "William Shakespeare"],
        }
        return HuggingFaceDataset.from_dict(data)

    @staticmethod
    def get_mock_dataset(
        path: str, split: Optional[str] = None, name: Optional[str] = None, **kwargs
    ) -> Any:
        """Mock replacement for datasets.load_dataset().

        This function mimics the behavior of HuggingFace's load_dataset()
        but returns pre-defined mock data instead of downloading from the Hub.

        Args:
            path: Dataset identifier (e.g., "imdb", "TIGER-Lab/MMLU-Pro")
            split: Optional split name (e.g., "test", "train", "validation")
            name: Optional config name
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            HuggingFaceDataset if split is specified, else HuggingFaceDatasetDict

        Raises:
            ValueError: If the dataset path is not mocked
        """
        # Map dataset paths to factory methods
        dataset_map = {
            "imdb": MockHFDatasets.create_imdb_dataset,
            "TIGER-Lab/MMLU-Pro": MockHFDatasets.create_mmlu_pro_dataset,
        }

        if path not in dataset_map:
            raise ValueError(
                f"Mock dataset not available for '{path}'. "
                f"Available datasets: {list(dataset_map.keys())}. "
                f"Add new mocks to tests/fixtures/mock_hf_datasets.py"
            )

        dataset = dataset_map[path]()

        # If split is requested, return the dataset directly
        if split is not None:
            return dataset

        # If no split specified, return a DatasetDict with common splits
        return HuggingFaceDatasetDict(
            {
                "train": dataset,
                "test": dataset,
                "validation": dataset,
            }
        )
