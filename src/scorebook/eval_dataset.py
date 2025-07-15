"""Eval Dataset implementation for scorebook."""

import csv
import json
from typing import Any, Dict, Iterator, List, Optional, Union

from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict
from datasets import load_dataset

from scorebook.utils import validate_path


class EvalDataset:
    """Eval Dataset implementation for scorebook."""

    def __init__(
        self,
        name: str,
        label: str,
        metrics: Optional[List[str]] = None,
        hf_dataset: Optional[HuggingFaceDataset] = None,
    ):
        """
        Create a new scorebook evaluation dataset instance.

        :param name: The name of the evaluation dataset.
        :param label: The label field of the dataset.
        :param metrics: The metrics associated with the dataset.
        :param hf_dataset: The dataset as a hugging face dataset object.
        """
        self.name: str = name
        self.label: str = label
        self.metrics: List[str] = metrics or []
        self._hf_dataset: Optional[HuggingFaceDataset] = hf_dataset

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        return len(self._hf_dataset)

    def __getitem__(self, key: Union[int, str]) -> Union[Dict[str, Any], List[Any]]:
        """
        Allow item access by index (int) or by column name (str).

        - eval_dataset[i] returns the i-th example (dict).
        - eval_dataset["feature"] returns a list of values for that feature.
        """
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        if isinstance(key, int):
            return dict(self._hf_dataset[key])  # Ensure we return a Dict[str, Any]
        elif isinstance(key, str):
            return list(self._hf_dataset[key])  # Ensure we return a List[Any]
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Must be int or str.")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return an iterator over all examples in the dataset."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        return iter(self._hf_dataset)

    @property
    def items(self) -> List[Any]:
        """Return a list of all examples in the dataset."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        return list(self._hf_dataset)

    @property
    def column_names(self) -> List[str]:
        """Return a list of column/feature names available in the dataset."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        return list(map(str, self._hf_dataset.column_names))

    @classmethod
    def from_list(cls, name: str, label: str, data: List[Dict[str, Any]]) -> "EvalDataset":
        """Instantiate an EvalDataset from a list of dictionaries.

        Args:
            cls: The class reference (automatically passed for staticmethod).
            name: The name of the evaluation dataset.
            label: The field used as the evaluation label (ground truth).
            data: List of dictionaries containing the dataset examples.

        Returns:
            A scorebook EvalDataset wrapping a Hugging Face dataset.
        """
        return cls(name=name, label=label, hf_dataset=HuggingFaceDataset.from_list(data))

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        label: str,
        encoding: str = "utf-8",
        newline: str = "",
        **reader_kwargs: Any,
    ) -> "EvalDataset":
        """Instantiate a scorebook dataset from a CSV file.

        Args:
            file_path: Path to the CSV file.
            label: The field used as the evaluation label (ground truth).
            encoding: Encoding of the CSV file.
            newline: Newline character of the CSV file.
            reader_kwargs: Dict of kwargs passed to `csv.DictReader`.

        Returns:
            A scorebook EvalDataset.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the CSV file cannot be parsed or is empty.
        """
        reader_kwargs = reader_kwargs or {}
        path = validate_path(file_path, expected_suffix=".csv")

        try:
            with open(path, encoding=encoding, newline=newline) as csvfile:
                reader = csv.DictReader(csvfile, **reader_kwargs)
                data = [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e

        if not data:
            raise ValueError(f"CSV file {file_path} is empty or contains only headers.")

        return cls(name=path.stem, label=label, hf_dataset=HuggingFaceDataset.from_list(data))

    @classmethod
    def from_json(cls, file_path: str, label: str, split: Optional[str] = None) -> "EvalDataset":
        """Instantiate an EvalDataset from a JSON file.

        The JSON file must follow one of two supported formats:

        1. **Flat format** – a list of dictionaries:
            [
                {"input": "What is 2+2?", "label": "4"},
                {"input": "Capital of France?", "label": "Paris"}
            ]

        2. **Split format** – a dictionary of named splits:
            {
                "train": [{"input": ..., "label": ...}],
                "test": [{"input": ..., "label": ...}]
            }

        Args:
            file_path: Path to the JSON file on disk.
            label: The field used as the evaluation label (ground truth).
            split: If the JSON uses a split structure, this is the split name to load.

        Returns:
            A scorebook EvalDataset wrapping a Hugging Face dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or the structure is unsupported.
        """
        path = validate_path(file_path, expected_suffix=".json")

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
            hf_dataset = HuggingFaceDataset.from_list(split_data)
        elif isinstance(data, list):
            hf_dataset = HuggingFaceDataset.from_list(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}. Expected list or dict.")

        return cls(name=path.stem, label=label, hf_dataset=hf_dataset)

    @classmethod
    def from_huggingface(
        cls, path: str, label: str, split: Optional[str] = None, name: Optional[str] = None
    ) -> "EvalDataset":
        """Instantiate an EvalDataset from a dataset available on Hugging Face Hub.

        If a specific split is provided (e.g., "train" or "test"), it will be loaded directly.
        If no split is specified, the method attempts to load the full dataset. If the dataset
        is split into multiple subsets (i.e., a DatasetDict), it defaults to loading the "test"
        split.

        Args:
            path: The path of the dataset on the Hugging Face Hub.
            label: The field used as the evaluation label (ground truth).
            split: Optional name of the split to load.
            name: Optional dataset configuration name.

        Returns:
            An EvalDataset wrapping the selected Hugging Face dataset.

        Raises:
            ValueError: If the dataset cannot be loaded, or the expected split is missing.
        """
        try:
            kwargs = {}
            if split is not None:
                kwargs["split"] = split
            if name is not None:
                kwargs["name"] = name
            ds = load_dataset(path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{path}' from Hugging Face: {e}") from e

        if isinstance(ds, HuggingFaceDataset):
            hf_dataset = ds
        elif isinstance(ds, HuggingFaceDatasetDict):
            if "test" in ds:
                hf_dataset = ds["test"]
            else:
                raise ValueError(
                    f"Split not specified and no 'test' split found in dataset '{path}'."
                )
        else:
            raise ValueError(f"Unexpected dataset type for '{path}': {type(ds)}")

        return cls(name=path, label=label, hf_dataset=hf_dataset)
