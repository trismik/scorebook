"""Eval Dataset implementation for scorebook."""

import csv
import json
import random
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import yaml
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict
from datasets import load_dataset

from scorebook.metrics import MetricBase, MetricRegistry
from scorebook.utils import render_template, validate_path


class EvalDataset:
    """Eval Dataset implementation for scorebook."""

    def __init__(
        self,
        name: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        hf_dataset: HuggingFaceDataset,
    ):
        """
        Create a new scorebook evaluation dataset instance.

        All EvalDatasets must have exactly 2 columns: 'input' and 'label'.
        Factory methods handle the transformation of raw data into this format.

        :param name: The name of the evaluation dataset.
        :param metrics: The specified metrics associated with the dataset.
        :param hf_dataset: The dataset as a hugging face dataset object
            with 'input' and 'label' columns.

        :raises ValueError: If the dataset doesn't have exactly 'input' and 'label' columns.
        """
        # Validate that dataset has exactly the required columns
        column_names = list(hf_dataset.column_names)
        required_columns = {"input", "label"}
        actual_columns = set(column_names)

        if actual_columns != required_columns:
            raise ValueError(
                f"EvalDataset must have exactly 'input' and 'label' columns. "
                f"Got: {column_names}"
            )

        self.name: str = name
        self.metrics: List[MetricBase] = self._resolve_metrics(metrics)
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

    def __str__(self) -> str:
        """Return a formatted string summary of the evaluation dataset."""
        if self._hf_dataset is None:
            return f"EvalDataset(name='{self.name}', status='uninitialized')"

        num_rows = len(self._hf_dataset)
        fields = ", ".join(self.column_names)
        metrics = ", ".join([metric.name for metric in self.metrics])

        return (
            f"EvalDataset(\n"
            f"  name='{self.name}',\n"
            f"  rows={num_rows},\n"
            f"  fields=[{fields}],\n"
            f"  metrics=[{metrics}]\n"
            f")"
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return an iterator over all examples in the dataset."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        return iter(self._hf_dataset)

    def shuffle(self) -> None:
        """Randomly shuffle the dataset items."""
        if self._hf_dataset is None:
            raise ValueError("Dataset is not initialized")
        self._hf_dataset.shuffle()

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

    # === EvalDataset Factory Methods ===

    @classmethod
    def from_list(
        cls,
        name: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        items: List[Dict[str, Any]],
        input: str,
        label: str,
    ) -> "EvalDataset":
        """Instantiate an EvalDataset from a list of dictionaries.

        Extracts the specified input and label fields from each item to create
        a standardized 2-column dataset (input, label).

        Args:
            name: The name of the evaluation dataset.
            metrics: The specified metrics associated with the dataset.
            items: List of dictionaries containing the dataset examples.
            input: The field name containing the input data.
            label: The field name containing the label (ground truth).

        Returns:
            A scorebook EvalDataset with 'input' and 'label' columns.

        Raises:
            KeyError: If input or label field is missing from any item.
        """
        # Transform items to standardized 2-column format
        transformed_items = cls._apply_templates(
            data=items,
            input_field=input,
            label_field=label,
            input_template=None,
            label_template=None,
        )

        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(transformed_items),
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        input: str,
        label: str,
        name: Optional[str] = None,
        encoding: str = "utf-8",
        newline: str = "",
        **reader_kwargs: Any,
    ) -> "EvalDataset":
        """Instantiate a scorebook dataset from a CSV file.

        Extracts the specified input and label fields from each row to create
        a standardized 2-column dataset (input, label).

        Args:
            path: Path to the CSV file.
            metrics: The specified metrics associated with the dataset.
            input: The field name containing the input data.
            label: The field name containing the label (ground truth).
            name: Optional name for the eval dataset, if not provided, the path is used
            encoding: Encoding of the CSV file.
            newline: Newline character of the CSV file.
            reader_kwargs: Dict of kwargs passed to `csv.DictReader`.

        Returns:
            A scorebook EvalDataset with 'input' and 'label' columns.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the CSV file cannot be parsed or is empty.
            KeyError: If input or label field is missing from any row.
        """
        reader_kwargs = reader_kwargs or {}
        validated_path = validate_path(path, expected_suffix=".csv")

        try:
            with open(validated_path, encoding=encoding, newline=newline) as csvfile:
                reader = csv.DictReader(csvfile, **reader_kwargs)
                data = [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Failed to parse CSV file {path}: {e}") from e

        if not data:
            raise ValueError(f"CSV file {path} is empty or contains only headers.")

        # Transform data to standardized 2-column format
        transformed_data = cls._apply_templates(
            data=data,
            input_field=input,
            label_field=label,
            input_template=None,
            label_template=None,
        )

        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(transformed_data),
        )

    @classmethod
    def from_json(
        cls,
        path: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        input: str,
        label: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> "EvalDataset":
        """Instantiate an EvalDataset from a JSON file.

        The JSON file must follow one of two supported formats:

        1. **Flat format** – a list of dictionaries:
            [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "Capital of France?", "answer": "Paris"}
            ]

        2. **Split format** – a dictionary of named splits:
            {
                "train": [{"question": ..., "answer": ...}],
                "test": [{"question": ..., "answer": ...}]
            }

        Extracts the specified input and label fields from each item to create
        a standardized 2-column dataset (input, label).

        Args:
            path: Path to the JSON file on disk.
            metrics: The specified metrics associated with the dataset.
            input: The field name containing the input data.
            label: The field name containing the label (ground truth).
            name: Optional name for the eval dataset, if not provided, the path is used
            split: If the JSON uses a split structure, this is the split name to load.

        Returns:
            A scorebook EvalDataset with 'input' and 'label' columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or the structure is unsupported.
            KeyError: If input or label field is missing from any item.
        """
        validated_path = validate_path(path, expected_suffix=".json")

        try:
            with validated_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

        if isinstance(data, dict):
            if split is None:
                raise ValueError(f"Split name must be provided for split-style JSON: {path}")
            split_data = data.get(split)
            if split_data is None:
                raise ValueError(f"Split '{split}' not found in JSON file: {path}")
            if not isinstance(split_data, list):
                raise ValueError(f"Split '{split}' is not a list of examples in: {path}")
            raw_data = split_data
        elif isinstance(data, list):
            raw_data = data
        else:
            raise ValueError(f"Unsupported JSON structure in {path}. Expected list or dict.")

        # Transform data to standardized 2-column format
        transformed_data = cls._apply_templates(
            data=raw_data,
            input_field=input,
            label_field=label,
            input_template=None,
            label_template=None,
        )

        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(transformed_data),
        )

    @classmethod
    def from_huggingface(
        cls,
        path: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        input: Optional[str] = None,
        input_template: Optional[str] = None,
        label: Optional[str] = None,
        label_template: Optional[str] = None,
        name: Optional[str] = None,
        split: Optional[str] = None,
        config: Optional[str] = None,
    ) -> "EvalDataset":
        """Instantiate an EvalDataset from a dataset available on Hugging Face Hub.

        If a specific split is provided (e.g., "train" or "test"), it will be loaded directly.
        If no split is specified, the method attempts to load the full dataset. If the dataset
        is split into multiple subsets (i.e., a DatasetDict), it defaults to loading the "test"
        split.

        For datasets where the input/label is already in a single column, use the `input`/`label`
        parameters to specify the column names. For datasets where the input/label needs to be
        constructed from multiple columns, use the `input_template`/`label_template` parameters
        with Jinja2 template strings.

        Args:
            path: The path of the dataset on the Hugging Face Hub.
            metrics: The specified metrics associated with the dataset.
            input: Field name containing the input data (mutually exclusive with input_template).
            input_template: Jinja2 template to construct input from multiple fields
                          (mutually exclusive with input).
            label: Field name containing the label (mutually exclusive with label_template).
            label_template: Jinja2 template to construct label from multiple fields
                          (mutually exclusive with label).
            name: Optional name for the eval dataset, by default HF "path:split:config" is used.
            split: Optional name of the split to load.
            config: Optional dataset configuration name.

        Returns:
            An EvalDataset with 'input' and 'label' columns.

        Raises:
            ValueError: If the dataset cannot be loaded, parameters are invalid, or the
                       expected split is missing.
        """
        # Validate mutually exclusive parameters
        if (input is None) == (input_template is None):
            raise ValueError(
                "Exactly one of 'input' or 'input_template' must be provided, not both or neither."
            )
        if (label is None) == (label_template is None):
            raise ValueError(
                "Exactly one of 'label' or 'label_template' must be provided, not both or neither."
            )

        try:
            kwargs = {}
            if split is not None:
                kwargs["split"] = split
            if config is not None:
                kwargs["name"] = (
                    config  # Hugging Face's load_dataset method param for config is "name"
                )
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

        # Transform data to standardized 2-column format using Dataset.map for memory efficiency
        def transform_row(row: Dict[str, Any]) -> Dict[str, Any]:
            """Transform a single row to have only 'input' and 'label' columns."""
            # Determine input value
            if input_template is not None:
                input_value = render_template(input_template, row)
            else:
                input_value = row[input] if input is not None else ""

            # Determine label value
            if label_template is not None:
                label_value = render_template(label_template, row)
            else:
                label_value = row[label] if label is not None else ""

            return {"input": input_value, "label": label_value}

        # Apply transformation using map for memory efficiency
        transformed_dataset = hf_dataset.map(
            transform_row,
            remove_columns=hf_dataset.column_names,
        )

        dataset_name = name if name else ":".join(filter(None, [path, split, config]))
        return cls(
            name=dataset_name,
            metrics=metrics,
            hf_dataset=transformed_dataset,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EvalDataset":
        r"""Instantiate an EvalDataset from a YAML file.

        The YAML file should contain configuration for loading a dataset from Hugging Face.

        Required fields:
        - path: Hugging Face dataset path
        - name: Name for the evaluation dataset
        - metrics: List of metrics to evaluate

        Input/Label specification (mutually exclusive options):
        - Direct field names:
            input: "question"
            label: "answer"
        - Templates for constructing from multiple fields:
            templates:
              input: "{{ question }}\nOptions: {{ options }}"
              label: "{{ answer }}"

        Optional fields:
        - split: Dataset split to load (e.g., "test")
        - config: Dataset configuration name

        Returns:
            An EvalDataset instance configured according to the YAML file.

        Raises:
            ValueError: If the YAML file is invalid or missing required fields.
        """
        validated_path = validate_path(path, expected_suffix=".yaml")

        try:
            with validated_path.open("r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        # Validate required fields
        required_fields = ["path", "name", "metrics"]
        missing_fields = [field for field in required_fields if field not in yaml_config]
        if missing_fields:
            raise ValueError(f"Missing required fields in YAML config: {', '.join(missing_fields)}")

        # Determine input/label specification
        has_templates = "templates" in yaml_config
        has_direct_input = "input" in yaml_config
        has_direct_label = "label" in yaml_config

        # Validate that we have proper input/label specification
        if has_templates:
            templates = yaml_config["templates"]
            if not isinstance(templates, dict):
                raise ValueError("'templates' must be a dictionary")
            if "input" not in templates or "label" not in templates:
                raise ValueError("'templates' must contain both 'input' and 'label' keys")
            if has_direct_input or has_direct_label:
                raise ValueError(
                    "Cannot specify both 'templates' and direct 'input'/'label' fields"
                )
            input_template = templates["input"]
            label_template = templates["label"]
            input_field = None
            label_field = None
        else:
            if not has_direct_input or not has_direct_label:
                raise ValueError(
                    "Must specify either 'templates' or both 'input' and 'label' fields"
                )
            input_field = yaml_config["input"]
            label_field = yaml_config["label"]
            input_template = None
            label_template = None

        # Load the dataset from Hugging Face
        return cls.from_huggingface(
            path=yaml_config["path"],
            metrics=yaml_config["metrics"],
            input=input_field,
            input_template=input_template,
            label=label_field,
            label_template=label_template,
            name=yaml_config.get("name"),
            split=yaml_config.get("split"),
            config=yaml_config.get("config"),
        )

    @staticmethod
    def _apply_templates(
        data: List[Dict[str, Any]],
        input_field: str,
        label_field: str,
        input_template: Optional[str] = None,
        label_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply templates to transform multi-column data into 2-column format (input, label).

        For each item in the data:
        - If input_template is provided, renders it using the item's data
        - Otherwise, extracts the value from input_field
        - If label_template is provided, renders it using the item's data
        - Otherwise, extracts the value from label_field

        Args:
            data: List of dictionaries containing raw data
            input_field: Field name to extract input from (if no template)
            label_field: Field name to extract label from (if no template)
            input_template: Optional Jinja2 template string for input
            label_template: Optional Jinja2 template string for label

        Returns:
            List of dictionaries with only "input" and "label" keys

        Raises:
            KeyError: If a required field is missing from an item
            jinja2.exceptions.UndefinedError: If template references undefined variable
        """
        result = []
        for item in data:
            # Determine input value
            if input_template is not None:
                input_value = render_template(input_template, item)
            else:
                input_value = item[input_field]

            # Determine label value
            if label_template is not None:
                label_value = render_template(label_template, item)
            else:
                label_value = item[label_field]

            result.append({"input": input_value, "label": label_value})

        return result

    @staticmethod
    def _resolve_metrics(
        metrics: Union[
            str, Type[MetricBase], MetricBase, List[Union[str, Type[MetricBase], MetricBase]]
        ]
    ) -> List[MetricBase]:
        """
        Convert metric names/classes into a list of MetricBase instances using MetricRegistry.

        Used to normalize metrics to a metric type.
        """
        if not isinstance(metrics, list):
            metrics = [metrics]

        resolved: List[MetricBase] = []
        for m in metrics:
            if isinstance(m, MetricBase):
                resolved.append(m)  # Already an instance
            else:
                resolved.append(MetricRegistry.get(m))  # Use registry for str or class

        return resolved

    def sample(self, sample_size: int) -> "EvalDataset":
        """Create a new dataset with randomly sampled items from this dataset.

        Args:
            sample_size: The number of items to sample from the dataset

        Returns:
            A new EvalDataset with randomly sampled items

        Raises:
            ValueError: If sample_size is larger than the dataset size
        """
        dataset_size = len(self.items)

        if sample_size > dataset_size:
            raise ValueError(
                f"Sample size {sample_size} is larger than dataset size {dataset_size} "
                f"for dataset '{self.name}'"
            )

        # Create randomly sampled items
        sampled_items = random.sample(self.items, sample_size)

        # Create a new EvalDataset instance with sampled items
        # Since items already have "input" and "label" columns, we can use from_list directly
        sampled_dataset = self.from_list(
            name=self.name,
            metrics=self.metrics,
            items=sampled_items,
            input="input",
            label="label",
        )

        return sampled_dataset
