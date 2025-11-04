"""Evaluation Dataset implementation for scorebook."""

import csv
import json
import random
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import yaml
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict
from datasets import load_dataset

from scorebook.exceptions import (
    DatasetConfigurationError,
    DatasetLoadError,
    DatasetNotInitializedError,
    DatasetParseError,
    DatasetSampleError,
    MissingFieldError,
)
from scorebook.metrics import MetricBase, MetricRegistry
from scorebook.utils import render_template, validate_path


class EvalDataset:
    """Evaluation Dataset for model evaluation and scoring.

    An evaluation dataset defines explicit input and label features.
    During evaluation, each input is passed to the model,
    and the resulting output is compared against the
    corresponding label using the configured metrics.

    Do not instantiate directly. Use a factory constructor:
        - from_list
        - from_csv
        - from_json
        - from_huggingface
        - from_yaml

    Attributes:
        name: Human-readable dataset name.
        metrics: List of MetricBase instances used for scoring.
        input: Column name used as the model input.
        label: Column name used as the ground-truth label.
        input_template: Optional Jinja2 template that renders the input from item features.
        label_template: Optional Jinja2 template that renders the label from item features.
    """

    def __init__(
        self,
        name: str,
        metrics: Union[str, Type[MetricBase], List[Union[str, Type[MetricBase]]]],
        hf_dataset: HuggingFaceDataset,
        input: Optional[str] = None,
        label: Optional[str] = None,
        input_template: Optional[str] = None,
        label_template: Optional[str] = None,
    ):
        """Create a new scorebook evaluation dataset instance.

        Args:
            name: The name of the evaluation dataset.
            metrics: The metrics used for scoring.
            hf_dataset: Evaluation items.
            input: Dataset feature containing input values.
            label: Dataset feature containing label values.
            input_template: Jinja2 template for input.
            label_template: Jinja2 template for label.

        Raises:
            DatasetConfigurationError:
                If both/neither of input and input_template,
                or both/neither of label and label_template are provided.
            MissingFieldError:
                If the resolved input or label column is not present in the HF dataset.
        """

        # Validate mutual exclusivity for input and input_template
        if (input is None) == (input_template is None):
            raise DatasetConfigurationError(
                "Exactly one of 'input' or 'input_template' must be provided, not both or neither."
            )

        # Validate mutual exclusivity for label and label_template
        if (label is None) == (label_template is None):
            raise DatasetConfigurationError(
                "Exactly one of 'label' or 'label_template' must be provided, not both or neither."
            )

        # Determine the feature to be used as inputs
        input_column: str = (
            "*input" if input_template is not None else input  # type: ignore[assignment]
        )

        # Determine the feature to be used as labels
        label_column: str = (
            "*label" if label_template is not None else label  # type: ignore[assignment]
        )

        # Validate that dataset has the required columns
        column_names = list(hf_dataset.column_names)
        actual_columns = set(column_names)

        if input_column not in actual_columns:
            raise MissingFieldError(
                field_name=input_column, field_type="input", available_fields=column_names
            )

        if label_column not in actual_columns:
            raise MissingFieldError(
                field_name=label_column, field_type="label", available_fields=column_names
            )

        self.name: str = name
        self.metrics: List[MetricBase] = self._resolve_metrics(metrics)
        self._hf_dataset: Optional[HuggingFaceDataset] = hf_dataset

        # Store which columns to use for input/label
        self.input: str = input_column
        self.label: str = label_column

        # Store templates for transparency (optional, for debugging)
        self.input_template: Optional[str] = input_template
        self.label_template: Optional[str] = label_template

    @property
    def items(self) -> List[Any]:
        """Return a list of all examples in the dataset."""
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        return list(self._hf_dataset)

    @property
    def column_names(self) -> List[str]:
        """Return a list of column/feature names available in the dataset."""
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        return list(map(str, self._hf_dataset.column_names))

    @property
    def split(self) -> Optional[str]:
        """Return the split name of the underlying HuggingFace dataset, if available.

        Returns:
            The split name (e.g., "train", "test", "validation") if the dataset was loaded
            from HuggingFace with a specific split. Returns None if the dataset was created
            from a list, CSV, JSON, or loaded without a split specification.

        Raises:
            DatasetNotInitializedError: If the dataset is not initialized.
        """
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")

        split = self._hf_dataset.split
        return str(split) if split is not None else None

    def shuffle(self) -> None:
        """Randomly shuffle the dataset items."""
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        self._hf_dataset.shuffle()

    def sample(self, sample_size: int) -> "EvalDataset":
        """Create a new dataset with randomly sampled items from this dataset.

        Args:
            sample_size: The number of items to sample from the dataset.

        Returns:
            A new EvalDataset with randomly sampled items.

        Raises:
            DatasetSampleError: If the sample size is smaller than the dataset.
        """

        # Validate requested sample size against available items
        dataset_size = len(self.items)
        if sample_size > dataset_size:
            raise DatasetSampleError(
                sample_size=sample_size, dataset_size=dataset_size, dataset_name=self.name
            )

        # Create randomly sampled items
        sampled_items = random.sample(self.items, sample_size)

        # Create HuggingFace dataset from sampled items
        sampled_hf_dataset = HuggingFaceDataset.from_list(sampled_items)

        # # Preserve original input/label spec; omit field names when templates are used
        input_param = None if self.input_template else self.input
        label_param = None if self.label_template else self.label

        return EvalDataset(
            name=self.name,
            metrics=self.metrics,
            hf_dataset=sampled_hf_dataset,
            input=input_param,
            label=label_param,
            input_template=self.input_template,
            label_template=self.label_template,
        )

    # === Factory Methods ===

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

        Args:
            name: The name of the evaluation dataset.
            metrics: The specified metrics associated with the dataset.
            items: List of dictionaries containing the dataset examples.
            input: The field name containing the input data.
            label: The field name containing the label.

        Returns:
            A scorebook EvalDataset.

        Raises:
            MissingFieldError: If the input or label feature is not present in the first item.
        """

        if items and items[0]:
            available_fields = list(items[0].keys())

            # Raise an error if the input feature is missing from the first item
            if input not in items[0]:
                raise MissingFieldError(
                    field_name=input, field_type="input", available_fields=available_fields
                )

            # Raises an error if the label feature is missing from the first item
            if label not in items[0]:
                raise MissingFieldError(
                    field_name=label, field_type="label", available_fields=available_fields
                )

        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(items),
            input=input,
            label=label,
            input_template=None,
            label_template=None,
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
        """Instantiate an EvalDataset from a CSV file.

        Args:
            path: Path to the CSV file.
            metrics: The specified metrics associated with the dataset.
            input: The field name containing the input data.
            label: The field name containing the label.
            name: Optional name for the eval dataset, if not provided, the path is used.
            encoding: Encoding of the CSV file.
            newline: Newline character of the CSV file.
            reader_kwargs: Dict of kwargs passed to csv.DictReader.

        Returns:
            A scorebook EvalDataset.

        Raises:
            DatasetParseError: If csv parsing fails.
            DatasetLoadError: If the csv file does not contain evaluation items.
            MissingFieldError: If the input or label feature is not present in the first item.
        """
        reader_kwargs = reader_kwargs or {}
        validated_path = validate_path(path, expected_suffix=".csv")

        try:
            with open(validated_path, encoding=encoding, newline=newline) as csvfile:
                items = list(csv.DictReader(csvfile, **reader_kwargs))
        except csv.Error as e:
            raise DatasetParseError(f"Failed to parse CSV file {path}: {e}") from e

        if not items:
            raise DatasetLoadError(f"CSV file {path} is empty or contains only headers.")

        available_fields = list(items[0].keys())
        if input not in items[0]:
            raise MissingFieldError(
                field_name=input, field_type="input", available_fields=available_fields
            )
        if label not in items[0]:
            raise MissingFieldError(
                field_name=label, field_type="label", available_fields=available_fields
            )

        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(items),
            input=input,
            label=label,
            input_template=None,
            label_template=None,
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

        1. Flat format – a list of dictionaries:
            [
                {"input": ..., "output": ...},
                {"input": ..., "output": ...},
            ]

        2. Split format – a dictionary of named splits:
            {
                "train": [{"input": ..., "output": ...}],
                "test": [{"input": ..., "output": ...}]
            }

        Args:
            path: Path to the JSON file on disk.
            metrics: The specified metrics associated with the dataset.
            input: The field name containing the input data.
            label: The field name containing the label.
            name: Optional name for the eval dataset, if not provided, the path is used
            split: If the JSON uses a split structure, this is the split name to load.

        Returns:
            A Scorebook EvalDataset.

        Raises:
            DatasetParseError: If JSON parsing fails.
            DatasetConfigurationError: If an invalid split is provided.
            MissingFieldError: If the input or label feature is not present in the first item.
        """
        validated_path = validate_path(path, expected_suffix=".json")

        try:
            with validated_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetParseError(f"Invalid JSON in {path}: {e}") from e

        if isinstance(json_data, dict):

            if split is None:
                raise DatasetConfigurationError(
                    f"Split name must be provided for split-style JSON: {path}"
                )

            items = json_data.get(split)
            if items is None:
                raise DatasetConfigurationError(f"Split '{split}' not found in JSON file: {path}")
            if not isinstance(items, list):
                raise DatasetConfigurationError(
                    f"Split '{split}' is not a list of examples in: {path}"
                )

        elif isinstance(json_data, list):
            items = json_data

        else:
            raise DatasetConfigurationError(
                f"Unsupported JSON structure in {path}. Expected list or dict."
            )

        # Validate that fields exist
        if items and items[0]:
            available_fields = list(items[0].keys())
            if input not in items[0]:
                raise MissingFieldError(
                    field_name=input, field_type="input", available_fields=available_fields
                )
            if label not in items[0]:
                raise MissingFieldError(
                    field_name=label, field_type="label", available_fields=available_fields
                )

        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(items),
            input=input,
            label=label,
            input_template=None,
            label_template=None,
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

        For datasets where the input/label is already in a single column, use the input/label
        parameters to specify the feature names. For datasets where the input/label needs to be
        constructed from multiple columns, use the input_template/label_template parameters
        with Jinja2 template strings.

        Args:
            path: The path of the dataset on the Hugging Face Hub.
            metrics: The specified metrics associated with the dataset.
            input: Field name containing the input data (mutually exclusive with input_template).
            input_template:
                Jinja2 template to construct input from multiple fields
                (mutually exclusive with input).
            label: Field name containing the label
                (mutually exclusive with label_template).
            label_template:
                Jinja2 template to construct label from multiple fields
                (mutually exclusive with label).
            name: Optional name for the eval dataset, by default HF "path:split:config" is used.
            split: Optional name of the split to load.
            config: Optional dataset configuration name.

        Returns:
            A Scorebook EvalDataset.

        Raises:
            DatasetConfigurationError:
                If both/neither of input and input_template,
                or both/neither of label and label_template are provided.
            DatasetLoadError: If HF dataset cannot be loaded.
        """

        # Validate mutual exclusivity for input and input_template
        if (input is None) == (input_template is None):
            raise DatasetConfigurationError(
                "Exactly one of 'input' or 'input_template' must be provided, not both or neither."
            )

        # Validate mutual exclusivity for label and label_template
        if (label is None) == (label_template is None):
            raise DatasetConfigurationError(
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
            raise DatasetLoadError(f"Failed to load dataset '{path}' from Hugging Face: {e}") from e

        if isinstance(ds, HuggingFaceDataset):
            hf_dataset = ds
        elif isinstance(ds, HuggingFaceDatasetDict):
            if "test" in ds:
                hf_dataset = ds["test"]
            else:
                raise DatasetConfigurationError(
                    f"Split not specified and no 'test' split found in dataset '{path}'."
                )
        else:
            raise DatasetConfigurationError(f"Unexpected dataset type for '{path}': {type(ds)}")

        # Only transform if templates are used
        if input_template is not None or label_template is not None:

            def transform_row(row: Dict[str, Any]) -> Dict[str, Any]:
                """Add computed columns (*input, *label) when templates are used."""
                # Start with all original data
                result = dict(row)

                # Add *input if template is used
                if input_template is not None:
                    result["*input"] = render_template(input_template, row)

                # Add *label if template is used
                if label_template is not None:
                    result["*label"] = render_template(label_template, row)

                return result

            transformed_dataset = hf_dataset.map(transform_row)
        else:

            transformed_dataset = hf_dataset

        dataset_name = name if name else ":".join(filter(None, [path, split, config]))
        return cls(
            name=dataset_name,
            metrics=metrics,
            hf_dataset=transformed_dataset,
            input=input,
            label=label,
            input_template=input_template,
            label_template=label_template,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EvalDataset":
        r"""Instantiate an EvalDataset from Huggingface with a YAML Config file.

        The YAML file should contain configuration for loading a dataset from Hugging Face.

        Required fields:
        - path: Hugging Face dataset path
        - name: Name for the evaluation dataset
        - metrics: List of metrics to evaluate

        The input / label features must be specified / constructed by one of the following:

        1. Feature Specification:
            input: "question"
            label: "answer"

        2. Mapping Templates:
            templates:
              input: "{{ question }}\nOptions: {{ options }}"
              label: "{{ answer }}"

        Optional fields:
        - split: Dataset split to load.
        - config: Dataset configuration name.
        - metadata: Any additional metadata.

        Args:
            path: The path of YAML configuration file.

        Returns:
            An EvalDataset.

        Raises:
            DatasetParseError: If YAML configuration file cannot be parsed.
            DatasetConfigurationError: Invalid YAML configuration file.
        """
        validated_path = validate_path(path, expected_suffix=(".yaml", ".yml"))

        try:
            with validated_path.open("r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise DatasetParseError(f"Invalid YAML in {path}: {e}") from e

        # Validate required fields
        required_fields = ["path", "name", "metrics"]
        missing_fields = [field for field in required_fields if field not in yaml_config]
        if missing_fields:
            raise DatasetConfigurationError(
                f"Missing required fields in YAML config: {', '.join(missing_fields)}"
            )

        # Validate metrics exist before calling from_huggingface
        metrics_to_validate = yaml_config["metrics"]
        if not isinstance(metrics_to_validate, list):
            metrics_to_validate = [metrics_to_validate]

        for metric in metrics_to_validate:
            try:
                MetricRegistry.get(metric)
            except Exception as e:
                raise DatasetConfigurationError(f"Invalid metric '{metric}' in YAML config: {e}")

        # Determine input/label specification
        has_templates = "templates" in yaml_config
        has_direct_input = "input" in yaml_config
        has_direct_label = "label" in yaml_config

        # Validate that we have proper input/label specification
        if has_templates:
            templates = yaml_config["templates"]
            if not isinstance(templates, dict):
                raise DatasetConfigurationError("'templates' must be a dictionary")
            if "input" not in templates or "label" not in templates:
                raise DatasetConfigurationError(
                    "'templates' must contain both 'input' and 'label' keys"
                )
            if has_direct_input or has_direct_label:
                raise DatasetConfigurationError(
                    "Cannot specify both 'templates' and direct 'input'/'label' fields"
                )
            input_template = templates["input"]
            label_template = templates["label"]
            input_field = None
            label_field = None
        else:
            if not has_direct_input or not has_direct_label:
                raise DatasetConfigurationError(
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

    # === Helper Methods ===

    @staticmethod
    def _resolve_metrics(
        metrics: Union[
            str, Type[MetricBase], MetricBase, List[Union[str, Type[MetricBase], MetricBase]]
        ]
    ) -> List[MetricBase]:
        """Normalize metrics params to a metric type."""

        if not isinstance(metrics, list):
            metrics = [metrics]

        resolved: List[MetricBase] = []
        for m in metrics:
            if isinstance(m, MetricBase):
                resolved.append(m)  # Already an instance
            else:
                resolved.append(MetricRegistry.get(m))  # Use registry for str or class

        return resolved

    # === Dunder Methods ===

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        return len(self._hf_dataset)

    def __getitem__(self, key: Union[int, str]) -> Union[Dict[str, Any], List[Any]]:
        """
        Allow item access by index (int) or by column name (str).

        - eval_dataset[i] returns the i-th example (dict).
        - eval_dataset["feature"] returns a list of values for that feature.
        """
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        if isinstance(key, int):
            return dict(self._hf_dataset[key])  # Ensure we return a Dict[str, Any]
        elif isinstance(key, str):
            return list(self._hf_dataset[key])  # Ensure we return a List[Any]
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Must be int or str.")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return an iterator over all examples in the dataset."""
        if self._hf_dataset is None:
            raise DatasetNotInitializedError("Dataset is not initialized")
        return iter(self._hf_dataset)

    def __str__(self) -> str:
        """Return a formatted string summary of the evaluation dataset."""
        if self._hf_dataset is None:
            return f"EvalDataset(name='{self.name}', status='uninitialized')"

        num_rows = len(self._hf_dataset)
        fields = ", ".join(self.column_names)
        metrics = ", ".join([metric.name for metric in self.metrics])

        # Build template info string
        template_info = []
        if self.input_template:
            template_preview = (
                self.input_template[:40] + "..."
                if len(self.input_template) > 40
                else self.input_template
            )
            template_info.append(f"input_template='{template_preview}'")

        if self.label_template:
            template_preview = (
                self.label_template[:40] + "..."
                if len(self.label_template) > 40
                else self.label_template
            )
            template_info.append(f"label_template='{template_preview}'")

        template_str = ", " + ", ".join(template_info) if template_info else ""

        return (
            f"EvalDataset(\n"
            f"  name='{self.name}',\n"
            f"  rows={num_rows},\n"
            f"  fields=[{fields}],\n"
            f"  metrics=[{metrics}],\n"
            f"  input='{self.input}',\n"
            f"  label='{self.label}'{template_str}\n"
            f")"
        )
