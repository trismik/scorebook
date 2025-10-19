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
    """Eval Dataset implementation for scorebook.

    EvalDatasets wrap HuggingFace datasets and provide a consistent interface for
    evaluation tasks. Each dataset must specify input and label columns, either
    directly or via Jinja2 templates.

    Important: Column Naming Convention
    ------------------------------------
    When using templates (input_template/label_template), the computed columns are
    named with a special prefix:
    - input_template → creates column "*input"
    - label_template → creates column "*label"

    Original columns are preserved alongside the computed columns. When debugging,
    you may see these "*input" and "*label" column names in error messages or when
    inspecting dataset.column_names or dataset.input/dataset.label attributes.

    Example:
        # Using templates
        dataset = EvalDataset.from_huggingface(
            path="squad",
            input_template="{{ question }} Context: {{ context }}",
            label="answers"
        )
        # dataset.input will be "*input" (computed column)
        # dataset.label will be "answers" (direct field)
        # Both "*input" and original "question", "context" columns exist
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
        """
        Create a new scorebook evaluation dataset instance.

        All EvalDatasets must have the specified input and label columns,
        but may contain any additional columns from the original data source.

        :param name: The name of the evaluation dataset.
        :param metrics: The specified metrics associated with the dataset.
        :param hf_dataset: The dataset as a hugging face dataset object.
        :param input: Field name for input (mutually exclusive with input_template).
        :param label: Field name for label (mutually exclusive with label_template).
        :param input_template: Jinja2 template for input (mutually exclusive with input).
        :param label_template: Jinja2 template for label (mutually exclusive with label).

        :raises ValueError: If validation fails.
        """
        # Validate mutual exclusivity for input
        if (input is None) == (input_template is None):
            raise ValueError(
                "Exactly one of 'input' or 'input_template' must be provided, not both or neither."
            )

        # Validate mutual exclusivity for label
        if (label is None) == (label_template is None):
            raise ValueError(
                "Exactly one of 'label' or 'label_template' must be provided, not both or neither."
            )

        # Determine column names to use
        input_column: str = (
            "*input" if input_template is not None else input  # type: ignore[assignment]
        )
        label_column: str = (
            "*label" if label_template is not None else label  # type: ignore[assignment]
        )

        # Validate that dataset has the required columns
        column_names = list(hf_dataset.column_names)
        required_columns = {input_column, label_column}
        actual_columns = set(column_names)

        if not required_columns.issubset(actual_columns):
            raise ValueError(
                f"EvalDataset must have columns '{input_column}' and '{label_column}'. "
                f"Got: {column_names}"
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

        All original columns are preserved. Since from_list only supports direct
        field extraction (no templates), the specified fields must exist in the data.

        Args:
            name: The name of the evaluation dataset.
            metrics: The specified metrics associated with the dataset.
            items: List of dictionaries containing the dataset examples.
            input: The field name containing the input data.
            label: The field name containing the label (ground truth).

        Returns:
            A scorebook EvalDataset with all original columns preserved.

        Raises:
            KeyError: If input or label field is missing from the first item.
        """
        # Validate that fields exist in the data
        if items and items[0]:
            if input not in items[0]:
                raise KeyError(f"Input field '{input}' not found in data")
            if label not in items[0]:
                raise KeyError(f"Label field '{label}' not found in data")

        # No transformation - use data as-is!
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
        """Instantiate a scorebook dataset from a CSV file.

        All original columns are preserved. Since from_csv only supports direct
        field extraction (no templates), the specified fields must exist in the data.

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
            A scorebook EvalDataset with all original columns preserved.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the CSV file cannot be parsed or is empty.
            KeyError: If input or label field is missing from the first row.
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

        # Validate that fields exist
        if data and data[0]:
            if input not in data[0]:
                raise KeyError(f"Input field '{input}' not found in CSV")
            if label not in data[0]:
                raise KeyError(f"Label field '{label}' not found in CSV")

        # No transformation - use data as-is!
        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(data),
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

        All original columns are preserved. Since from_json only supports direct
        field extraction (no templates), the specified fields must exist in the data.

        Args:
            path: Path to the JSON file on disk.
            metrics: The specified metrics associated with the dataset.
            input: The field name containing the input data.
            label: The field name containing the label (ground truth).
            name: Optional name for the eval dataset, if not provided, the path is used
            split: If the JSON uses a split structure, this is the split name to load.

        Returns:
            A scorebook EvalDataset with all original columns preserved.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or the structure is unsupported.
            KeyError: If input or label field is missing from the first item.
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

        # Validate that fields exist
        if raw_data and raw_data[0]:
            if input not in raw_data[0]:
                raise KeyError(f"Input field '{input}' not found in JSON")
            if label not in raw_data[0]:
                raise KeyError(f"Label field '{label}' not found in JSON")

        # No transformation - use data as-is!
        name = name if name else validated_path.stem
        return cls(
            name=name,
            metrics=metrics,
            hf_dataset=HuggingFaceDataset.from_list(raw_data),
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

            # Apply transformation - DON'T remove any columns
            transformed_dataset = hf_dataset.map(transform_row)
        else:
            # No templates - use dataset as-is (optimization!)
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
        validated_path = validate_path(path, expected_suffix=(".yaml", ".yml"))

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

        # Validate metrics exist before calling from_huggingface (fail fast)
        metrics_to_validate = yaml_config["metrics"]
        if not isinstance(metrics_to_validate, list):
            metrics_to_validate = [metrics_to_validate]

        for metric in metrics_to_validate:
            try:
                MetricRegistry.get(metric)
            except Exception as e:
                raise ValueError(f"Invalid metric '{metric}' in YAML config: {e}")

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

        Preserves all columns and the input/label specifications from the
        original dataset.

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

        # Create HuggingFace dataset from sampled items
        sampled_hf_dataset = HuggingFaceDataset.from_list(sampled_items)

        # Determine which parameters to pass based on what we have
        # If we have templates, the column is "*input" or "*label"
        # So we pass None for input/label field names
        if self.input_template is not None:
            input_param = None
        else:
            input_param = self.input

        if self.label_template is not None:
            label_param = None
        else:
            label_param = self.label

        # Create new EvalDataset with same specifications as original
        sampled_dataset = EvalDataset(
            name=self.name,
            metrics=self.metrics,
            hf_dataset=sampled_hf_dataset,
            input=input_param,
            label=label_param,
            input_template=self.input_template,
            label_template=self.label_template,
        )

        return sampled_dataset
