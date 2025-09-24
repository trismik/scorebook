---
id: scorebook_metrics_metric_base
title: scorebook.metrics.metric_base
sidebar_label: metric_base
---

<a id="scorebook.metrics.metric_registry"></a>

# scorebook.metrics.metric\_registry

Registry module for evaluation metrics.

This module maintains a centralized registry of available evaluation metrics
that can be used to assess model performance. It provides a single access point
to retrieve all implemented metric classes.

<a id="scorebook.metrics.metric_registry.MetricRegistry"></a>

## MetricRegistry Objects

```python
class MetricRegistry()
```

A registry for evaluation metrics.

This class provides a central registry for all evaluation metrics in the system.
It allows metrics to be registered with unique names and retrieved either by
name or by class. The registry ensures that metrics are properly initialized
and accessible throughout the application.

The registry supports:
- Registering new metric classes with optional custom names
- Retrieving metric instances by name or class
- Listing all available metrics

Usage:
    @MetricRegistry.register("custom_name")
    class MyMetric(MetricBase):
        ...

    # Get by name
    metric = MetricRegistry.get("custom_name")

    # Get by class
    metric = MetricRegistry.get(MyMetric)

    # List available metrics
    metrics = MetricRegistry.list_metrics()

<a id="scorebook.metrics.metric_registry.MetricRegistry.register"></a>

#### register

```python
@classmethod
def register(cls) -> Callable[[Type[MetricBase]], Type[MetricBase]]
```

Register a metric class in the registry.

**Returns**:

  A decorator that registers the class and returns it.


**Raises**:

- `ValueError` - If a metric with the given name is already registered.

<a id="scorebook.metrics.metric_registry.MetricRegistry.get"></a>

#### get

```python
@classmethod
def get(cls, name_or_class: Union[str, Type[MetricBase]],
        **kwargs: Any) -> MetricBase
```

Get an instance of a registered metric by name or class.

**Arguments**:

- `name_or_class` - The metric name (string) or class (subclass of BaseMetric).
- `**kwargs` - Additional arguments to pass to the metric's constructor.


**Returns**:

  An instance of the requested metric.


**Raises**:

- `ValueError` - If the metric name is not registered.

<a id="scorebook.metrics.metric_registry.MetricRegistry.list_metrics"></a>

#### list\_metrics

```python
@classmethod
def list_metrics(cls) -> List[str]
```

List all registered metrics.

**Returns**:

  A list of metric names.

<a id="scorebook.metrics.precision"></a>

# scorebook.metrics.precision

Precision metric implementation for Scorebook.

<a id="scorebook.metrics.precision.Precision"></a>

## Precision Objects

```python
@MetricRegistry.register()
class Precision(MetricBase)
```

Precision metric for binary classification.

Precision = TP / (TP + FP)

<a id="scorebook.metrics.precision.Precision.score"></a>

#### score

```python
@staticmethod
def score(outputs: List[Any],
          labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]
```

Not implemented.

<a id="scorebook.metrics.metric_base"></a>

# scorebook.metrics.metric\_base

Base class for evaluation metrics.

<a id="scorebook.metrics.metric_base.MetricBase"></a>

## MetricBase Objects

```python
class MetricBase(ABC)
```

Base class for all evaluation metrics.

<a id="scorebook.metrics.metric_base.MetricBase.name"></a>

#### name

```python
@property
def name() -> str
```

Return the metric name based on the class name.

<a id="scorebook.metrics.metric_base.MetricBase.score"></a>

#### score

```python
@staticmethod
@abstractmethod
def score(outputs: List[Any],
          labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]
```

Calculate the metric score for a list of outputs and labels.

**Arguments**:

- `outputs` - A list of inference outputs.
- `labels` - A list of ground truth labels.


**Returns**:

  Aggregate metric scores for all items.
  Individual scores for each item.

<a id="scorebook.metrics.accuracy"></a>

# scorebook.metrics.accuracy

Accuracy metric implementation for Scorebook.

<a id="scorebook.metrics.accuracy.Accuracy"></a>

## Accuracy Objects

```python
@MetricRegistry.register()
class Accuracy(MetricBase)
```

Accuracy metric for evaluating model predictions of any type.

Accuracy = correct predictions / total predictions

<a id="scorebook.metrics.accuracy.Accuracy.score"></a>

#### score

```python
@staticmethod
def score(outputs: List[Any],
          labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]
```

Calculate accuracy score between predictions and references.

**Arguments**:

- `outputs` - A list of inference outputs.
- `labels` - A list of ground truth labels.


**Returns**:

  The aggregate accuracy score for all items (correct predictions / total predictions).
  The item scores for each output-label pair (true/false).

<a id="scorebook.trismik_services.upload_classic_eval_run"></a>

# scorebook.trismik\_services.upload\_classic\_eval\_run

Upload classic evaluation run results to Trismik platform.

<a id="scorebook.trismik_services.upload_classic_eval_run.upload_classic_eval_run"></a>

#### upload\_classic\_eval\_run

```python
async def upload_classic_eval_run(
        run: ClassicEvalRunResult, experiment_id: str, project_id: str,
        model: str,
        metadata: Optional[Dict[str, Any]]) -> TrismikClassicEvalResponse
```

Upload a classic evaluation run result to Trismik platform.

**Arguments**:

- `run` - The evaluation run result to upload
- `experiment_id` - Trismik experiment identifier
- `project_id` - Trismik project identifier
- `model` - Model name used for evaluation
- `metadata` - Optional metadata dictionary


**Returns**:

  Response from Trismik API containing the upload result

<a id="scorebook.trismik_services.login"></a>

# scorebook.trismik\_services.login

Authentication and token management for Trismik API.

<a id="scorebook.trismik_services.login.get_scorebook_config_dir"></a>

#### get\_scorebook\_config\_dir

```python
def get_scorebook_config_dir() -> str
```

Get the scorebook config directory.

<a id="scorebook.trismik_services.login.get_token_path"></a>

#### get\_token\_path

```python
def get_token_path() -> str
```

Get the path where the trismik token is stored.

<a id="scorebook.trismik_services.login.save_token"></a>

#### save\_token

```python
def save_token(token: str) -> None
```

Save the token to the local cache directory.

<a id="scorebook.trismik_services.login.get_stored_token"></a>

#### get\_stored\_token

```python
def get_stored_token() -> Optional[str]
```

Retrieve the stored token from the cache directory.

<a id="scorebook.trismik_services.login.get_token"></a>

#### get\_token

```python
def get_token() -> Optional[str]
```

Get the trismik API token in order of priority.

Priority order:
1. TRISMIK_API_KEY environment variable
2. Stored token file

<a id="scorebook.trismik_services.login.validate_token"></a>

#### validate\_token

```python
def validate_token(token: str) -> bool
```

Validate the token by making a test API call to trismik.

<a id="scorebook.trismik_services.login.login"></a>

#### login

```python
def login(trismik_api_key: str) -> None
```

Login to trismik by saving API key locally.

**Arguments**:

- `trismik_api_key` - The API key to use.

**Raises**:

- `ValueError` - If API key is empty or invalid.

<a id="scorebook.trismik_services.login.logout"></a>

#### logout

```python
def logout() -> bool
```

Remove the stored token.

**Returns**:

- `bool` - True if a token was removed, False if no token was found.

<a id="scorebook.trismik_services.login.whoami"></a>

#### whoami

```python
def whoami() -> Optional[str]
```

Return information about the current user/token.

**Returns**:

- `str` - The stored token if logged in, None if not logged in.

<a id="scorebook.trismik_services.adaptive_testing_service"></a>

# scorebook.trismik\_services.adaptive\_testing\_service

Trismik adaptive testing service integration.

<a id="scorebook.trismik_services.adaptive_testing_service.run_adaptive_evaluation"></a>

#### run\_adaptive\_evaluation

```python
async def run_adaptive_evaluation(inference: Callable,
                                  adaptive_run_spec: AdaptiveEvalRunSpec,
                                  experiment_id: str, project_id: str,
                                  metadata: Any) -> AdaptiveEvalRunResult
```

Run an adaptive evaluation using the Trismik API.

**Arguments**:

- `inference` - Function to run inference
- `adaptive_run_spec` - Specification for the adaptive evaluation
- `experiment_id` - Experiment identifier
- `project_id` - Trismik project ID
- `metadata` - Additional metadata

**Returns**:

  Results from the adaptive evaluation

<a id="scorebook.trismik_services.adaptive_testing_service.make_trismik_inference"></a>

#### make\_trismik\_inference

```python
def make_trismik_inference(inference_function: Callable,
                           return_list: bool = False) -> Callable[[Any], Any]
```

Wrap an inference function for flexible input handling.

Takes a function expecting list[dict] and makes it accept single dict
or TrismikMultipleChoiceTextItem.

<a id="scorebook.utils.async_utils"></a>

# scorebook.utils.async\_utils

Async utilities for handling callable objects and coroutines.

<a id="scorebook.utils.async_utils.is_awaitable"></a>

#### is\_awaitable

```python
def is_awaitable(obj: Callable) -> bool
```

Check if a callable object is awaitable.

This handles both coroutine functions and callable instances (like classes
with __call__ methods) that may return coroutines.

**Arguments**:

- `obj` - The callable object to check


**Returns**:

  True if the object is awaitable, False otherwise

<a id="scorebook.utils.build_prompt"></a>

# scorebook.utils.build\_prompt

Module for building prompt strings using Jinja2 templating.

Provides functionality to render prompts from templates with custom filters
and global variables, using strict undefined handling for better error detection.

<a id="scorebook.utils.build_prompt.build_prompt"></a>

#### build\_prompt

```python
def build_prompt(prompt_template: str,
                 prompt_args: Dict[str, Any],
                 filters: Optional[Dict[str, Any]] = None,
                 globals_dict: Optional[Dict[str, Any]] = None) -> str
```

Build a prompt string from a template and arguments.

**Arguments**:

- `prompt_template` - Jinja2 template string
- `prompt_args` - Dictionary of arguments to pass to the template
- `filters` - Dictionary of Jinja2 filters. Defaults to default_jinja_filters().
- `globals_dict` - Dictionary of global functions/variables. Defaults to default_jinja_globals().


**Returns**:

- `str` - Rendered prompt string

<a id="scorebook.utils.jinja_helpers"></a>

# scorebook.utils.jinja\_helpers

Jinja2 template helper functions for Scorebook.

<a id="scorebook.utils.jinja_helpers.number_to_letter"></a>

#### number\_to\_letter

```python
def number_to_letter(index: int, uppercase: bool = True) -> str
```

Convert a number to a letter (0->A, 1->B, etc.).

**Arguments**:

- `index` - The number to convert to a letter (0-based index, must be 0-25)
- `uppercase` - If True, returns uppercase letter; if False, returns lowercase


**Returns**:

- `str` - A letter from A-Z (or a-z if uppercase is False)


**Raises**:

- `ValueError` - If index is less than 0 or greater than 25

<a id="scorebook.utils.jinja_helpers.letter_to_number"></a>

#### letter\_to\_number

```python
def letter_to_number(letter: str) -> int
```

Convert a letter to a number (A->0, B->1, etc.).

**Arguments**:

- `letter` - A single letter character (A-Z or a-z)


**Returns**:

- `int` - The zero-based position of the letter in the alphabet


**Raises**:

- `ValueError` - If the input is not a single letter character

<a id="scorebook.utils.jinja_helpers.format_list"></a>

#### format\_list

```python
def format_list(items: List[Any],
                separator: str = ", ",
                last_separator: str = " and ") -> str
```

Format a list with proper separators and conjunction.

**Examples**:

  format_list(["a", "b", "c"]) -> "a, b and c"
  format_list(["a", "b"]) -> "a and b"
  format_list(["a"]) -> "a"

<a id="scorebook.utils.jinja_helpers.truncate_text"></a>

#### truncate\_text

```python
def truncate_text(text: str, max_length: int, suffix: str = "...") -> str
```

Truncate text to a maximum length with optional suffix.

<a id="scorebook.utils.jinja_helpers.format_number"></a>

#### format\_number

```python
def format_number(number: float, precision: int = 2) -> str
```

Format a number with specified decimal places.

<a id="scorebook.utils.jinja_helpers.extract_initials"></a>

#### extract\_initials

```python
def extract_initials(text: str) -> str
```

Extract initials from a text string.

**Examples**:

  extract_initials("John Doe") -> "JD"
  extract_initials("Machine Learning Model") -> "MLM"

<a id="scorebook.utils.jinja_helpers.json_pretty"></a>

#### json\_pretty

```python
def json_pretty(obj: Any, indent: int = 2) -> str
```

Pretty-print an object as JSON.

<a id="scorebook.utils.jinja_helpers.percentage"></a>

#### percentage

```python
def percentage(value: float, total: float, precision: int = 1) -> str
```

Calculate and format a percentage.

**Examples**:

  percentage(25, 100) -> "25.0%"
  percentage(1, 3, 2) -> "33.33%"

<a id="scorebook.utils.jinja_helpers.ordinal"></a>

#### ordinal

```python
def ordinal(n: int) -> str
```

Convert number to ordinal format like 1st, 2nd, 3rd, etc.

<a id="scorebook.utils.jinja_helpers.default_jinja_globals"></a>

#### default\_jinja\_globals

```python
def default_jinja_globals() -> Dict[str, Any]
```

Get default global functions for Jinja templates.

<a id="scorebook.utils.jinja_helpers.default_jinja_filters"></a>

#### default\_jinja\_filters

```python
def default_jinja_filters() -> Dict[str, Any]
```

Get default filters for Jinja templates.

<a id="scorebook.utils.io_helpers"></a>

# scorebook.utils.io\_helpers

Input/output helper functions for Scorebook.

<a id="scorebook.utils.io_helpers.validate_path"></a>

#### validate\_path

```python
def validate_path(file_path: str,
                  expected_suffix: Optional[str] = None) -> Path
```

Validate that a file path exists and optionally check its suffix.

**Arguments**:

- `file_path` - Path to the file as string or Path object
- `expected_suffix` - Optional file extension to validate (e.g. ".json", ".csv")


**Returns**:

  Path object for the validated file path


**Raises**:

- `FileNotFoundError` - If the file does not exist
- `ValueError` - If the file has the wrong extension

<a id="scorebook.utils.mappers"></a>

# scorebook.utils.mappers

Utility functions for mapping and converting data types in Scorebook.

<a id="scorebook.utils.mappers.to_binary"></a>

#### to\_binary

```python
def to_binary(value: Any) -> int
```

Transform various input types to binary (0/1) classification value.

<a id="scorebook.utils.mappers.to_binary_classification"></a>

#### to\_binary\_classification

```python
def to_binary_classification(prediction: Any,
                             reference: Any) -> ClassificationResult
```

Determine classification result based on prediction and reference values.

**Arguments**:

- `prediction` - Predicted value (will be converted to binary)
- `reference` - Reference/true value (will be converted to binary)


**Returns**:

  Classification result as one of: "true_positive", "false_positive",
  "true_negative", "false_negative"

<a id="scorebook.utils.transform_helpers"></a>

# scorebook.utils.transform\_helpers

Utility functions for transforming and manipulating data structures.

<a id="scorebook.utils.transform_helpers.expand_dict"></a>

#### expand\_dict

```python
def expand_dict(data: dict) -> list[dict]
```

Expand a dictionary with list values into multiple dictionaries.

Takes a dictionary that may contain list values and expands it into a list of dictionaries,
where each dictionary represents one possible combination of values from the lists.
Non-list values remain constant across all generated dictionaries.

**Arguments**:

- `data` - A dictionary potentially containing list values to be expanded


**Returns**:

  A list of dictionaries representing all possible combinations of the input values

<a id="scorebook.utils.progress_bars"></a>

# scorebook.utils.progress\_bars

Progress bar utilities for evaluation tracking.

<a id="scorebook.utils.progress_bars.EvaluationProgressBars"></a>

## EvaluationProgressBars Objects

```python
class EvaluationProgressBars()
```

Manages nested progress bars for evaluation tracking.

<a id="scorebook.utils.progress_bars.EvaluationProgressBars.__init__"></a>

#### \_\_init\_\_

```python
def __init__(datasets: List[Any], hyperparam_count: int, parallel: bool,
             total_eval_runs: int) -> None
```

Initialize progress bar manager.

**Arguments**:

- `datasets` - List of datasets being evaluated
- `hyperparam_count` - Number of hyperparameter configurations per dataset
- `parallel` - Whether running in parallel mode
- `total_eval_runs` - Total number of EvalRunSpecs (dataset_count * hyperparam_count)

<a id="scorebook.utils.progress_bars.EvaluationProgressBars.start_progress_bars"></a>

#### start\_progress\_bars

```python
def start_progress_bars() -> None
```

Start both progress bars.

<a id="scorebook.utils.progress_bars.EvaluationProgressBars.on_eval_run_completed"></a>

#### on\_eval\_run\_completed

```python
def on_eval_run_completed(dataset_idx: int) -> None
```

Update progress when an eval run (EvalRunSpec) completes in parallel mode.

<a id="scorebook.utils.progress_bars.EvaluationProgressBars.on_hyperparam_completed"></a>

#### on\_hyperparam\_completed

```python
def on_hyperparam_completed(dataset_idx: int) -> None
```

Update progress when a hyperparameter config completes in sequential mode.

<a id="scorebook.utils.progress_bars.EvaluationProgressBars.close_progress_bars"></a>

#### close\_progress\_bars

```python
def close_progress_bars() -> None
```

Close both progress bars.

<a id="scorebook.utils.progress_bars.evaluation_progress"></a>

#### evaluation\_progress

```python
@contextmanager
def evaluation_progress(
        datasets: List[Any], hyperparam_count: int, parallel: bool,
        total_eval_runs: int) -> Generator[EvaluationProgressBars, None, None]
```

Context manager for evaluation progress bars.

**Arguments**:

- `datasets` - List of datasets being evaluated
- `hyperparam_count` - Number of hyperparameter configurations per dataset
- `parallel` - Whether running in parallel mode
- `total_eval_runs` - Total number of EvalRunSpecs


**Yields**:

- `EvaluationProgressBars` - Progress bar manager instance

<a id="scorebook.types"></a>

# scorebook.types

Type definitions for scorebook evaluation framework.

<a id="scorebook.types.AdaptiveEvalDataset"></a>

## AdaptiveEvalDataset Objects

```python
@dataclass
class AdaptiveEvalDataset()
```

Represents a dataset configured for adaptive evaluation.

<a id="scorebook.types.EvalRunSpec"></a>

## EvalRunSpec Objects

```python
@dataclass
class EvalRunSpec()
```

Specification for a single evaluation run with dataset and hyperparameters.

<a id="scorebook.types.EvalRunSpec.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

Return string representation of EvalRunSpec.

<a id="scorebook.types.AdaptiveEvalRunSpec"></a>

## AdaptiveEvalRunSpec Objects

```python
@dataclass
class AdaptiveEvalRunSpec()
```

Specification for an adaptive evaluation run.

<a id="scorebook.types.ClassicEvalRunResult"></a>

## ClassicEvalRunResult Objects

```python
@dataclass
class ClassicEvalRunResult()
```

Results from executing a classic evaluation run.

<a id="scorebook.types.ClassicEvalRunResult.item_scores"></a>

#### item\_scores

```python
@property
def item_scores() -> List[Dict[str, Any]]
```

Return a list of dictionaries containing scores for each evaluated item.

<a id="scorebook.types.ClassicEvalRunResult.aggregate_scores"></a>

#### aggregate\_scores

```python
@property
def aggregate_scores() -> Dict[str, Any]
```

Return the aggregated scores for this run.

<a id="scorebook.types.AdaptiveEvalRunResult"></a>

## AdaptiveEvalRunResult Objects

```python
@dataclass
class AdaptiveEvalRunResult()
```

Results from executing an adaptive evaluation run.

<a id="scorebook.types.AdaptiveEvalRunResult.aggregate_scores"></a>

#### aggregate\_scores

```python
@property
def aggregate_scores() -> Dict[str, Any]
```

Return the aggregated scores for this adaptive run.

<a id="scorebook.types.EvalResult"></a>

## EvalResult Objects

```python
@dataclass
class EvalResult()
```

Container for evaluation results across multiple runs.

<a id="scorebook.types.EvalResult.item_scores"></a>

#### item\_scores

```python
@property
def item_scores() -> List[Dict[str, Any]]
```

Return a list of dictionaries containing scores for each evaluated item.

<a id="scorebook.types.EvalResult.aggregate_scores"></a>

#### aggregate\_scores

```python
@property
def aggregate_scores() -> List[Dict[str, Any]]
```

Return the aggregated scores across all evaluated runs.

<a id="scorebook.cli.auth"></a>

# scorebook.cli.auth

Authentication CLI commands.

<a id="scorebook.cli.auth.auth_command"></a>

#### auth\_command

```python
def auth_command(args: argparse.Namespace) -> int
```

Handle auth subcommands.

<a id="scorebook.cli.auth.login_command"></a>

#### login\_command

```python
def login_command(args: argparse.Namespace) -> int
```

Handle login command.

<a id="scorebook.cli.auth.logout_command"></a>

#### logout\_command

```python
def logout_command(args: argparse.Namespace) -> int
```

Handle logout command.

<a id="scorebook.cli.auth.whoami_command"></a>

#### whoami\_command

```python
def whoami_command(args: argparse.Namespace) -> int
```

Handle whoami command.

<a id="scorebook.cli.main"></a>

# scorebook.cli.main

Main CLI entry point for scorebook.

<a id="scorebook.cli.main.create_parser"></a>

#### create\_parser

```python
def create_parser() -> argparse.ArgumentParser
```

Create the main argument parser.

<a id="scorebook.cli.main.main"></a>

#### main

```python
def main(argv: Optional[List[str]] = None) -> int
```

Run the main CLI entry point.

<a id="scorebook.eval_dataset"></a>

# scorebook.eval\_dataset

Eval Dataset implementation for scorebook.

<a id="scorebook.eval_dataset.EvalDataset"></a>

## EvalDataset Objects

```python
class EvalDataset()
```

Eval Dataset implementation for scorebook.

<a id="scorebook.eval_dataset.EvalDataset.__init__"></a>

#### \_\_init\_\_

```python
def __init__(name: str,
             label: str,
             metrics: Union[str, Type[MetricBase],
                            List[Union[str, Type[MetricBase]]]],
             hf_dataset: HuggingFaceDataset,
             prompt_template: Optional[str] = None)
```

Create a new scorebook evaluation dataset instance.

**Arguments**:

- `name`: The name of the evaluation dataset.
- `label`: The label field of the dataset.
- `metrics`: The specified metrics associated with the dataset.
- `hf_dataset`: The dataset as a hugging face dataset object.
- `prompt_template`: Optional prompt template for building prompts from dataset items.

<a id="scorebook.eval_dataset.EvalDataset.__len__"></a>

#### \_\_len\_\_

```python
def __len__() -> int
```

Return the number of items in the dataset.

<a id="scorebook.eval_dataset.EvalDataset.__getitem__"></a>

#### \_\_getitem\_\_

```python
def __getitem__(key: Union[int, str]) -> Union[Dict[str, Any], List[Any]]
```

Allow item access by index (int) or by column name (str).

- eval_dataset[i] returns the i-th example (dict).
- eval_dataset["feature"] returns a list of values for that feature.

<a id="scorebook.eval_dataset.EvalDataset.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

Return a formatted string summary of the evaluation dataset.

<a id="scorebook.eval_dataset.EvalDataset.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[Dict[str, Any]]
```

Return an iterator over all examples in the dataset.

<a id="scorebook.eval_dataset.EvalDataset.shuffle"></a>

#### shuffle

```python
def shuffle() -> None
```

Randomly shuffle the dataset items.

<a id="scorebook.eval_dataset.EvalDataset.items"></a>

#### items

```python
@property
def items() -> List[Any]
```

Return a list of all examples in the dataset.

<a id="scorebook.eval_dataset.EvalDataset.column_names"></a>

#### column\_names

```python
@property
def column_names() -> List[str]
```

Return a list of column/feature names available in the dataset.

<a id="scorebook.eval_dataset.EvalDataset.from_list"></a>

#### from\_list

```python
@classmethod
def from_list(cls, name: str, label: str,
              metrics: Union[str, Type[MetricBase],
                             List[Union[str, Type[MetricBase]]]],
              data: List[Dict[str, Any]]) -> "EvalDataset"
```

Instantiate an EvalDataset from a list of dictionaries.

**Arguments**:

- `name` - The name of the evaluation dataset.
- `label` - The field used as the evaluation label (ground truth).
- `metrics` - The specified metrics associated with the dataset.
- `data` - List of dictionaries containing the dataset examples.


**Returns**:

  A scorebook EvalDataset wrapping a Hugging Face dataset.

<a id="scorebook.eval_dataset.EvalDataset.from_csv"></a>

#### from\_csv

```python
@classmethod
def from_csv(cls,
             file_path: str,
             label: str,
             metrics: Union[str, Type[MetricBase],
                            List[Union[str, Type[MetricBase]]]],
             name: Optional[str] = None,
             encoding: str = "utf-8",
             newline: str = "",
             **reader_kwargs: Any) -> "EvalDataset"
```

Instantiate a scorebook dataset from a CSV file.

**Arguments**:

- `file_path` - Path to the CSV file.
- `label` - The field used as the evaluation label (ground truth).
- `metrics` - The specified metrics associated with the dataset.
- `name` - Optional name for the eval dataset, if not provided, the path is used
- `encoding` - Encoding of the CSV file.
- `newline` - Newline character of the CSV file.
- `reader_kwargs` - Dict of kwargs passed to `csv.DictReader`.


**Returns**:

  A scorebook EvalDataset.


**Raises**:

- `FileNotFoundError` - If the file does not exist at the given path.
- `ValueError` - If the CSV file cannot be parsed or is empty.

<a id="scorebook.eval_dataset.EvalDataset.from_json"></a>

#### from\_json

```python
@classmethod
def from_json(cls,
              file_path: str,
              label: str,
              metrics: Union[str, Type[MetricBase],
                             List[Union[str, Type[MetricBase]]]],
              name: Optional[str] = None,
              split: Optional[str] = None) -> "EvalDataset"
```

Instantiate an EvalDataset from a JSON file.

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

**Arguments**:

- `file_path` - Path to the JSON file on disk.
- `label` - The field used as the evaluation label (ground truth).
- `metrics` - The specified metrics associated with the dataset.
- `name` - Optional name for the eval dataset, if not provided, the path is used
- `split` - If the JSON uses a split structure, this is the split name to load.


**Returns**:

  A scorebook EvalDataset wrapping a Hugging Face dataset.


**Raises**:

- `FileNotFoundError` - If the file does not exist.
- `ValueError` - If the JSON is invalid or the structure is unsupported.

<a id="scorebook.eval_dataset.EvalDataset.from_huggingface"></a>

#### from\_huggingface

```python
@classmethod
def from_huggingface(cls,
                     path: str,
                     label: str,
                     metrics: Union[str, Type[MetricBase],
                                    List[Union[str, Type[MetricBase]]]],
                     split: Optional[str] = None,
                     name: Optional[str] = None) -> "EvalDataset"
```

Instantiate an EvalDataset from a dataset available on Hugging Face Hub.

If a specific split is provided (e.g., "train" or "test"), it will be loaded directly.
If no split is specified, the method attempts to load the full dataset. If the dataset
is split into multiple subsets (i.e., a DatasetDict), it defaults to loading the "test"
split.

**Arguments**:

- `path` - The path of the dataset on the Hugging Face Hub.
- `label` - The field used as the evaluation label (ground truth).
- `metrics` - The specified metrics associated with the dataset.
- `split` - Optional name of the split to load.
- `name` - Optional dataset configuration name.


**Returns**:

  An EvalDataset wrapping the selected Hugging Face dataset.


**Raises**:

- `ValueError` - If the dataset cannot be loaded, or the expected split is missing.

<a id="scorebook.eval_dataset.EvalDataset.from_yaml"></a>

#### from\_yaml

```python
@classmethod
def from_yaml(cls, file_path: str) -> "EvalDataset"
```

Instantiate an EvalDataset from a YAML file.

The YAML file should contain configuration for loading a dataset, including:
- name: Name of the dataset or Hugging Face dataset path
- label: The field used as the evaluation label
- metrics: List of metrics to evaluate
- split: Optional split name to load
- template: Optional prompt template

**Returns**:

  An EvalDataset instance configured according to the YAML file.


**Raises**:

- `ValueError` - If the YAML file is invalid or missing required fields.

<a id="scorebook.eval_dataset.EvalDataset.sample"></a>

#### sample

```python
def sample(sample_size: int) -> "EvalDataset"
```

Create a new dataset with randomly sampled items from this dataset.

**Arguments**:

- `sample_size` - The number of items to sample from the dataset


**Returns**:

  A new EvalDataset with randomly sampled items


**Raises**:

- `ValueError` - If sample_size is larger than the dataset size

<a id="scorebook.inference.portkey"></a>

# scorebook.inference.portkey

Portkey inference implementation for Scorebook.

This module provides utilities for running inference using Portkey's API,
supporting both single response and batch inference operations. It handles
API communication, request formatting, and response processing.

<a id="scorebook.inference.portkey.responses"></a>

#### responses

```python
async def responses(items: List[Any],
                    model: str,
                    client: Optional[AsyncPortkey] = None,
                    **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests using Portkey's API.

This asynchronous function handles multiple inference requests,
manages the API communication, and processes the responses.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - Model to use via Portkey.
- `client` - Optional Portkey client instance.
- `hyperparameters` - Dictionary of hyperparameters for inference.


**Returns**:

  List of raw model responses.

<a id="scorebook.inference.portkey.batch"></a>

#### batch

```python
async def batch(items: List[Any],
                model: str,
                client: Optional[AsyncPortkey] = None,
                **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests in batch using Portkey's API.

This asynchronous function handles batch processing of inference requests,
optimizing for throughput while respecting API rate limits.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - Model to use via Portkey.
- `client` - Optional Portkey client instance.
- `hyperparameters` - Dictionary of hyperparameters for inference.


**Returns**:

  A list of raw model responses.

<a id="scorebook.inference.vertex"></a>

# scorebook.inference.vertex

Google Cloud Vertex AI batch inference implementation for Scorebook.

This module provides utilities for running batch inference using Google Cloud
Vertex AI Gemini models, supporting large-scale asynchronous processing. It handles
API communication, request formatting, response processing, and Cloud Storage operations.

<a id="scorebook.inference.vertex.responses"></a>

#### responses

```python
async def responses(
        items: List[Union[
            str,
            List[str],
            types.Content,
            List[types.Content],
            types.FunctionCall,
            List[types.FunctionCall],
            types.Part,
            List[types.Part],
        ]],
        model: str,
        client: Optional[genai.Client] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        system_instruction: Optional[str] = None,
        **hyperparameters: Any) -> List[types.GenerateContentResponse]
```

Process multiple inference requests using Google Cloud Vertex AI.

This asynchronous function handles multiple inference requests,
manages the API communication, and processes the responses.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - Gemini model ID to use (e.g., 'gemini-2.0-flash-001').
- `client` - Optional Vertex AI client instance.
- `project_id` - Google Cloud Project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
- `location` - Google Cloud region (default: 'us-central1').
- `system_instruction` - Optional system instruction to guide model behavior.
- `hyperparameters` - Additional parameters for the requests.


**Returns**:

  List of raw model responses.

<a id="scorebook.inference.vertex.batch"></a>

#### batch

```python
async def batch(items: List[Any],
                model: str,
                project_id: Optional[str] = None,
                location: str = "us-central1",
                input_bucket: Optional[str] = None,
                output_bucket: Optional[str] = None,
                **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests in batch using Google Cloud Vertex AI.

This asynchronous function handles batch processing of inference requests,
optimizing for cost and throughput using Google Cloud's batch prediction API.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - Gemini model ID to use (e.g., 'gemini-2.0-flash-001').
- `project_id` - Google Cloud Project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
- `location` - Google Cloud region (default: 'us-central1').
- `input_bucket` - GCS bucket for input data (required).
- `output_bucket` - GCS bucket for output data (required).
- `hyperparameters` - Additional parameters for the batch requests.


**Returns**:

  A list of raw model responses.

<a id="scorebook.inference.openai"></a>

# scorebook.inference.openai

OpenAI inference implementation for Scorebook.

This module provides utilities for running inference using OpenAI's models,
supporting both single response and batch inference operations. It handles
API communication, request formatting, and response processing.

<a id="scorebook.inference.openai.responses"></a>

#### responses

```python
async def responses(items: List[Any],
                    model: str = "gpt-4.1-nano",
                    client: Any = None,
                    **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests using OpenAI's Async API.

This asynchronous function handles multiple inference requests,
manages the API communication, and processes the responses.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - OpenAI model to use.
- `client` - Optional OpenAI client instance.
- `hyperparameters` - Dictionary of hyperparameters for inference.


**Returns**:

  List of raw model responses.


**Raises**:

- `NotImplementedError` - Currently not implemented.

<a id="scorebook.inference.openai.batch"></a>

#### batch

```python
async def batch(items: List[Any],
                model: str = "gpt-4.1-nano",
                client: Any = None,
                **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests in batch using OpenAI's API.

This asynchronous function handles batch processing of inference requests,
optimizing for throughput while respecting API rate limits.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - OpenAI model to use.
- `client` - Optional OpenAI client instance.
- `hyperparameters` - Dictionary of hyperparameters for inference.


**Returns**:

  A list of raw model responses.


**Raises**:

- `NotImplementedError` - Currently not implemented.

<a id="scorebook.inference.bedrock"></a>

# scorebook.inference.bedrock

AWS Bedrock batch inference implementation for Scorebook.

This module provides utilities for running batch inference using AWS Bedrock's
Model Invocation Jobs, supporting large-scale asynchronous processing. It handles
API communication, request formatting, response processing, and S3 operations.

<a id="scorebook.inference.bedrock.batch"></a>

#### batch

```python
async def batch(items: List[Any],
                model: Optional[str] = None,
                aws_region: Optional[str] = None,
                aws_profile: Optional[str] = None,
                bucket: Optional[str] = None,
                input_prefix: Optional[str] = None,
                output_prefix: Optional[str] = None,
                role_arn: Optional[str] = None,
                **hyperparameters: Any) -> List[Any]
```

Process multiple inference requests in batch using AWS Bedrock.

This asynchronous function handles batch processing of inference requests,
optimizing for cost and throughput using AWS Bedrock's Model Invocation Jobs.

**Arguments**:

- `items` - List of preprocessed items to process.
- `model` - Bedrock model ID (e.g., 'us.anthropic.claude-3-5-sonnet-20241022-v2:0').
- `aws_region` - AWS region for Bedrock and S3.
- `aws_profile` - AWS profile name for authentication.
- `bucket` - S3 bucket name for input/output data.
- `input_prefix` - S3 prefix for input data.
- `output_prefix` - S3 prefix for output data.
- `role_arn` - IAM role ARN for Bedrock execution.
- `hyperparameters` - Additional parameters for the batch requests.


**Returns**:

  A list of raw model responses.

<a id="scorebook.inference.bedrock.s3_uri_to_bucket_and_prefix"></a>

#### s3\_uri\_to\_bucket\_and\_prefix

```python
def s3_uri_to_bucket_and_prefix(s3_uri: str) -> Tuple[str, str]
```

Parse S3 URI to bucket and prefix.

<a id="scorebook.exceptions"></a>

# scorebook.exceptions

Custom exceptions for the Scorebook framework.

This module defines specific exception types used throughout the Scorebook
evaluation framework to provide clear error handling and debugging information.

<a id="scorebook.exceptions.ScoreBookError"></a>

## ScoreBookError Objects

```python
class ScoreBookError(Exception)
```

Base exception class for all Scorebook-related errors.

<a id="scorebook.exceptions.EvaluationError"></a>

## EvaluationError Objects

```python
class EvaluationError(ScoreBookError)
```

Raised when there are errors during model evaluation.

<a id="scorebook.exceptions.ParameterValidationError"></a>

## ParameterValidationError Objects

```python
class ParameterValidationError(ScoreBookError)
```

Raised when invalid parameters are provided to evaluation functions.

<a id="scorebook.exceptions.InferenceError"></a>

## InferenceError Objects

```python
class InferenceError(EvaluationError)
```

Raised when there are errors during model inference.

<a id="scorebook.exceptions.MetricComputationError"></a>

## MetricComputationError Objects

```python
class MetricComputationError(EvaluationError)
```

Raised when metric computation fails.

<a id="scorebook.exceptions.MetricComputationError.__init__"></a>

#### \_\_init\_\_

```python
def __init__(metric_name: str, dataset_name: str, original_error: Exception)
```

Initialize metric computation error.

<a id="scorebook.exceptions.DataMismatchError"></a>

## DataMismatchError Objects

```python
class DataMismatchError(EvaluationError)
```

Raised when there's a mismatch between outputs and expected labels.

<a id="scorebook.exceptions.DataMismatchError.__init__"></a>

#### \_\_init\_\_

```python
def __init__(outputs_count: int, labels_count: int, dataset_name: str)
```

Initialize data mismatch error.

<a id="scorebook.exceptions.ParallelExecutionError"></a>

## ParallelExecutionError Objects

```python
class ParallelExecutionError(ScoreBookError)
```

Raised when parallel execution requirements are not met.

<a id="scorebook.evaluate"></a>

# scorebook.evaluate

Model evaluation functionality for the Scorebook framework.

This module provides the core evaluation logic to assess model predictions
against ground truth labels using configurable metrics. It supports:

- Batch evaluation of models across multiple datasets
- Flexible metric computation and aggregation
- Optional parameter sweeping and experiment tracking
- Customizable inference functions

The main entry point is the `evaluate()` function which handles running
models on datasets and computing metric scores.

<a id="scorebook.evaluate.evaluate"></a>

#### evaluate

```python
def evaluate(inference: Callable,
             datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
             hyperparameters: Optional[Union[Dict[str, Any],
                                             List[Dict[str, Any]]]] = None,
             experiment_id: Optional[str] = None,
             project_id: Optional[str] = None,
             metadata: Optional[Dict[str, Any]] = None,
             upload_results: Union[Literal["auto"], bool] = "auto",
             sample_size: Optional[int] = None,
             parallel: bool = False,
             return_dict: bool = True,
             return_aggregates: bool = True,
             return_items: bool = False,
             return_output: bool = False) -> Union[Dict, List]
```

Evaluate a model and collection of hyperparameters over datasets with specified metrics.

**Arguments**:

- `inference` - A callable that runs model inference over a list of evaluation items
- `datasets` - One or more evaluation datasets to run evaluation on.
- `hyperparameters` - Optional list of hyperparameter configurations or grid to evaluate
- `experiment_id` - Optional ID of the experiment to upload results to on Trismik's dashboard.
- `project_id` - Optional ID of the project to upload results to on Trismik's dashboard.
- `metadata` - Optional metadata to attach to the evaluation.
- `upload_results` - If True, uploads results to Trismik's dashboard.
- `sample_size` - Optional number of items to sample from each dataset.
- `parallel` - If True, runs evaluation in parallel. Requires the inference callable to be async.
- `return_dict` - If True, returns eval results as a dict
- `return_aggregates` - If True, returns aggregate scores for each dataset
- `return_items` - If True, returns individual items for each dataset
- `return_output` - If True, returns model outputs for each dataset item evaluated


**Returns**:

  Union[Dict, List, EvalResult]:
  The evaluation results in the format specified by return parameters:
  - If return_dict=False: Returns an EvalResult object containing all run results
  - If return_dict=True Returns the evaluation results as a dict

<a id="scorebook.inference_pipeline"></a>

# scorebook.inference\_pipeline

Inference pipeline implementation for processing items through model inference.

This module provides a pipeline structure for handling model inference tasks,
supporting preprocessing, model inference, and postprocessing steps in a
configurable way.

<a id="scorebook.inference_pipeline.InferencePipeline"></a>

## InferencePipeline Objects

```python
class InferencePipeline()
```

A pipeline for processing items through model inference.

This class implements a three-stage pipeline that handles:
1. Preprocessing of input items
2. Model inference
3. Postprocessing of model outputs


**Attributes**:

- `model` - Name or identifier of the model being used
- `preprocessor` - Function to prepare items for model inference
- `inference_function` - Function that performs the actual model inference
- `postprocessor` - Function to process the model outputs

<a id="scorebook.inference_pipeline.InferencePipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model: str,
             inference_function: Callable,
             preprocessor: Optional[Callable] = None,
             postprocessor: Optional[Callable] = None) -> None
```

Initialize the inference pipeline.

**Arguments**:

- `model` - Name or identifier of the model to use
- `inference_function` - Function that performs model inference
- `preprocessor` - Optional function to prepare items for inference.
- `postprocessor` - Optional function to process model outputs.

<a id="scorebook.inference_pipeline.InferencePipeline.run"></a>

#### run

```python
async def run(items: List[Dict[str, Any]],
              **hyperparameters: Any) -> List[Any]
```

Execute the complete inference pipeline on a list of items.

**Arguments**:

- `items` - List of items to process through the pipeline
- `**hyperparameters` - Model-specific parameters for inference


**Returns**:

  List of processed outputs after running through the complete pipeline

<a id="scorebook.inference_pipeline.InferencePipeline.__call__"></a>

#### \_\_call\_\_

```python
async def __call__(items: List[Dict[str, Any]],
                   **hyperparameters: Any) -> List[Any]
```

Make the pipeline instance callable by wrapping the run method.

**Arguments**:

- `items` - List of items to process through the pipeline
- `**hyperparameters` - Model-specific parameters for inference


**Returns**:

  List of processed outputs after running through the complete pipeline
