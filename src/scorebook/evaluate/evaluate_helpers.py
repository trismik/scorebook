"""Helper utilities shared by synchronous and asynchronous evaluation flows."""

import asyncio
import dataclasses
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from trismik._async.client import TrismikAsyncClient
from trismik._sync.client import TrismikClient
from trismik.types import TrismikMultipleChoiceTextItem

from scorebook import EvalDataset
from scorebook.exceptions import (
    DataMismatchError,
    MetricComputationError,
    ParameterValidationError,
    ScoreBookError,
)
from scorebook.settings import TRISMIK_SERVICE_URL
from scorebook.trismik.credentials import get_token
from scorebook.types import AdaptiveEvalDataset, AdaptiveEvalRunSpec, EvalResult, EvalRunSpec
from scorebook.utils import expand_dict, is_awaitable

logger = logging.getLogger(__name__)


# TODO: Remove this when backend supports boolean item metrics
NORMALIZE_METRICS_FOR_UPLOAD = True


def normalize_metric_value(value: Any) -> Any:
    """Normalize metric values for API upload compatibility.

    TEMPORARY WORKAROUND: The Trismik API currently rejects boolean metric values.
    This function converts boolean values to floats (True -> 1.0, False -> 0.0)
    to ensure upload compatibility.

    Args:
        value: The metric value to normalize

    Returns:
        Float if value is bool, otherwise unchanged

    TODO: Remove this function when backend supports boolean metrics natively.
          To revert: Set NORMALIZE_METRICS_FOR_UPLOAD = False
    """
    if not NORMALIZE_METRICS_FOR_UPLOAD:
        return value

    # Convert booleans to floats for API compatibility
    if isinstance(value, bool):
        return float(value)  # True -> 1.0, False -> 0.0

    return value


def validate_parameters(params: Dict[str, Any], caller: Callable[..., Any]) -> None:
    """Validate all parameters for evaluation."""

    caller_is_async = is_awaitable(caller)

    # Sync evaluate() should only accept sync inference functions
    if not caller_is_async and is_awaitable(params.get("inference")):
        raise ParameterValidationError(
            "evaluate() only accepts synchronous inference functions. "
            "Use evaluate_async() for async inference functions."
        )

    # Async evaluate_async() should only accept async inference functions
    if caller_is_async and not is_awaitable(params.get("inference")):
        raise ParameterValidationError(
            "evaluate_async() only accepts asynchronous inference functions. "
            "Use evaluate() for sync inference functions."
        )

    # If returning a dict, it must contain items and/or aggregates
    if params["return_dict"] and not params["return_aggregates"] and not params["return_items"]:
        raise ParameterValidationError(
            "When return_dict=True, at least one of return_aggregates or return_items must be True"
        )

    # If uploading results, experiment_id and project_id must be specified
    if params["upload_results"]:
        if params["experiment_id"] is None or params["project_id"] is None:
            raise ParameterValidationError(
                "experiment_id and project_id are required for upload_results=True"
            )

    logger.debug("Parameter validation successful")


def prepare_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    split: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> List[Union[EvalDataset, AdaptiveEvalDataset]]:
    """Prepare and separate input datasets into classic and adaptive evaluation datasets."""

    # Ensure datasets is always a list for consistent processing
    if not isinstance(datasets, list):
        datasets = [datasets]

    datasets_out: List[Union[EvalDataset, AdaptiveEvalDataset]] = []
    for dataset in datasets:

        # Prepare classic datasets
        if isinstance(dataset, EvalDataset):
            # Warn if dataset split differs from provided split parameter
            if split is not None and dataset.split is not None and dataset.split != split:
                logger.warning(
                    f"Dataset '{dataset.name}' has split '{dataset.split}' but evaluate split "
                    f"parameter is '{split}'. The dataset split will be used."
                )

            if sample_size is not None:
                dataset = dataset.sample(sample_size)

            datasets_out.append(dataset)

        # Prepare adaptive datasets
        elif isinstance(dataset, str) and ":adaptive" in dataset:
            # Parse adaptive dataset
            parts = dataset.split(":")
            if len(parts) != 2 or parts[1] != "adaptive":
                raise ParameterValidationError(
                    f"Invalid adaptive dataset format: '{dataset}'. "
                    f"Use 'test_id:adaptive' format and specify split via the split parameter."
                )

            # Use the split parameter for all adaptive datasets
            datasets_out.append(AdaptiveEvalDataset(name=dataset, split=split))

        # TODO: dataset name string registry
        elif isinstance(dataset, str):
            pass

        else:
            raise ParameterValidationError(f"Unrecognized dataset type: {type(dataset)}")

    return datasets_out


def prepare_hyperparameter_configs(
    hyperparameters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """Prepare hyperparameters for evaluation by returning a list of hyper-param configs."""
    if hyperparameters is None:
        return [{}]
    if not isinstance(hyperparameters, list):
        expanded: List[Dict[str, Any]] = expand_dict(hyperparameters or {})
        return expanded

    logger.info("Evaluating with hyperparameters: %s", hyperparameters)

    return hyperparameters


def build_eval_run_specs(
    datasets: List[Union[EvalDataset, str]],
    hyperparameters: Any,
    experiment_id: Optional[str],
    project_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Union[EvalRunSpec, AdaptiveEvalRunSpec]]:
    """Build All RunSpec objects for each dataset/hyperparameter combination."""

    eval_run_specs: List[Union[EvalRunSpec, AdaptiveEvalRunSpec]] = []
    for dataset_index, dataset in enumerate(datasets):
        for hyperparameters_index, hyperparameter_config in enumerate(hyperparameters):

            # Create classic eval run spec
            if isinstance(dataset, EvalDataset):
                eval_run_specs.append(
                    build_classic_eval_run_spec(
                        dataset, dataset_index, hyperparameter_config, hyperparameters_index
                    )
                )

            # Create adaptive eval run spec from string
            elif isinstance(dataset, AdaptiveEvalDataset):
                if not experiment_id or not project_id:
                    raise ScoreBookError(
                        "experiment_id and project_id are required for adaptive evaluations"
                    )
                eval_run_specs.append(
                    build_adaptive_eval_run_spec(
                        dataset.name,
                        dataset_index,
                        hyperparameter_config,
                        hyperparameters_index,
                        experiment_id,
                        project_id,
                        dataset.split,
                        metadata,
                    )
                )

            # Log warning - should never happen
            else:
                logger.warning("Unrecognized dataset type: %s", dataset)

    return eval_run_specs


def build_classic_eval_run_spec(
    dataset: EvalDataset,
    dataset_index: int,
    hyperparameters: Dict[str, Any],
    hyperparameters_index: int,
) -> EvalRunSpec:
    """Build EvalRunSpec objects for a classic dataset and hyperparameter combination.

    Extracts input and label values from the appropriate columns in the dataset.
    The column names are determined by dataset.input and dataset.label,
    which may be original field names (e.g., "question", "answer") or computed
    column names (e.g., "*input", "*label") if templates were used.
    """
    # Extract inputs and labels using the dataset's column specifications
    inputs = dataset[dataset.input]  # Returns List[Any]
    labels = dataset[dataset.label]  # Returns List[Any]
    eval_run_spec = EvalRunSpec(
        dataset,
        dataset_index,
        hyperparameters,
        hyperparameters_index,
        inputs,
        labels,
    )
    logger.debug("Built EvalRunSpec: %s", eval_run_spec)
    return eval_run_spec


def build_adaptive_eval_run_spec(
    adaptive_dataset: str,
    dataset_index: int,
    hyperparameter_config: Dict[str, Any],
    hyperparameter_config_index: int,
    experiment_id: str,
    project_id: str,
    split: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AdaptiveEvalRunSpec:
    """Build AdaptiveEvalRunSpec objects for a dataset/hyperparameter combination."""
    # Keep the full dataset name including ":adaptive" suffix for backend API
    adaptive_eval_run_spec = AdaptiveEvalRunSpec(
        adaptive_dataset,
        dataset_index,
        hyperparameter_config,
        hyperparameter_config_index,
        experiment_id,
        project_id,
        split,
        metadata,
    )
    logger.debug("Built AdaptiveEvalRunSpec: %s", adaptive_eval_run_spec)
    return adaptive_eval_run_spec


def score_metrics(
    eval_dataset: EvalDataset, outputs: List[Any], labels: List[Any]
) -> Dict[str, Dict[str, Any]]:
    """Compute metric scores for a given dataset and inference outputs."""
    metric_scores: Dict[str, Dict[str, Any]] = {}

    if len(outputs) != len(labels):
        raise DataMismatchError(len(outputs), len(labels), eval_dataset.name)

    for metric in eval_dataset.metrics:
        try:
            aggregate_scores, item_scores = metric.score(outputs, labels)
            metric_scores[metric.name] = {
                "aggregate_scores": aggregate_scores,
                "item_scores": item_scores,
            }
        except Exception as e:
            logger.error(
                "Failed to compute metric '%s' for dataset '%s': %s",
                metric.name,
                eval_dataset.name,
                str(e),
            )
            raise MetricComputationError(metric.name, eval_dataset.name, e)

    return metric_scores


def create_trismik_async_client() -> TrismikAsyncClient:
    """Create a new async Trismik client instance."""
    api_key = get_token()
    logger.debug("Creating new async Trismik client")
    return TrismikAsyncClient(service_url=TRISMIK_SERVICE_URL, api_key=api_key)


def create_trismik_sync_client() -> TrismikClient:
    """Create a new sync Trismik client instance."""
    api_key = get_token()
    logger.debug("Creating new sync Trismik client")
    return TrismikClient(service_url=TRISMIK_SERVICE_URL, api_key=api_key)


def get_model_name(
    inference_callable: Optional[Callable] = None, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Determine a model's name with the fallback "Model"."""

    # First priority: metadata.model
    if metadata and "model" in metadata:
        return str(metadata["model"])

    # Second priority: inference_pipeline.model (if callable is an InferencePipeline)
    if inference_callable and hasattr(inference_callable, "model"):
        return str(inference_callable.model)

    # Fallback: "Model"
    return "Model"


def format_results(
    eval_result: EvalResult,
    return_dict: bool,
    return_aggregates: bool,
    return_items: bool,
    return_output: bool,
) -> Union[EvalResult, Dict, List]:
    """Format an `EvalResult` into the requested output structure."""

    # Return results as a dict
    if return_dict:
        results = {}

        if return_aggregates:
            results["aggregate_results"] = eval_result.aggregate_scores

        if return_items:
            item_scores = eval_result.item_scores

            # Remove inference output if not requested
            if not return_output:
                for item in item_scores:
                    item.pop("output", None)

            results["item_results"] = item_scores

        # If both are requested, return the combined structure
        if return_aggregates and return_items:
            return results
        # If only aggregates requested, return just the list
        elif return_aggregates:
            return results["aggregate_results"]
        # If only items requested, return just the list
        else:
            return results["item_results"]

    # Return results as an EvalResult object
    else:
        return eval_result


def make_trismik_inference(
    inference_function: Callable[..., Any],
    return_list: bool = False,
) -> Callable[[Any], Any]:
    """Wrap an inference function for flexible input handling.

    Takes a function expecting list[dict] and makes it accept single dict
    or TrismikMultipleChoiceTextItem.
    """

    # Check if the inference function is async
    is_async = is_awaitable(inference_function)

    def sync_trismik_inference_function(eval_items: Any, **kwargs: Any) -> Any:
        # Single TrismikMultipleChoiceTextItem dataclass
        if isinstance(eval_items, TrismikMultipleChoiceTextItem):
            eval_item_dict = dataclasses.asdict(eval_items)
            results = inference_function([eval_item_dict], **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results if return_list else results[0]

        # Single item (a mapping)
        if isinstance(eval_items, Mapping):
            results = inference_function([eval_items], **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results if return_list else results[0]

        # Iterable of items (but not a string/bytes)
        if isinstance(eval_items, Iterable) and not isinstance(eval_items, (str, bytes)):
            # Convert any TrismikMultipleChoiceTextItem instances to dicts
            converted_items = []
            for item in eval_items:
                if isinstance(item, TrismikMultipleChoiceTextItem):
                    converted_items.append(dataclasses.asdict(item))
                else:
                    converted_items.append(item)
            results = inference_function(converted_items, **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results

        raise TypeError(
            "Expected a single item (Mapping[str, Any] or TrismikMultipleChoiceTextItem) "
            "or an iterable of such items."
        )

    return sync_trismik_inference_function


def resolve_adaptive_split(
    test_id: str,
    user_specified_split: Optional[str],
    available_splits: List[str],
) -> str:
    """Resolve the dataset split to use for adaptive evaluation.

    Resolution order:
    1. If user specified a split, validate it exists and use it
    2. If not specified and exactly one split is available, use it
    3. If not specified and multiple splits are available, raise an error
    4. If no splits are available, raise an error

    Args:
        test_id: The test dataset ID (without ":adaptive" suffix)
        user_specified_split: Optional split name specified by the user
        available_splits: List of available split names for this dataset

    Returns:
        The resolved split name to use

    Raises:
        ScoreBookError: If the specified split doesn't exist, multiple splits exist without
            user specification, or no splits are available
    """
    logger.debug(f"Available splits for {test_id}: {available_splits}")

    # If user specified a split, validate and use it
    if user_specified_split is not None:
        if user_specified_split in available_splits:
            logger.info(f"Using user-specified split '{user_specified_split}' for {test_id}")
            return user_specified_split
        else:
            raise ScoreBookError(
                f"Specified split '{user_specified_split}' not found for dataset '{test_id}'. "
                f"Available splits: {available_splits}"
            )

    # No split specified - check available splits
    if len(available_splits) == 0:
        raise ScoreBookError(f"No splits available for dataset '{test_id}'. ")
    elif len(available_splits) == 1:
        # Exactly one split - auto-select it
        selected_split = available_splits[0]
        logger.info(f"Auto-selecting only available split '{selected_split}' for {test_id}")
        return selected_split
    else:
        # Multiple splits available - user must specify
        raise ScoreBookError(
            f"Multiple splits available for dataset '{test_id}': {available_splits}. "
            f"Please specify which split to use via evaluate's 'split' parameter."
        )
