"""Helper utilities shared by synchronous and asynchronous evaluation flows."""

import asyncio
import atexit
import dataclasses
import inspect
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Union

from trismik import TrismikAsyncClient, TrismikClient
from trismik.types import TrismikMultipleChoiceTextItem

from scorebook import EvalDataset
from scorebook.exceptions import (
    DataMismatchError,
    MetricComputationError,
    ParameterValidationError,
    ScoreBookError,
)
from scorebook.trismik_services.login import get_token
from scorebook.types import AdaptiveEvalDataset, AdaptiveEvalRunSpec, EvalResult, EvalRunSpec
from scorebook.utils import expand_dict, is_awaitable

logger = logging.getLogger(__name__)

# Global singleton client instances with state tracking
_sync_client: Optional[TrismikClient] = None
_async_client: Optional[TrismikAsyncClient] = None
_current_api_key: Optional[str] = None
_current_service_url: Optional[str] = None
_cleanup_registered: bool = False


def _register_cleanup() -> None:
    """Register cleanup handlers for proper client shutdown."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_clients)
        _cleanup_registered = True


def _cleanup_clients() -> None:
    """Clean up all client instances."""
    global _sync_client, _async_client

    # Clean up sync client
    if _sync_client and hasattr(_sync_client, "close"):
        try:
            _sync_client.close()
        except Exception as e:
            logger.debug(f"Error closing sync client: {e}")

    # Clean up async client (requires event loop)
    if _async_client and hasattr(_async_client, "aclose"):
        try:
            # Try to close async client if event loop is still running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task to close the client
                asyncio.create_task(_async_client.aclose())
            else:
                # If no loop is running, create one temporarily
                asyncio.run(_async_client.aclose())
        except Exception as e:
            logger.debug(f"Error closing async client: {e}")

    _sync_client = None
    _async_client = None


async def cleanup_clients_async() -> None:
    """Perform async cleanup for proper client closure."""
    global _async_client
    if _async_client and hasattr(_async_client, "aclose"):
        try:
            await _async_client.aclose()
            logger.debug("Successfully closed async Trismik client")
        except Exception as e:
            logger.debug(f"Error closing async client: {e}")
        finally:
            _async_client = None


def resolve_upload_results(upload_results: Union[Literal["auto"], bool]) -> bool:
    """Resolve the upload_results parameter based on trismik login status."""

    if upload_results == "auto":
        upload_results = get_token() is not None
        logger.debug("Auto upload results resolved to: %s", upload_results)

    return upload_results


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

            if sample_size is not None:
                dataset = dataset.sample(sample_size)

            datasets_out.append(dataset)

        # Prepare adaptive datasets
        elif isinstance(dataset, str) and dataset.endswith(":adaptive"):
            datasets_out.append(AdaptiveEvalDataset(dataset.replace(":adaptive", "")))

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
    """Build EvalRunSpec objects for a classic dataset and hyperparameter combination."""
    items = dataset.items
    labels = [item.get(dataset.label) for item in items]
    eval_run_spec = EvalRunSpec(
        dataset,
        dataset_index,
        hyperparameters,
        hyperparameters_index,
        items,
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
    metadata: Optional[Dict[str, Any]] = None,
) -> AdaptiveEvalRunSpec:
    """Build AdaptiveEvalRunSpec objects for a dataset/hyperparameter combination."""
    dataset = adaptive_dataset.replace(":adaptive", "")
    adaptive_eval_run_spec = AdaptiveEvalRunSpec(
        dataset,
        dataset_index,
        hyperparameter_config,
        hyperparameter_config_index,
        experiment_id,
        project_id,
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


def get_trismik_client(caller: Callable[..., Any]) -> Union[TrismikClient, TrismikAsyncClient]:
    """Return a singleton Trismik client with proper resource management."""
    global _sync_client, _async_client, _current_api_key, _current_service_url

    # Register cleanup handlers on first use
    _register_cleanup()

    # Get current environment values
    caller_is_async = is_awaitable(caller)
    service_url = os.environ.get("TRISMIK_SERVICE_URL", "https://api.trismik.com/adaptive-testing")
    api_key = get_token()

    # Check if we need to recreate clients due to config changes
    config_changed = api_key != _current_api_key or service_url != _current_service_url

    if config_changed:
        logger.debug("Trismik client config changed, recreating clients")
        # Clean up existing clients
        _cleanup_clients()
        # Update tracking variables
        _current_api_key = api_key
        _current_service_url = service_url

    # Return appropriate client type
    if caller_is_async:
        if _async_client is None:
            logger.debug("Creating new async Trismik client")
            _async_client = TrismikAsyncClient(service_url=service_url, api_key=api_key)
        return _async_client
    else:
        if _sync_client is None:
            logger.debug("Creating new sync Trismik client")
            _sync_client = TrismikClient(service_url=service_url, api_key=api_key)
        return _sync_client


def get_model_name(
    inference_callable: Optional[Callable] = None, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Determine a model's name with the fallback "unspecified"."""

    # First priority: metadata.model
    if metadata and "model" in metadata:
        return str(metadata["model"])

    # Second priority: inference_pipeline.model (if callable is an InferencePipeline)
    if inference_callable and hasattr(inference_callable, "model"):
        return str(inference_callable.model)

    # Fallback: "unspecified"
    return "unspecified"


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
                    item.pop("inference_output", None)

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
    is_async = inspect.iscoroutinefunction(inference_function) or (
        hasattr(inference_function, "__call__")
        and inspect.iscoroutinefunction(inference_function.__call__)
    )

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
