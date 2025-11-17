"""Upload evaluation and scoring results to Trismik's experimentation platform."""

import logging
from typing import Any, Dict, List, Optional

from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
)

from scorebook.evaluate.evaluate_helpers import (
    create_trismik_async_client,
    create_trismik_sync_client,
    get_model_name,
    normalize_metric_value,
)

logger = logging.getLogger(__name__)

# Known fields that are not metrics or hyperparameters
KNOWN_AGGREGATE_FIELDS = {"dataset", "run_id", "run_completed"}
KNOWN_ITEM_FIELDS = {"id", "dataset", "input", "output", "label", "run_id"}


def upload_result(
    run_result: Dict[str, List[Dict[str, Any]]],
    experiment_id: str,
    project_id: str,
    dataset_name: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> str:
    """Upload evaluation or scoring results to Trismik's platform (synchronous).

    This function uploads results in the format returned by the evaluate or score
    functions to the Trismik platform for tracking and analysis.

    Args:
        run_result: Dict with keys 'aggregate_results' and 'item_results' containing
            evaluation/scoring results. Structure matches the output of evaluate()/score().
        experiment_id: Trismik experiment identifier
        project_id: Trismik project identifier
        dataset_name: Optional dataset name. If not provided, extracted from metadata
            or defaults to "Dataset"
        hyperparameters: Optional dict of hyperparameters. If not provided, extracted
            from run_result.
        metadata: Optional metadata dict (can include 'model' and 'dataset' keys)
        model_name: Optional model name. If not provided, extracted from metadata
            or defaults to "Model"

    Returns:
        str: Run ID assigned by Trismik

    Raises:
        Exception: If upload fails (re-raises underlying exceptions)
    """
    # Create Trismik client
    trismik_client = create_trismik_sync_client()

    # Get model name - use provided model_name, or extract from metadata, or use default
    if model_name is not None:
        model = model_name
    else:
        model = get_model_name(metadata=metadata)

    # Get dataset name - use provided dataset_name, or extract from metadata, or use default
    if dataset_name is None:
        if metadata and "dataset" in metadata:
            dataset_name = str(metadata["dataset"])
        else:
            dataset_name = "Dataset"

    # Extract aggregate and item results
    aggregate_results = run_result.get("aggregate_results", [])
    item_results = run_result.get("item_results", [])

    # Use provided hyperparameters or default to empty dict
    # Note: We don't extract hyperparameters from aggregate_results to avoid
    # misclassifying metrics as hyperparameters
    if hyperparameters is None:
        hyperparameters = {}

    # Create eval items from item_results
    trismik_items: List[TrismikClassicEvalItem] = []
    for item in item_results:
        # Extract inputs, outputs, labels
        item_id = str(item.get("id", 0))
        model_input = str(item.get("input", ""))
        model_output = str(item.get("output", ""))
        gold_output = str(item.get("label", ""))

        # Extract item-level metrics (exclude known fields and hyperparameters)
        item_metrics: Dict[str, Any] = {}
        for key, value in item.items():
            if key not in KNOWN_ITEM_FIELDS and key not in (hyperparameters or {}):
                # Normalize metric value for API compatibility
                item_metrics[key] = normalize_metric_value(value)

        eval_item = TrismikClassicEvalItem(  # pragma: allowlist secret
            datasetItemId=item_id,
            modelInput=model_input,
            modelOutput=model_output,
            goldOutput=gold_output,
            metrics=item_metrics,
        )
        trismik_items.append(eval_item)

    # Extract aggregate metrics from aggregate_results
    trismik_metrics: List[TrismikClassicEvalMetric] = []
    if aggregate_results:
        for key, value in aggregate_results[0].items():
            if key not in KNOWN_AGGREGATE_FIELDS and key not in (hyperparameters or {}):
                # This is a metric  # pragma: allowlist secret
                metric = TrismikClassicEvalMetric(metricId=key, value=normalize_metric_value(value))
                trismik_metrics.append(metric)  # pragma: allowlist secret

    # Create classic eval request
    classic_eval_request = TrismikClassicEvalRequest(
        project_id,
        experiment_id,
        dataset_name,
        model,
        hyperparameters,
        trismik_items,
        trismik_metrics,
    )

    # Submit to Trismik  # pragma: allowlist secret
    response: TrismikClassicEvalResponse = trismik_client.submit_classic_eval(
        classic_eval_request
    )  # pragma: allowlist secret

    run_id: str = response.id
    logger.info(f"Run result uploaded successfully to Trismik with run_id: {run_id}")

    return run_id


async def upload_result_async(
    run_result: Dict[str, List[Dict[str, Any]]],
    experiment_id: str,
    project_id: str,
    dataset_name: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> str:
    """Upload evaluation or scoring results to Trismik's platform (asynchronous).

    This function uploads results in the format returned by the evaluate or
    score functions to the Trismik platform for tracking and analysis.

    Args:
        run_result: Dict with keys 'aggregate_results' and 'item_results' containing
            evaluation/scoring results. Structure matches the output of evaluate()/score().
        experiment_id: Trismik experiment identifier
        project_id: Trismik project identifier
        dataset_name: Optional dataset name. If not provided, extracted from metadata
        or defaults to "Dataset"
        hyperparameters: Optional dict of hyperparameters. If not provided, extracted
            from run_result.
        metadata: Optional metadata dict (can include 'model' and 'dataset' keys)
        model_name: Optional model name. If not provided, extracted from metadata
            or defaults to "Model"

    Returns:
        str: Run ID assigned by Trismik

    Raises:
        Exception: If upload fails (re-raises underlying exceptions)
    """
    # Create Trismik async client
    trismik_client = create_trismik_async_client()

    # Get model name - use provided model_name, or extract from metadata, or use default
    if model_name is not None:
        model = model_name
    else:
        model = get_model_name(metadata=metadata)

    # Get dataset name - use provided dataset_name, or extract from metadata, or use default
    if dataset_name is None:
        if metadata and "dataset" in metadata:
            dataset_name = str(metadata["dataset"])
        else:
            dataset_name = "Dataset"

    # Extract aggregate and item results
    aggregate_results = run_result.get("aggregate_results", [])
    item_results = run_result.get("item_results", [])

    # Use provided hyperparameters or default to empty dict
    # Note: We don't extract hyperparameters from aggregate_results to avoid
    # misclassifying metrics as hyperparameters
    if hyperparameters is None:
        hyperparameters = {}

    # Create eval items from item_results
    trismik_items: List[TrismikClassicEvalItem] = []
    for item in item_results:
        # Extract inputs, outputs, labels
        item_id = str(item.get("id", 0))
        model_input = str(item.get("input", ""))
        model_output = str(item.get("output", ""))
        gold_output = str(item.get("label", ""))

        # Extract item-level metrics (exclude known fields and hyperparameters)
        item_metrics: Dict[str, Any] = {}
        for key, value in item.items():
            if key not in KNOWN_ITEM_FIELDS and key not in (hyperparameters or {}):
                # Normalize metric value for API compatibility
                item_metrics[key] = normalize_metric_value(value)

        eval_item = TrismikClassicEvalItem(  # pragma: allowlist secret
            datasetItemId=item_id,
            modelInput=model_input,
            modelOutput=model_output,
            goldOutput=gold_output,
            metrics=item_metrics,
        )
        trismik_items.append(eval_item)

    # Extract aggregate metrics from aggregate_results
    trismik_metrics: List[TrismikClassicEvalMetric] = []
    if aggregate_results:
        for key, value in aggregate_results[0].items():
            if key not in KNOWN_AGGREGATE_FIELDS and key not in (hyperparameters or {}):
                # This is a metric  # pragma: allowlist secret
                metric = TrismikClassicEvalMetric(metricId=key, value=normalize_metric_value(value))
                trismik_metrics.append(metric)  # pragma: allowlist secret

    # Create classic eval request
    classic_eval_request = TrismikClassicEvalRequest(
        project_id,
        experiment_id,
        dataset_name,
        model,
        hyperparameters,
        trismik_items,
        trismik_metrics,
    )

    # Submit to Trismik (async)  # pragma: allowlist secret
    response: TrismikClassicEvalResponse = await trismik_client.submit_classic_eval(
        classic_eval_request
    )  # pragma: allowlist secret

    run_id: str = response.id
    logger.info(f"Run result uploaded successfully to Trismik with run_id: {run_id}")

    return run_id
