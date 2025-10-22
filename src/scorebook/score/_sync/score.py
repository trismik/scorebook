import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

from scorebook.evaluate.evaluate_helpers import resolve_upload_results
from scorebook.exceptions import MetricComputationError, ParameterValidationError
from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry

logger = logging.getLogger(__name__)


def score(
    items: List[Dict[str, Any]],
    metrics: Union[
        str, MetricBase, Type[MetricBase], List[Union[str, MetricBase, Type[MetricBase]]]
    ],
    hyperparameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    upload_results: Union[Literal["auto"], bool] = "auto",
    show_progress: Optional[bool] = None,
    dataset: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Score pre-computed model outputs against labels using specified metrics.

    Args:
        items: List of dicts with required keys 'output' and 'label', optional key 'input'
               Example: [{"output": "4", "label": "4"}, ...]
               Or: [{"input": "What is 2+2?", "output": "4", "label": "4"}, ...]
        metrics: Metric(s) to compute (string name, MetricBase class,
            or list)
        hyperparameters: Optional dict of hyperparameters used to generate
            outputs (for tracking/upload)
        metadata: Optional metadata to attach to the scoring run
        experiment_id: Optional experiment identifier
        project_id: Optional project identifier
        upload_results: If True, uploads results to Trismik's dashboard
        show_progress: If None, uses SHOW_PROGRESS_BARS from settings
        dataset: Optional dataset name (for tracking/upload)

    Returns:
        Dict with keys:
            - "aggregate_results": List with one dict containing aggregate scores and metadata
            - "item_results": List of dicts with per-item scores, outputs, labels, and metadata
    """
    # Resolve and validate parameters
    upload_results = cast(bool, resolve_upload_results(upload_results))

    # Validate items parameter
    _validate_items(items)

    # Validate upload requirements
    if upload_results and (experiment_id is None or project_id is None):
        raise ParameterValidationError(
            "experiment_id and project_id are required for upload_results=True"
        )

    # Validate hyperparameters is a dict (not list)
    if hyperparameters is not None and not isinstance(hyperparameters, dict):
        raise ParameterValidationError(
            "hyperparameters must be a dict, not a list. "
            "For score(), hyperparameters are metadata about the run, not configurations to sweep."
        )

    # Extract outputs and labels from items
    outputs = [item["output"] for item in items]
    labels = [item["label"] for item in items]
    inputs = [item.get("input") for item in items]

    # Resolve metrics to list of MetricBase instances
    metric_instances = _resolve_metrics(metrics)

    # Compute scores for each metric
    metric_scores = _compute_metric_scores(outputs, labels, metric_instances)

    # Build results
    dataset_name = dataset or "scored_items"
    results = _build_results(
        metric_scores=metric_scores,
        items=items,
        inputs=inputs,
        outputs=outputs,
        labels=labels,
        hyperparameters=hyperparameters or {},
        dataset_name=dataset_name,
    )

    # Upload if requested
    if upload_results and experiment_id and project_id:
        _upload_results(
            metric_scores=metric_scores,
            items=items,
            inputs=inputs,
            outputs=outputs,
            labels=labels,
            hyperparameters=hyperparameters or {},
            dataset_name=dataset_name,
            experiment_id=experiment_id,
            project_id=project_id,
            metadata=metadata,
        )

    logger.info("Scoring complete")
    return results


def _validate_items(items: List[Dict[str, Any]]) -> None:
    """Validate the items parameter."""
    if not isinstance(items, list):
        raise ParameterValidationError("items must be a list")

    if len(items) == 0:
        raise ParameterValidationError("items list cannot be empty")

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ParameterValidationError(
                f"Item at index {idx} is not a dict. All items must be dicts."
            )

        if "output" not in item:
            raise ParameterValidationError(f"Item at index {idx} is missing required 'output' key")

        if "label" not in item:
            raise ParameterValidationError(f"Item at index {idx} is missing required 'label' key")


def _resolve_metrics(
    metrics: Union[
        str, MetricBase, Type[MetricBase], List[Union[str, MetricBase, Type[MetricBase]]]
    ]
) -> List[MetricBase]:
    """Resolve metrics parameter to list of MetricBase instances."""
    # Ensure metrics is a list
    if not isinstance(metrics, list):
        metrics = [metrics]

    # Resolve each metric
    metric_instances = []
    for metric in metrics:
        if isinstance(metric, str) or (isinstance(metric, type) and issubclass(metric, MetricBase)):
            # Use MetricRegistry to resolve string names or classes
            metric_instance = MetricRegistry.get(metric)
            metric_instances.append(metric_instance)
        elif isinstance(metric, MetricBase):
            # Already an instance
            metric_instances.append(metric)
        else:
            raise ParameterValidationError(
                f"Invalid metric type: {type(metric)}. "
                "Metrics must be string names, MetricBase classes, or MetricBase instances"
            )

    return metric_instances


def _compute_metric_scores(
    outputs: List[Any],
    labels: List[Any],
    metrics: List[MetricBase],
) -> Dict[str, Dict[str, Any]]:
    """Compute scores for all metrics."""
    metric_scores: Dict[str, Dict[str, Any]] = {}

    for metric in metrics:
        try:
            # Check if the metric's score method is async
            if inspect.iscoroutinefunction(metric.score):
                # Async metric not supported in sync version
                raise MetricComputationError(
                    metric.name,
                    "scored_items",
                    Exception(
                        f"Metric '{metric.name}' has an async score() method. "
                        "Use score_async() instead of score() for async metrics."
                    ),
                )
            else:
                # Sync metric - call normally
                aggregate_scores, item_scores = metric.score(outputs, labels)

            metric_scores[metric.name] = {
                "aggregate_scores": aggregate_scores,
                "item_scores": item_scores,
            }
        except Exception as e:
            logger.error(
                "Failed to compute metric '%s': %s",
                metric.name,
                str(e),
            )
            raise MetricComputationError(metric.name, "scored_items", e) from e

    return metric_scores


def _build_results(
    metric_scores: Dict[str, Dict[str, Any]],
    items: List[Dict[str, Any]],
    inputs: List[Any],
    outputs: List[Any],
    labels: List[Any],
    hyperparameters: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build results dict with both aggregates and items."""
    # Build aggregate results
    aggregate_result = {
        "dataset": dataset_name,
        **hyperparameters,
    }

    # Add aggregate scores from metrics
    for metric_name, metric_data in metric_scores.items():
        if "aggregate_scores" in metric_data:
            for key, value in metric_data["aggregate_scores"].items():
                score_key = key if key == metric_name else f"{metric_name}_{key}"
                aggregate_result[score_key] = value

    # Build item results
    item_results = []
    for idx in range(len(outputs)):
        item_result: Dict[str, Any] = {
            "id": idx,
            "dataset_name": dataset_name,
            "output": outputs[idx],
            "label": labels[idx],
            **hyperparameters,
        }

        # Add input if present
        if inputs[idx] is not None:
            item_result["input"] = inputs[idx]

        # Add item-level metric scores
        for metric_name, metric_data in metric_scores.items():
            if "item_scores" in metric_data and idx < len(metric_data["item_scores"]):
                item_result[metric_name] = metric_data["item_scores"][idx]

        item_results.append(item_result)

    # Always return both aggregates and items
    return {
        "aggregate_results": [aggregate_result],
        "item_results": item_results,
    }


def _upload_results(
    metric_scores: Dict[str, Dict[str, Any]],
    items: List[Dict[str, Any]],
    inputs: List[Any],
    outputs: List[Any],
    labels: List[Any],
    hyperparameters: Dict[str, Any],
    dataset_name: str,
    experiment_id: str,
    project_id: str,
    metadata: Optional[Dict[str, Any]],
) -> None:
    """Upload results to Trismik platform."""
    # TODO: Implement upload functionality
    # This would use the Trismik client to upload the results
    # Similar to upload_classic_run_results in evaluate.py
    logger.warning("Upload functionality not yet implemented for score()")
