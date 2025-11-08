"""Helper functions shared between score() and score_async()."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from scorebook.exceptions import DataMismatchError, ParameterValidationError
from scorebook.metrics.metric_base import MetricBase
from scorebook.metrics.metric_registry import MetricRegistry
from scorebook.types import MetricScore
from scorebook.utils.async_utils import is_awaitable

logger = logging.getLogger(__name__)


def validate_items(items: List[Dict[str, Any]], output_column: str, label_column: str) -> None:
    """Validate the items parameter."""
    if not isinstance(items, list):
        raise ParameterValidationError("items must be a list")

    if len(items) == 0:
        raise ParameterValidationError("items list cannot be empty")

    required = {output_column, label_column}
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ParameterValidationError(f"Item at index {idx} is not a dict")

        missing = required - item.keys()
        if missing:
            for key in sorted(missing):
                raise ParameterValidationError(f"Item at index {idx} missing required '{key}' key")


def resolve_metrics(
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


async def calculate_metric_scores_async(
    metrics: List[MetricBase],
    outputs: List[Any],
    labels: List[Any],
    dataset_name: Optional[str],
    progress_bar: Optional[Any] = None,
) -> List[MetricScore]:
    """Calculate metric scores asynchronously (supports both sync and async metrics).

    Args:
        metrics: List of metric instances to compute scores for.
        outputs: List of model outputs.
        labels: List of ground truth labels.
        dataset_name: Name of the dataset being scored.
        progress_bar: Optional progress bar to update during computation.

    Returns:
        List of MetricScore objects containing aggregate and item-level scores.

    Raises:
        DataMismatchError: If outputs and labels have different lengths.
    """
    if len(outputs) != len(labels):
        raise DataMismatchError(len(outputs), len(labels), dataset_name)

    metric_scores: List[MetricScore] = []
    for metric in metrics:

        if progress_bar is not None:
            progress_bar.set_current_metric(metric.name)

        if is_awaitable(metric.score):
            aggregate_scores, item_scores = await metric.score(outputs, labels)
        else:
            aggregate_scores, item_scores = metric.score(outputs, labels)

        metric_scores.append(MetricScore(metric.name, aggregate_scores, item_scores))

        if progress_bar is not None:
            progress_bar.update(1)

    return metric_scores


def calculate_metric_scores(
    metrics: List[MetricBase],
    outputs: List[Any],
    labels: List[Any],
    dataset_name: Optional[str],
    progress_bar: Optional[Any] = None,
) -> List[MetricScore]:
    """Calculate metric scores synchronously (sync metrics only).

    Args:
        metrics: List of metric instances to compute scores for.
        outputs: List of model outputs.
        labels: List of ground truth labels.
        dataset_name: Name of the dataset being scored.
        progress_bar: Optional progress bar to update during computation.

    Returns:
        List of MetricScore objects containing aggregate and item-level scores.

    Raises:
        DataMismatchError: If outputs and labels have different lengths.
        ParameterValidationError: If any metric has an async score method.
    """
    if len(outputs) != len(labels):
        raise DataMismatchError(len(outputs), len(labels), dataset_name)

    metric_scores: List[MetricScore] = []
    for metric in metrics:

        if progress_bar is not None:
            progress_bar.set_current_metric(metric.name)

        if is_awaitable(metric.score):
            raise ParameterValidationError(
                f"Metric '{metric.name}' has an async score() method. "
                "Use score_async() instead of score() for async metrics."
            )

        aggregate_scores, item_scores = metric.score(outputs, labels)
        metric_scores.append(MetricScore(metric.name, aggregate_scores, item_scores))

        if progress_bar is not None:
            progress_bar.update(1)

    return metric_scores


def format_results(
    inputs: Optional[List[Any]],
    outputs: List[Any],
    labels: List[Any],
    metric_scores: List[MetricScore],
    hyperparameters: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Format results dict with both aggregates and items."""
    # Use defaults if not provided
    hyperparameters = hyperparameters or {}
    dataset_name = dataset_name or "scored_items"

    # Build aggregate results
    aggregate_result = {
        "dataset": dataset_name,
        **hyperparameters,
    }

    # Add aggregate scores from metrics
    for metric_score in metric_scores:
        for key, value in metric_score.aggregate_scores.items():
            score_key = (
                key if key == metric_score.metric_name else f"{metric_score.metric_name}_{key}"
            )
            aggregate_result[score_key] = value

    # Build item results
    item_results = []
    for idx in range(len(outputs)):
        item_result: Dict[str, Any] = {
            "id": idx,
            "dataset": dataset_name,
            "output": outputs[idx],
            "label": labels[idx],
            **hyperparameters,
        }

        # Add input if present
        if inputs is not None and inputs[idx] is not None:
            item_result["input"] = inputs[idx]

        # Add item-level metric scores
        for metric_score in metric_scores:
            if idx < len(metric_score.item_scores):
                item_result[metric_score.metric_name] = metric_score.item_scores[idx]

        item_results.append(item_result)

    # Always return both aggregates and items
    return {
        "aggregate_results": [aggregate_result],
        "item_results": item_results,
    }
