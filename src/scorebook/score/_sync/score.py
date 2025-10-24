import logging
from typing import Any, Dict, List, Literal, Optional, Union, cast

from scorebook.evaluate.evaluate_helpers import resolve_upload_results
from scorebook.exceptions import ParameterValidationError
from scorebook.score.score_helpers import (
    calculate_metric_scores,
    format_results,
    resolve_metrics,
    validate_items,
)
from scorebook.trismik.upload_results import upload_run_result
from scorebook.types import Metrics

logger = logging.getLogger(__name__)


def score(
    items: List[Dict[str, Any]],
    metrics: Metrics,
    output: str = "output",
    label: str = "label",
    input: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    upload_results: Union[Literal["auto"], bool] = "auto",
    show_progress: Optional[bool] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Score pre-computed model outputs against labels using specified metrics."""

    # Resolve and validate parameters
    upload_results = cast(bool, resolve_upload_results(upload_results))

    # Validate upload requirements
    if upload_results and (experiment_id is None or project_id is None):
        raise ParameterValidationError(
            "experiment_id and project_id are required to upload a run",
        )

    # Validate items parameter
    validate_items(items, output, label)

    # Validate hyperparameters is a dict (not list)
    if hyperparameters is not None and not isinstance(hyperparameters, dict):
        raise ParameterValidationError("hyperparameters must be a dict")

    # Resolve metrics to a list of Metrics
    metric_instances = resolve_metrics(metrics)

    # Extract outputs and labels from items
    input_key = input if input is not None else "input"
    inputs = [item.get(input_key) for item in items]
    outputs = [item.get(output) for item in items]
    labels = [item.get(label) for item in items]

    # Compute scores for each metric
    metric_scores = calculate_metric_scores(metric_instances, outputs, labels, dataset_name)

    # Build results
    results: Dict[str, List[Dict[str, Any]]] = format_results(
        inputs=inputs,
        outputs=outputs,
        labels=labels,
        metric_scores=metric_scores,
        hyperparameters=hyperparameters,
        dataset_name=dataset_name,
    )

    # Upload if requested
    if upload_results and experiment_id and project_id:
        try:
            run_id = upload_run_result(
                run_result=results,
                experiment_id=experiment_id,
                project_id=project_id,
                dataset_name=dataset_name,
                hyperparameters=hyperparameters,
                metadata=metadata,
                model_name=model_name,
            )
            logger.info(f"Score results uploaded successfully with run_id: {run_id}")

            # Add run_id to aggregate results
            if results.get("aggregate_results"):
                results["aggregate_results"][0]["run_id"] = run_id

            # Add run_id to each item result
            if results.get("item_results"):
                for item in results["item_results"]:
                    item["run_id"] = run_id

        except Exception as e:
            logger.warning(f"Failed to upload score results: {e}")
            # Don't raise - continue execution even if upload fails

    logger.info("Scoring complete")
    return results
