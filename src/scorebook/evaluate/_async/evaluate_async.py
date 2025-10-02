import asyncio
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
    TrismikRunMetadata,
)

from scorebook.eval_dataset import EvalDataset
from scorebook.evaluate.evaluate_helpers import (
    build_eval_run_specs,
    create_trismik_async_client,
    format_results,
    get_model_name,
    make_trismik_inference,
    prepare_datasets,
    prepare_hyperparameter_configs,
    resolve_upload_results,
    score_metrics,
    validate_parameters,
)
from scorebook.exceptions import InferenceError, ScoreBookError
from scorebook.types import (
    AdaptiveEvalRunResult,
    AdaptiveEvalRunSpec,
    ClassicEvalRunResult,
    EvalResult,
    EvalRunSpec,
)
from scorebook.utils import evaluation_progress

logger = logging.getLogger(__name__)


async def evaluate_async(
    inference: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    return_dict: bool = True,
    return_aggregates: bool = True,
    return_items: bool = False,
    return_output: bool = False,
    upload_results: Union[Literal["auto"], bool] = "auto",
    sample_size: Optional[int] = None,
) -> Union[Dict, List, EvalResult]:
    """
    Evaluate a model across a collection of hyperparameters and datasets.

    Args:
        inference: The inference callable to evaluate
        datasets: Dataset(s) to evaluate on
        hyperparameters: Hyperparameter configuration(s) to evaluate with
        metadata: Optional metadata to attach to the evaluation
        experiment_id: Optional experiment identifier
        project_id: Optional project identifier
        return_dict: If True, returns eval results as a dict
        return_aggregates: If True, returns aggregate scores for each dataset
        return_items: If True, returns individual items for each dataset
        return_output: If True, returns model outputs for each dataset item
        upload_results: If True, uploads results to Trismik's dashboard
        sample_size: Optional number of items to sample from each dataset

    Returns:
        The evaluation results in the format specified by return parameters:
            - return_dict=True: Returns the evaluation results as a dict
            - return_dict=False: Returns an EvalResult object containing all run results
    """
    # Resolve and validate parameters
    upload_results = cast(bool, resolve_upload_results(upload_results))
    validate_parameters(locals(), evaluate_async)

    # Prepare datasets, hyperparameters, and eval run specs
    datasets = prepare_datasets(datasets, sample_size)
    hyperparameter_configs = prepare_hyperparameter_configs(hyperparameters)
    eval_run_specs = sorted(
        build_eval_run_specs(datasets, hyperparameter_configs, experiment_id, project_id, metadata),
        key=lambda run: (run.dataset_index, run.hyperparameters_index),
    )

    # Create Trismik client if needed (for adaptive evals or uploads)
    trismik_client = None
    needs_client = upload_results or any(
        isinstance(run, AdaptiveEvalRunSpec) for run in eval_run_specs
    )

    if needs_client:
        trismik_client = create_trismik_async_client()

    try:
        # Execute evaluation runs
        with evaluation_progress(
            dataset_count=len(datasets),
            hyperparameter_config_count=len(hyperparameter_configs),
            run_count=len(eval_run_specs),
        ) as progress_bars:
            eval_result = await execute_runs_async(
                inference,
                eval_run_specs,
                progress_bars,
                experiment_id,
                project_id,
                metadata,
                upload_results,
                trismik_client,
            )
            logger.info("Asynchronous evaluation complete")

        return format_results(
            eval_result, return_dict, return_aggregates, return_items, return_output
        )
    finally:
        # Clean up client
        if trismik_client and hasattr(trismik_client, "aclose"):
            try:
                await trismik_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing Trismik client: {e}")


async def execute_runs_async(
    inference: Callable,
    runs: List[Union[EvalRunSpec, AdaptiveEvalRunSpec]],
    progress_bars: Any,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_results: bool = False,
    trismik_client: Any = None,
) -> EvalResult:
    """Run evaluation in parallel."""

    # Worker function to execute individual runs and handle uploads
    async def worker(
        run: Union[EvalRunSpec, AdaptiveEvalRunSpec]
    ) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
        run_result = await execute_run_async(
            inference, run, experiment_id, project_id, metadata, trismik_client
        )
        progress_bars.on_eval_run_completed(run.dataset_index)

        if (
            upload_results
            and isinstance(run_result, ClassicEvalRunResult)
            and experiment_id
            and project_id
            and run_result.run_completed
        ):
            run_id = await upload_classic_run_results_async(
                run_result, experiment_id, project_id, inference, metadata, trismik_client
            )
            run_result.run_id = run_id

        return run_result

    # Execute all runs concurrently
    run_results = await asyncio.gather(*[worker(run) for run in runs])

    # Return in canonical (dataset_idx, hp_idx) order for stability
    run_results.sort(
        key=lambda result: (result.run_spec.dataset_index, result.run_spec.hyperparameters_index)
    )

    # Return EvalResult
    return EvalResult(run_results)


async def execute_run_async(
    inference: Callable,
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec],
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Any = None,
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    """Execute a single evaluation run."""

    if isinstance(run, EvalRunSpec):
        return await execute_classic_eval_run_async(inference, run)

    elif isinstance(run, AdaptiveEvalRunSpec):
        resolved_experiment_id = experiment_id if experiment_id is not None else run.experiment_id
        resolved_project_id = project_id if project_id is not None else run.project_id
        return await execute_adaptive_eval_run_async(
            inference,
            run,
            resolved_experiment_id,
            resolved_project_id,
            metadata,
            trismik_client,
        )

    else:
        raise ScoreBookError(f"An internal error occurred: {type(run)} is not a valid run type")


async def execute_classic_eval_run_async(
    inference: Callable, run: EvalRunSpec
) -> ClassicEvalRunResult:
    """Execute a classic evaluation run."""
    logger.debug("Executing classic eval run for %s", run)

    inference_outputs = None
    metric_scores = None

    try:
        inference_outputs = await run_inference_callable_async(
            inference, run.dataset.items, run.hyperparameter_config
        )
        metric_scores = score_metrics(run.dataset, inference_outputs, run.labels)
        logger.debug("Classic evaluation completed for run %s", run)
        return ClassicEvalRunResult(run, True, inference_outputs, metric_scores)

    except Exception as e:
        logger.warning("Failed to complete classic eval run for %s: %s", run, str(e))
        return ClassicEvalRunResult(run, False, inference_outputs, metric_scores)


async def run_inference_callable_async(
    inference: Callable,
    items: List[Dict[str, Any]],
    hyperparameter_config: Dict[str, Any],
) -> Any:
    """Run inference on a given dataset and hyperparameter configuration."""

    try:
        predictions = await inference(items, **hyperparameter_config)
    except Exception as e:
        logger.error(
            "Inference callable raised an exception: %s",
            str(e),
        )
        raise InferenceError(f"Inference failed: {str(e)}") from e

    if not isinstance(predictions, list) or len(predictions) != len(items):
        raise InferenceError(
            "Inference callable must return a list of predictions "
            "of shared length as the input items. "
            f"Items length: {len(items)}, predictions length: {len(predictions)}"
        )

    if all(prediction == "" for prediction in predictions):
        logger.warning("Inference callable returned all empty strings for all items")

    if all(prediction is None for prediction in predictions):
        raise InferenceError("Inference callable returned all None for all items")

    return predictions


async def execute_adaptive_eval_run_async(
    inference: Callable,
    run: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Any = None,
) -> AdaptiveEvalRunResult:
    """Execute an adaptive evaluation run."""
    logger.debug("Executing adaptive run for %s", run)

    adaptive_eval_run_result = await run_adaptive_evaluation_async(
        inference, run, experiment_id, project_id, metadata, trismik_client
    )
    logger.debug("Adaptive evaluation completed for run %s", adaptive_eval_run_result)

    return adaptive_eval_run_result


async def upload_classic_run_results_async(
    run_result: ClassicEvalRunResult,
    experiment_id: str,
    project_id: str,
    inference_callable: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Any = None,
) -> str:
    """Upload a classic evaluation run result to Trismik platform.

    Args:
        run: The evaluation run result to upload
        experiment_id: Trismik experiment identifier
        project_id: Trismik project identifier
        model: Model name used for evaluation
        metadata: Optional metadata dictionary
        trismik_client: Trismik client instance

    Returns:
        Run id
    """
    model = get_model_name(inference_callable)

    # Create eval items from run_spec items, outputs, and labels
    items: List[TrismikClassicEvalItem] = []
    for idx, (item, output) in enumerate(zip(run_result.run_spec.items, run_result.outputs)):
        label = run_result.run_spec.labels[idx] if idx < len(run_result.run_spec.labels) else ""

        # Calculate item-level metrics for this item
        item_metrics: Dict[str, Any] = {}
        if run_result.scores:
            for metric_name, metric_data in run_result.scores.items():
                if isinstance(metric_data, dict) and "item_scores" in metric_data:
                    if idx < len(metric_data["item_scores"]):
                        item_metrics[metric_name] = metric_data["item_scores"][idx]
                else:
                    # If scores is just a single value, use it for all items
                    item_metrics[metric_name] = metric_data

        eval_item = TrismikClassicEvalItem(
            datasetItemId=str(idx),
            modelInput=str(item),
            modelOutput=str(output),
            goldOutput=str(label),
            metrics=item_metrics,
        )
        items.append(eval_item)

    # Create eval metrics from run aggregate scores
    metrics: List[TrismikClassicEvalMetric] = []
    if run_result.scores:
        for metric_name, metric_data in run_result.scores.items():
            if isinstance(metric_data, dict) and "aggregate_scores" in metric_data:
                # Handle structured metric data with aggregate scores
                for agg_name, agg_value in metric_data["aggregate_scores"].items():
                    metric_id = (
                        f"{metric_name}_{agg_name}" if agg_name != metric_name else metric_name
                    )
                    metric = TrismikClassicEvalMetric(metricId=metric_id, value=agg_value)
                    metrics.append(metric)
            else:
                # Handle simple metric data (single value)
                metric = TrismikClassicEvalMetric(metricId=metric_name, value=metric_data)
                metrics.append(metric)

    classic_eval_request = TrismikClassicEvalRequest(
        project_id,
        experiment_id,
        run_result.run_spec.dataset.name,
        model,
        run_result.run_spec.hyperparameter_config,
        items,
        metrics,
    )

    response: TrismikClassicEvalResponse = trismik_client.submit_classic_eval(classic_eval_request)

    run_id: str = response.id
    logger.info(f"Classic eval run uploaded successfully with run_id: {run_id}")

    return run_id


async def run_adaptive_evaluation_async(
    inference: Callable,
    adaptive_run_spec: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Any,
    trismik_client: Any = None,
) -> AdaptiveEvalRunResult:
    """Run an adaptive evaluation using the Trismik API.

    Args:
        inference: Function to run inference
        adaptive_run_spec: Specification for the adaptive evaluation
        experiment_id: Experiment identifier
        project_id: Trismik project ID
        metadata: Additional metadata
        trismik_client: Trismik client instance
    Returns:
        Results from the adaptive evaluation
    """
    trismik_results = await trismik_client.run(
        test_id=adaptive_run_spec.dataset,
        project_id=project_id,
        experiment=experiment_id,
        run_metadata=TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="unknown"),
            test_configuration={},
            inference_setup={},
        ),
        item_processor=make_trismik_inference(inference),
        return_dict=False,
    )

    # Convert TrismikRunResults to AdaptiveEvalRunResult
    # Extract scores from the Trismik results
    scores = {}
    if hasattr(trismik_results, "scores") and trismik_results.scores:
        scores = trismik_results.scores
    elif hasattr(trismik_results, "__dict__"):
        # If scores aren't directly available, include all attributes as scores
        scores = {k: v for k, v in trismik_results.__dict__.items() if not k.startswith("_")}

    # Convert AdaptiveTestScore objects to JSON-serializable dictionaries
    def make_json_serializable(obj: Any) -> Any:
        if hasattr(obj, "theta") and hasattr(obj, "std_error"):
            # This is likely an AdaptiveTestScore object
            return {"theta": obj.theta, "std_error": obj.std_error}
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    # Make scores JSON serializable
    scores = make_json_serializable(scores)

    return AdaptiveEvalRunResult(run_spec=adaptive_run_spec, scores=scores)
