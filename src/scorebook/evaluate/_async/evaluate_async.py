import asyncio
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

from trismik import TrismikAsyncClient, TrismikClient
from trismik.settings import evaluation_settings
from trismik.types import TrismikRunMetadata

from scorebook.eval_datasets import EvalDataset
from scorebook.evaluate.evaluate_helpers import (
    build_eval_run_specs,
    create_trismik_async_client,
    format_results,
    get_model_name,
    make_trismik_inference,
    prepare_datasets,
    prepare_hyperparameter_configs,
    resolve_split_async,
    validate_parameters,
)
from scorebook.exceptions import InferenceError, ScoreBookError
from scorebook.inference.inference_pipeline import InferencePipeline
from scorebook.score._async.score_async import score_async
from scorebook.types import (
    AdaptiveEvalRunResult,
    AdaptiveEvalRunSpec,
    ClassicEvalRunResult,
    EvalResult,
    EvalRunSpec,
)
from scorebook.utils import (
    async_nullcontext,
    evaluation_progress_context,
    resolve_show_progress,
    resolve_upload_results,
)

logger = logging.getLogger(__name__)


async def evaluate_async(
    inference: Union[Callable, InferencePipeline],
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
    show_progress: Optional[bool] = None,
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
        show_progress: If None, uses SHOW_PROGRESS_BARS from settings.
            If True/False, explicitly enables/disables progress bars for this evaluation.

    Returns:
        The evaluation results in the format specified by return parameters:
            - return_dict=True: Returns the evaluation results as a dict
            - return_dict=False: Returns an EvalResult object containing all run results
    """
    # Resolve and validate parameters
    upload_results = cast(bool, resolve_upload_results(upload_results))
    show_progress_bars = resolve_show_progress(show_progress)
    validate_parameters(locals(), evaluate_async)

    # Prepare datasets, hyperparameters, and eval run specs
    datasets = prepare_datasets(datasets, sample_size)
    hyperparameter_configs = prepare_hyperparameter_configs(hyperparameters)
    eval_run_specs = sorted(
        build_eval_run_specs(datasets, hyperparameter_configs, experiment_id, project_id, metadata),
        key=lambda run: (run.dataset_index, run.hyperparameters_index),
    )

    # Create a Trismik client if needed (for adaptive evals or uploads)
    needs_client = upload_results or any(
        isinstance(run, AdaptiveEvalRunSpec) for run in eval_run_specs
    )

    # Use context manager for automatic cleanup, or None if not needed
    trismik_client = create_trismik_async_client() if needs_client else None

    async with trismik_client or async_nullcontext():
        # Execute evaluation runs
        # Calculate total items across all runs
        total_items = sum(
            (
                len(run.dataset.items)
                if isinstance(run, EvalRunSpec)
                else evaluation_settings["max_iterations"]
            )  # Adaptive evals use max_iterations
            for run in eval_run_specs
        )
        model_display = get_model_name(inference)

        with evaluation_progress_context(
            total_eval_runs=len(eval_run_specs),
            total_items=total_items,
            dataset_count=len(datasets),
            hyperparam_count=len(hyperparameter_configs),
            model_display=model_display,
            enabled=show_progress_bars,
        ) as progress_bars:
            eval_result = await execute_runs(
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


async def execute_runs(
    inference: Callable,
    runs: List[Union[EvalRunSpec, AdaptiveEvalRunSpec]],
    progress_bars: Any,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_results: bool = False,
    trismik_client: Optional[Union[TrismikClient, TrismikAsyncClient]] = None,
) -> EvalResult:
    """Run evaluation in parallel."""

    # Worker function to execute individual runs and handle uploads
    async def worker(
        run: Union[EvalRunSpec, AdaptiveEvalRunSpec]
    ) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
        # Execute run (score_async handles upload internally for classic evals)
        run_result = await execute_run(
            inference, run, upload_results, experiment_id, project_id, metadata, trismik_client
        )

        # Update progress bars with items processed and success status
        if progress_bars is not None:
            # Classic evals have .items; adaptive evals use max_iterations
            items_processed = (
                len(run.dataset.items)
                if isinstance(run, EvalRunSpec)
                else evaluation_settings["max_iterations"]
            )
            progress_bars.on_run_completed(items_processed, run_result.run_completed)

        # Update upload progress for classic evals
        if (
            upload_results
            and isinstance(run_result, ClassicEvalRunResult)
            and run_result.run_completed
        ):
            # Check if upload succeeded by checking for run_id
            if experiment_id and project_id:
                upload_succeeded = run_result.run_id is not None
                if progress_bars is not None:
                    progress_bars.on_upload_completed(succeeded=upload_succeeded)

        return run_result

    # Execute all runs concurrently
    run_results = await asyncio.gather(*[worker(run) for run in runs])

    # Return in canonical (dataset_idx, hp_idx) order for stability
    run_results.sort(
        key=lambda result: (result.run_spec.dataset_index, result.run_spec.hyperparameters_index)
    )

    # Return EvalResult
    return EvalResult(run_results)


async def execute_run(
    inference: Callable,
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec],
    upload_results: bool,  # NEW PARAMETER
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Optional[Union[TrismikClient, TrismikAsyncClient]] = None,
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    """Execute a single evaluation run."""

    if isinstance(run, EvalRunSpec):
        return await execute_classic_eval_run(
            inference, run, upload_results, experiment_id, project_id, metadata
        )

    elif isinstance(run, AdaptiveEvalRunSpec):
        resolved_experiment_id = experiment_id if experiment_id is not None else run.experiment_id
        resolved_project_id = project_id if project_id is not None else run.project_id
        return await execute_adaptive_eval_run(
            inference,
            run,
            resolved_experiment_id,
            resolved_project_id,
            metadata,
            trismik_client,
        )

    else:
        raise ScoreBookError(f"An internal error occurred: {type(run)} is not a valid run type")


async def execute_classic_eval_run(
    inference: Callable,
    run: EvalRunSpec,
    upload_results: bool,
    experiment_id: Optional[str],
    project_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> ClassicEvalRunResult:
    """Execute a classic evaluation run using score_async() for scoring and uploading."""
    logger.debug("Executing classic eval run for %s", run)

    inference_outputs = None
    scores = None

    try:
        # 1. Run inference
        inference_outputs = await run_inference_callable(
            inference, run.inputs, run.hyperparameter_config
        )

        # 2. Build items for score_async
        items = [
            {
                "input": run.inputs[i] if i < len(run.inputs) else None,
                "output": inference_outputs[i],
                "label": run.labels[i] if i < len(run.labels) else "",
            }
            for i in range(len(inference_outputs))
        ]

        # 3. Get the model name for upload
        model_name = get_model_name(inference, metadata)

        # 4. Call score_async
        scores = await score_async(
            items=items,
            metrics=run.dataset.metrics,
            output_column="output",  # Explicit parameter
            label_column="label",  # Explicit parameter
            input_column="input",  # Explicit parameter
            hyperparameters=run.hyperparameter_config,
            dataset_name=run.dataset.name,
            model_name=model_name,
            metadata=metadata,
            experiment_id=experiment_id,
            project_id=project_id,
            upload_results=upload_results,
            show_progress=False,
        )

        # 5. Extract run_id if upload succeeded
        run_id = None
        if scores.get("aggregate_results") and len(scores["aggregate_results"]) > 0:
            run_id = scores["aggregate_results"][0].get("run_id")

        logger.debug("Classic evaluation completed for run %s (run_id: %s)", run, run_id)
        return ClassicEvalRunResult(
            run_spec=run,
            run_completed=True,
            outputs=inference_outputs,
            scores=scores,
            run_id=run_id,
        )

    except Exception as e:
        logger.warning("Failed to complete classic eval run for %s: %s", run, str(e))
        return ClassicEvalRunResult(
            run_spec=run,
            run_completed=False,
            outputs=inference_outputs,
            scores=scores,
            run_id=None,
        )


async def run_inference_callable(
    inference: Callable,
    inputs: List[Any],
    hyperparameter_config: Dict[str, Any],
) -> Any:
    """Run inference on a given dataset and hyperparameter configuration."""

    try:
        predictions = await inference(inputs, **hyperparameter_config)
    except Exception as e:
        logger.error(
            "Inference callable raised an exception: %s",
            str(e),
        )
        raise InferenceError(f"Inference failed: {str(e)}") from e

    if not isinstance(predictions, list) or len(predictions) != len(inputs):
        raise InferenceError(
            "Inference callable must return a list of predictions "
            "of shared length as the inputs. "
            f"Inputs length: {len(inputs)}, predictions length: {len(predictions)}"
        )

    if all(prediction == "" for prediction in predictions):
        logger.warning("Inference callable returned all empty strings for all items")

    if all(prediction is None for prediction in predictions):
        raise InferenceError("Inference callable returned all None for all items")

    return predictions


async def execute_adaptive_eval_run(
    inference: Callable,
    run: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Optional[Union[TrismikClient, TrismikAsyncClient]] = None,
) -> AdaptiveEvalRunResult:
    """Execute an adaptive evaluation run."""
    logger.debug("Executing adaptive run for %s", run)

    try:
        if trismik_client is None:
            raise ScoreBookError("Trismik client is required for adaptive evaluation")

        adaptive_eval_run_result = await run_adaptive_evaluation(
            inference, run, experiment_id, project_id, metadata, trismik_client
        )
        logger.debug("Adaptive evaluation completed for run %s", adaptive_eval_run_result)

        return adaptive_eval_run_result

    except Exception as e:
        logger.warning("Failed to complete adaptive eval run for %s: %s", run, str(e))
        return AdaptiveEvalRunResult(run, False, {})


async def run_adaptive_evaluation(
    inference: Callable,
    adaptive_run_spec: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Any,
    trismik_client: Union[TrismikClient, TrismikAsyncClient],
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
    # Resolve the split to use (with fallback: user-specified -> validation -> test)
    resolved_split = await resolve_split_async(
        test_id=adaptive_run_spec.dataset,
        user_specified_split=adaptive_run_spec.split,
        trismik_client=trismik_client,
    )

    trismik_results = await trismik_client.run(
        test_id=adaptive_run_spec.dataset,
        split=resolved_split,
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

    return AdaptiveEvalRunResult(run_spec=adaptive_run_spec, run_completed=True, scores=scores)
