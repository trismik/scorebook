"""
Model evaluation functionality for the Scorebook framework.

This module provides the core evaluation logic to assess model predictions
against ground truth labels using configurable metrics. It supports:

- Batch evaluation of models across multiple datasets
- Flexible metric computation and aggregation
- Optional parameter sweeping and experiment tracking
- Customizable inference functions

The main entry point is the `evaluate()` function which handles running
models on datasets and computing metric scores.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from scorebook.eval_dataset import EvalDataset
from scorebook.exceptions import (
    DataMismatchError,
    MetricComputationError,
    ParallelExecutionError,
    ParameterValidationError,
    ScoreBookError,
)
from scorebook.trismik_services import run_adaptive_evaluation
from scorebook.trismik_services.login import get_token
from scorebook.trismik_services.upload_classic_eval_run import upload_classic_eval_run
from scorebook.types import (
    AdaptiveEvalDataset,
    AdaptiveEvalRunResult,
    AdaptiveEvalRunSpec,
    ClassicEvalRunResult,
    EvalResult,
    EvalRunSpec,
)
from scorebook.utils import evaluation_progress, expand_dict, is_awaitable

logger = logging.getLogger(__name__)


def evaluate(
    inference: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_results: Union[Literal["auto"], bool] = "auto",
    sample_size: Optional[int] = None,
    parallel: bool = False,
    return_dict: bool = True,
    return_aggregates: bool = True,
    return_items: bool = False,
    return_output: bool = False,
) -> Union[Dict, List]:
    """
    Evaluate a model and collection of hyperparameters over datasets with specified metrics.

    Args:
        inference: A callable that runs model inference over a list of evaluation items
        datasets: One or more evaluation datasets to run evaluation on.
        hyperparameters: Optional list of hyperparameter configurations or grid to evaluate
        experiment_id: Optional ID of the experiment to upload results to on Trismik's dashboard.
        project_id: Optional ID of the project to upload results to on Trismik's dashboard.
        metadata: Optional metadata to attach to the evaluation.
        upload_results: If True, uploads results to Trismik's dashboard.
        sample_size: Optional number of items to sample from each dataset.
        parallel: If True, runs evaluation in parallel. Requires the inference callable to be async.
        return_dict: If True, returns eval results as a dict
        return_aggregates: If True, returns aggregate scores for each dataset
        return_items: If True, returns individual items for each dataset
        return_output: If True, returns model outputs for each dataset item evaluated

    Returns:
        Union[Dict, List, EvalResult]:
        The evaluation results in the format specified by return parameters:
            - If return_dict=False: Returns an EvalResult object containing all run results
            - If return_dict=True Returns the evaluation results as a dict
    """

    logger.info(
        "Starting evaluation: experiment_id=%s, project_id=%s, parallel=%s",
        experiment_id,
        project_id,
        parallel,
    )

    return asyncio.run(
        _evaluate_async(
            inference=inference,
            datasets=datasets,
            hyperparameters=hyperparameters,
            metadata=metadata,
            experiment_id=experiment_id,
            project_id=project_id,
            parallel=parallel,
            return_dict=return_dict,
            return_aggregates=return_aggregates,
            return_items=return_items,
            return_output=return_output,
            upload_results=upload_results,
            sample_size=sample_size,
        )
    )


async def _evaluate_async(
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
    parallel: bool = False,
    upload_results: Union[Literal["auto"], bool] = "auto",
    sample_size: Optional[int] = None,
) -> Union[Dict, List]:
    """Run evaluation asynchronously."""

    upload_results = _resolve_upload_results(upload_results)

    _validate_parameters(locals())
    datasets = _prepare_datasets(datasets, sample_size)
    hyperparameter_configs = _prepare_hyperparameter_configs(hyperparameters)

    logger.info(
        "Prepared %d datasets and %d hyperparameter configurations",
        len(datasets),
        len(hyperparameter_configs),
    )

    eval_run_specs = _build_eval_run_specs(
        datasets, hyperparameter_configs, experiment_id, project_id, metadata
    )
    eval_run_specs.sort(key=lambda run: (run.dataset_index, run.hyperparameters_index))

    logger.info("Created %d evaluation run specs", len(eval_run_specs))

    with evaluation_progress(
        datasets, len(hyperparameter_configs), parallel, len(eval_run_specs)
    ) as progress_bars:
        if parallel:
            eval_result = await _run_parallel(
                inference,
                eval_run_specs,
                progress_bars,
                experiment_id,
                project_id,
                metadata,
                upload_results,
            )
        else:
            eval_result = await _run_sequential(
                inference,
                eval_run_specs,
                progress_bars,
                experiment_id,
                project_id,
                metadata,
                upload_results,
            )

        logger.info("Evaluation completed successfully")

    return _format_results(eval_result, return_dict, return_aggregates, return_items, return_output)


# ===== ORCHESTRATION PATHS =====


async def _run_parallel(
    inference: Callable,
    runs: List[Union[EvalRunSpec, AdaptiveEvalRunSpec]],
    progress_bars: Any,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_results: bool = False,
) -> EvalResult:
    """Run evaluation in parallel."""

    logger.debug("Running inference in parallel")

    # Worker function to execute individual runs and handle uploads
    async def worker(
        run: Union[EvalRunSpec, AdaptiveEvalRunSpec]
    ) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
        run_result = await _execute_run(inference, run, experiment_id, project_id, metadata)
        progress_bars.on_eval_run_completed(run.dataset_index)

        if (
            upload_results
            and isinstance(run_result, ClassicEvalRunResult)
            and experiment_id
            and project_id
        ):
            # Only upload runs that completed successfully
            if run_result.run_completed:
                run_id = await _upload_classic_run(
                    run_result, experiment_id, project_id, inference, metadata
                )
                run_result.run_id = run_id

        return run_result

    # Execute all runs concurrently
    run_results = await asyncio.gather(*[worker(run) for run in runs])
    # Return in canonical (dataset_idx, hp_idx) order for stability
    run_results.sort(
        key=lambda result: (result.run_spec.dataset_index, result.run_spec.hyperparameters_index)
    )
    return EvalResult(run_results)


async def _run_sequential(
    inference: Callable,
    runs: List[Union[EvalRunSpec, AdaptiveEvalRunSpec]],
    progress_bars: Any,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    upload_results: bool = False,
) -> EvalResult:
    """Run evaluation sequentially."""

    logger.debug("Running inference sequentially")

    run_results: List[Union[ClassicEvalRunResult, AdaptiveEvalRunResult]] = []
    for run in runs:
        run_result = await _execute_run(inference, run, experiment_id, project_id, metadata)
        run_results.append(run_result)
        progress_bars.on_hyperparam_completed(run_result.run_spec.dataset_index)

        # Upload a classic eval run result immediately if upload_results is enabled
        if (
            upload_results
            and isinstance(run_result, ClassicEvalRunResult)
            and experiment_id
            and project_id
        ):
            # Only upload runs that completed successfully
            if run_result.run_completed:
                run_id = await _upload_classic_run(
                    run_result, experiment_id, project_id, inference, metadata
                )
                run_result.run_id = run_id

    return EvalResult(run_results)


# ===== EVALUATION RUN EXECUTIONS =====


async def _execute_run(
    inference: Callable,
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec],
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    """Execute a single evaluation run."""

    if isinstance(run, EvalRunSpec):
        return await _execute_classic_eval_run(inference, run)

    elif isinstance(run, AdaptiveEvalRunSpec):
        if not experiment_id or not project_id:
            raise ScoreBookError(
                "experiment_id and project_id are required for adaptive evaluations"
            )
        return await _execute_adaptive_eval_run(inference, run, experiment_id, project_id, metadata)

    else:
        raise ScoreBookError(f"An internal error occurred: {type(run)} is not a valid run type")


async def _execute_classic_eval_run(inference: Callable, run: EvalRunSpec) -> ClassicEvalRunResult:
    """Execute a classic evaluation run."""
    logger.debug("Executing classic eval run for %s", run)

    inference_outputs = None
    metric_scores = None

    try:
        inference_outputs = await _run_inference_callable(
            inference, run.dataset.items, run.hyperparameter_config
        )
        metric_scores = _score_metrics(run.dataset, inference_outputs, run.labels)
        logger.debug("Classic evaluation completed for run %s", run)
        return ClassicEvalRunResult(run, True, inference_outputs, metric_scores)

    except Exception as e:
        logger.warning("Failed to complete classic eval run for %s: %s", run, str(e))
        return ClassicEvalRunResult(run, False, inference_outputs, metric_scores)


async def _execute_adaptive_eval_run(
    inference: Callable,
    run: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AdaptiveEvalRunResult:
    """Execute an adaptive evaluation run."""
    logger.debug("Executing adaptive run for %s", run)

    adaptive_eval_run_result = await run_adaptive_evaluation(
        inference, run, experiment_id, project_id, metadata
    )
    logger.debug("Adaptive evaluation completed for run %s", adaptive_eval_run_result)

    return adaptive_eval_run_result


# ===== HELPER FUNCTIONS =====


def _resolve_upload_results(upload_results: Union[Literal["auto"], bool]) -> bool:
    """Resolve the upload_results parameter based on trismik login status."""

    if upload_results == "auto":
        upload_results = get_token() is not None
        logger.debug("Auto upload results resolved to: %s", upload_results)

    return upload_results


def _validate_parameters(params: Dict[str, Any]) -> None:
    """Validate all parameters for evaluation."""

    # If returning a dict, it must contain items and/or aggregates
    if params["return_dict"] and not params["return_aggregates"] and not params["return_items"]:
        raise ParameterValidationError(
            "When return_dict=True, at least one of return_aggregates or return_items must be True"
        )

    # Parallel runs require an asynchronous inference callable
    if params["parallel"] and not is_awaitable(params["inference"]):
        raise ParallelExecutionError(
            "parallel=True requires the inference_callable to be async. "
            "Please make your inference function async or set parallel=False."
        )

    # If uploading results, experiment_id and project_id must be specified
    if params["upload_results"]:
        if params["experiment_id"] is None or params["project_id"] is None:
            raise ParameterValidationError(
                "experiment_id and project_id are required for upload_results=True"
            )

    logger.debug("Parameter validation successful")


def _prepare_datasets(
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


def _prepare_hyperparameter_configs(
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


def _build_eval_run_specs(
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
                    _build_classic_eval_run_spec(
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
                    _build_adaptive_eval_run_spec(
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


def _build_classic_eval_run_spec(
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


def _build_adaptive_eval_run_spec(
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


async def _run_inference_callable(
    inference: Callable,
    items: List[Dict[str, Any]],
    hyperparameter_config: Dict[str, Any],
) -> Any:
    """Run inference on a given dataset and hyperparameter configuration."""
    if is_awaitable(inference):
        return await inference(items, **hyperparameter_config)
    else:
        return inference(items, **hyperparameter_config)


def _score_metrics(
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


async def _upload_classic_run(
    run_result: ClassicEvalRunResult,
    experiment_id: str,
    project_id: str,
    inference_callable: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Upload a ClassicEvalRunResult to Trismik."""

    logger.debug("Uploading classic eval run: %s", run_result.run_spec)
    try:
        model_name = _get_model_name(inference_callable, metadata)
        response = await upload_classic_eval_run(
            run=run_result,
            experiment_id=experiment_id,
            project_id=project_id,
            model=model_name,
            metadata=metadata,
        )
        logger.info("Successfully uploaded classic eval run: %s", response.id)
        return str(response.id)

    except Exception as e:
        logger.error("Failed to upload classic eval run: %s", str(e))
        return None


def _get_model_name(
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


def _format_results(
    eval_result: EvalResult,
    return_dict: bool,
    return_aggregates: bool,
    return_items: bool,
    return_output: bool,
) -> Union[EvalResult, Dict, List]:

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
