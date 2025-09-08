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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scorebook.exceptions import (
    DataMismatchError,
    MetricComputationError,
    ParallelExecutionError,
    ParameterValidationError,
)
from scorebook.types import EvalDataset, EvalResult, EvalRunSpec
from scorebook.utils import evaluation_progress, expand_dict, is_awaitable

logger = logging.getLogger(__name__)


def evaluate(
    inference_callable: Callable,
    eval_datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    parallel: bool = False,
    return_dict: bool = True,
    return_aggregates: bool = True,
    return_items: bool = False,
    return_output: bool = False,
    sample_size: Optional[int] = None,
) -> Union[Dict, List]:
    """
    Evaluate model predictions using specified metrics on given datasets.

    This function runs the provided inference callable on one or more evaluation datasets,
    computes metric scores, and returns the evaluation results. It supports batch processing,
    parameter sweeping, and different result formatting options.

    Args:
        inference_callable: A callable function or object that takes (items, hyperparameters)
                           and returns predictions. Can be a regular function, async function,
                           or callable instance (like a class with __call__ method).
        eval_datasets: One or more evaluation datasets to run evaluation on. Can be:
                 - A single EvalDataset instance
                 - A list of EvalDataset instances
                 - A string identifier (for future dataset registry support)
                 - A list of string identifiers
        hyperparameters: Optional dictionary containing hyperparameter sweep configuration.
        experiment_id: Optional string identifier for tracking multiple evaluation runs.
        return_dict: If True, returns eval results as a dict
        return_aggregates: If True, returns aggregate scores for each dataset
        return_items: If True, returns individual items for each dataset
        return_output: If True, returns model outputs for each dataset item evaluated
        sample_size: If set, only return a sample of the dataset items (for debugging)
        parallel: If True, run inference functions in parallel (requires all functions to be async)

    Returns:
        Dictionary mapping dataset names to their evaluation results. For each dataset,
        returns a dictionary containing:
        - items: List of EvalResult objects with predictions and ground truth
        - metrics: Dictionary mapping metric names to their computed scores

    Example:

    python
        dataset = EvalDataset.from_huggingface("dataset_name", label="answer", metrics=[Precision])
        def inference_fn(items):
            # Model inference logic here - process all items at once
            return [prediction for item in items]

        results = evaluate(inference_fn, dataset, item_limit=100)
    """

    logger.info(
        "Starting evaluation: experiment_id=%s, project_id=%s, parallel=%s",
        experiment_id,
        project_id,
        parallel,
    )

    return asyncio.run(
        _evaluate_async(
            inference_callable=inference_callable,
            eval_datasets=eval_datasets,
            hyperparameters=hyperparameters,
            experiment_id=experiment_id,
            project_id=project_id,
            parallel=parallel,
            return_dict=return_dict,
            return_aggregates=return_aggregates,
            return_items=return_items,
            return_output=return_output,
            sample_size=sample_size,
        )
    )


async def _evaluate_async(
    inference_callable: Callable,
    eval_datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    return_dict: bool = True,
    return_aggregates: bool = True,
    return_items: bool = False,
    return_output: bool = False,
    parallel: bool = False,
    sample_size: Optional[int] = None,
) -> Union[Dict, List]:
    _validate_parameters(locals())
    datasets, adaptive_datasets = _prepare_datasets(eval_datasets, sample_size)
    hyperparameters = _prepare_hyperparameters(hyperparameters)

    logger.info(
        "Prepared %d datasets and %d hyperparameter configurations",
        len(datasets),
        len(hyperparameters),
    )

    runs = _build_runs(datasets, hyperparameters)
    runs.sort(key=lambda run: (run.dataset_idx, run.hp_idx))

    logger.info("Created %d evaluation runs", len(runs))

    with evaluation_progress(datasets, len(hyperparameters), parallel, len(runs)) as progress_bars:
        if parallel:
            eval_results = await _run_parallel(inference_callable, runs, progress_bars)
        else:
            eval_results = await _run_sequential(inference_callable, runs, progress_bars)

        logger.info("Evaluation completed successfully")

    return _format_results(
        eval_results, return_dict, return_aggregates, return_items, return_output
    )


# ===== ORCHESTRATION PATHS =====


async def _run_parallel(
    inference_callable: Callable,
    runs: List[EvalRunSpec],
    progress_bars: Any,
) -> List["EvalResult"]:
    logger.debug("Running inference in parallel")

    async def worker(run: EvalRunSpec) -> Tuple[EvalRunSpec, "EvalResult"]:
        er = await _execute_run(inference_callable, run)
        progress_bars.on_eval_run_completed(run.dataset_idx)
        return run, er

    pairs = await asyncio.gather(*[worker(r) for r in runs])
    # Return in canonical (dataset_idx, hp_idx) order for stability
    pairs.sort(key=lambda p: (p[0].dataset_idx, p[0].hp_idx))
    return [er for _, er in pairs]


async def _run_sequential(
    inference_callable: Callable,
    runs: List[EvalRunSpec],
    progress_bars: Any,
) -> List["EvalResult"]:
    logger.debug("Running inference sequentially")
    results: List["EvalResult"] = []
    for run in runs:
        er = await _execute_run(inference_callable, run)
        results.append(er)
        progress_bars.on_hyperparam_completed(run.dataset_idx)
    return results


# ===== EVALUATION EXECUTIONS =====


async def _execute_run(inference_callable: Callable, run: EvalRunSpec) -> "EvalResult":
    logger.debug("Executing run for %s", run)

    outputs = await _run_inference_callable(inference_callable, run.items, run.hyperparams)
    logger.debug("Inference completed for run %s", run)

    metric_scores = _score_metrics(run.eval_dataset, outputs, run.labels)
    logger.debug("Metrics computed for run %s. - scores:  %s", run, list(metric_scores.keys()))

    return EvalResult(run.eval_dataset, outputs, metric_scores, run.hyperparams)


# ===== HELPER FUNCTIONS =====


def _validate_parameters(params: Dict[str, Any]) -> None:
    """Validate all parameters for evaluation."""

    if params["return_dict"] and not params["return_aggregates"] and not params["return_items"]:
        raise ParameterValidationError(
            "When return_dict=True, at least one of return_aggregates or return_items must be True"
        )

    if params["parallel"] and not is_awaitable(params["inference_callable"]):
        raise ParallelExecutionError(
            "parallel=True requires the inference_callable to be async. "
            "Please make your inference function async or set parallel=False."
        )


def _prepare_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    sample_size: Optional[int] = None,
) -> Tuple[List[EvalDataset], List[str]]:
    """Prepare and separate input datasets into classic and adaptive evaluation datasets."""

    # Ensure datasets is always a list for consistent processing
    if not isinstance(datasets, list):
        datasets = [datasets]

    # Extract classical datasets TODO: handle other types (string registry)
    classic_eval_datasets = [dataset for dataset in datasets if isinstance(dataset, EvalDataset)]

    # Reduce datasets to a random sample
    if sample_size:
        logger.info("Sampling datasets to %d items each", sample_size)
        for dataset in classic_eval_datasets:
            dataset.shuffle()
            if len(dataset) > sample_size:
                original_size = len(dataset)
                dataset._hf_dataset = dataset._hf_dataset.select(range(sample_size))
                logger.debug(
                    "Sampled dataset '%s' from %d to %d items",
                    dataset.name,
                    original_size,
                    sample_size,
                )

    # Extract adaptive dataset strings
    adaptive_eval_datasets = [
        dataset.replace(":adaptive", "")
        for dataset in datasets
        if isinstance(dataset, str) and dataset.endswith(":adaptive")
    ]

    logger.info("Evaluating on classic datasets: %s", [ds.name for ds in classic_eval_datasets])
    logger.info("Evaluating on adaptive datasets: %s", adaptive_eval_datasets)

    return classic_eval_datasets, adaptive_eval_datasets


def _prepare_hyperparameters(
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


async def _run_inference_callable(
    inference_callable: Callable,
    items: List[Dict[str, Any]],
    hyperparams: Dict[str, Any],
) -> Any:
    if is_awaitable(inference_callable):
        return await inference_callable(items, **hyperparams)
    else:
        return inference_callable(items, **hyperparams)


def _build_runs(
    datasets: List["EvalDataset"],
    hyperparameters: List[Dict[str, Any]],
) -> List[EvalRunSpec]:
    """Build RunSpec objects for each dataset/hyperparameter combination."""
    runs: List[EvalRunSpec] = []
    for d_idx, ds in enumerate(datasets):
        items = ds.items
        labels = [item.get(ds.label) for item in items]
        for hp_idx, hp in enumerate(hyperparameters):
            run_spec = EvalRunSpec(d_idx, ds, items, labels, hp, hp_idx)
            logger.debug("Built RunSpec: %s", run_spec)
            runs.append(run_spec)
    return runs


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


def _format_results(
    eval_results: List[EvalResult],
    return_dict: bool,
    return_aggregates: bool,
    return_items: bool,
    return_output: bool,
) -> Union[Dict, List]:

    # Return results as a dict
    if return_dict:

        # Include both aggregate and item scores in dict returned
        if return_aggregates and return_items:
            results: Dict[str, List[Dict[str, Any]]] = {"aggregate_results": [], "item_results": []}
            for eval_result in eval_results:
                eval_result_dict = eval_result.to_dict()
                results["aggregate_results"].extend(eval_result_dict["aggregate_results"])
                if return_output:
                    results["item_results"].extend(eval_result_dict["item_results"])
                else:
                    results["item_results"].extend(
                        [
                            {k: v for k, v in item.items() if k != "inference_output"}
                            for item in eval_result_dict["item_results"]
                        ]
                    )
            return results

        # Include only aggregate scores in dict returned
        elif return_aggregates:
            return [eval_result.aggregate_scores for eval_result in eval_results]

        # Include only item scores in dict returned
        else:
            if return_output:
                return [item for eval_result in eval_results for item in eval_result.item_scores]
            else:
                return [
                    {k: v for k, v in item.items() if k != "inference_output"}
                    for eval_result in eval_results
                    for item in eval_result.item_scores
                ]

    # Return results as an EvalResult object
    else:
        out: Dict[str, List[EvalResult]] = {}
        for er in eval_results:
            out.setdefault(er.eval_dataset.name, []).append(er)
        return out
