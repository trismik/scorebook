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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult
from scorebook.utils import evaluation_progress, expand_dict, is_awaitable


async def _evaluate_async(
    inference_callable: Callable,
    eval_datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
    return_type: str = "dict",
    score_type: str = "aggregate",
) -> Union[Dict, List]:
    """Run inference across datasets/hyperparams, compute metrics, and format results."""
    _validate_score_type(score_type)

    normalized_datasets = _normalize_datasets(eval_datasets)
    hyperparam_grid = _expand_hyperparams(hyperparameters)

    eval_results: List[EvalResult] = []

    with evaluation_progress(normalized_datasets, len(hyperparam_grid)) as progress_bars:
        # Loop through datasets, then hyperparameters for clear progress tracking
        for dataset_idx, eval_dataset in enumerate(normalized_datasets):
            with progress_bars.hyperparam_progress_context():
                # Run inference for each hyperparameter configuration on this dataset
                for hp_idx, hyperparam_config in enumerate(hyperparam_grid):
                    items = _clip_items(eval_dataset.items, item_limit)
                    labels = _labels_for(items, eval_dataset.label)

                    # 1) Run inference
                    outputs = await _run_inference_callable(
                        inference_callable, items, hyperparam_config
                    )

                    # 2) Score metrics
                    metric_scores = _score_metrics(eval_dataset, outputs, labels)

                    # 3) Wrap into EvalResult
                    eval_results.append(
                        EvalResult(eval_dataset, outputs, metric_scores, hyperparam_config)
                    )

                    # Update inner progress bar
                    progress_bars.update_hyperparam_progress()

            # Update the outer progress bar
            progress_bars.update_dataset_progress()

    # TODO: experiment_id handling (left as passthrough to preserve behavior)
    if experiment_id:
        pass

    # 4) Format as requested
    return _format_results(eval_results, return_type, score_type)


def evaluate(
    inference_callable: Callable,
    eval_datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
    return_type: str = "dict",
    score_type: str = "aggregate",
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
        item_limit: Optional integer limiting the number of items to evaluate per dataset.
        return_type: Format of the return value. Currently only "dict" is supported.
        score_type: Type of score aggregation to return. Options:
                   - "aggregate": Return aggregated metrics
                   - "item": Return per-item scores
                   - "all": Return both aggregate and per-item scores

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
    return asyncio.run(
        _evaluate_async(
            inference_callable=inference_callable,
            eval_datasets=eval_datasets,
            hyperparameters=hyperparameters,
            experiment_id=experiment_id,
            item_limit=item_limit,
            return_type=return_type,
            score_type=score_type,
        )
    )


# ===== Helper Functions =====


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:
    if not isinstance(datasets, list):
        datasets = [datasets]
    # TODO: handle other types (string registry, etc.)
    return [d for d in datasets if isinstance(d, EvalDataset)]


def _validate_score_type(score_type: str) -> None:
    if score_type not in {"aggregate", "item", "all"}:
        raise ValueError("score_type must be 'aggregate', 'item', or 'all'")


def _expand_hyperparams(hyperparameters: Optional[Dict[str, Any]]) -> Any:
    return expand_dict(hyperparameters or {})


def _clip_items(items: List[Dict[str, Any]], item_limit: Optional[int]) -> List[Dict[str, Any]]:
    return items[:item_limit] if item_limit else items


def _labels_for(items: List[Dict[str, Any]], label_key: str) -> List[Any]:
    return [item.get(label_key) for item in items]


async def _run_inference_callable(
    inference_callable: Callable,
    items: List[Dict[str, Any]],
    hyperparams: Dict[str, Any],
) -> Any:
    if is_awaitable(inference_callable):
        return await inference_callable(items, **hyperparams)
    else:
        return inference_callable(items, **hyperparams)


# Yields (eval_dataset, items, labels, hyperparams) for every dataset x hyperparam combo.
def _iter_dataset_jobs(
    datasets: List[EvalDataset],
    hyperparam_grid: List[Dict[str, Any]],
    item_limit: Optional[int],
) -> Iterable[Tuple[EvalDataset, List[Dict[str, Any]], List[Any], Dict[str, Any]]]:
    for eval_dataset in datasets:
        for hp in hyperparam_grid:
            items = _clip_items(eval_dataset.items, item_limit)
            labels = _labels_for(items, eval_dataset.label)
            yield eval_dataset, items, labels, hp


def _score_metrics(
    eval_dataset: EvalDataset, outputs: List[Any], labels: List[Any]
) -> Dict[str, Dict[str, Any]]:
    metric_scores: Dict[str, Dict[str, Any]] = {}
    for metric in eval_dataset.metrics:
        aggregate_scores, item_scores = metric.score(outputs, labels)
        metric_scores[metric.name] = {
            "aggregate_scores": aggregate_scores,
            "item_scores": item_scores,
        }
    return metric_scores


def _format_results(
    eval_results: List[EvalResult], return_type: str, score_type: str
) -> Union[Dict, List]:

    if return_type != "dict":
        return {er.eval_dataset.name: er for er in eval_results}

    if score_type == "all":
        combined: Dict[str, List[Dict[str, Any]]] = {"aggregate": [], "per_sample": []}
        for er in eval_results:
            d = er.to_dict()
            combined["aggregate"].extend(d["aggregate"])
            combined["per_sample"].extend(d["per_sample"])
        return combined

    if score_type == "aggregate":
        return [er.aggregate_scores for er in eval_results]

    if score_type == "item":
        return [item for er in eval_results for item in er.item_scores]

    # Should be unreachable due to validation
    return {}
