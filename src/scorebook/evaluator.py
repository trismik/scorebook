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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult
from scorebook.utils import expand_dict


async def _evaluate_async(
    inference_fn: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
    return_type: str = "dict",
    score_type: str = "aggregate",
) -> Union[Dict, List]:
    """
    Evaluate model predictions using specified metrics on given datasets.

    This function runs the provided inference function on one or more evaluation datasets,
    computes metric scores, and returns the evaluation results. It supports batch processing,
    parameter sweeping, and different result formatting options.

    Args:
        inference_fn: Function that takes a list of dataset items and returns a list of predictions.
        datasets: One or more evaluation datasets to run evaluation on. Can be:
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
        ```python
        dataset = EvalDataset.from_huggingface("dataset_name", label="answer", metrics=[Precision])
        def inference_fn(items):
            # Model inference logic here - process all items at once
            return [prediction for item in items]

        results = evaluate(inference_fn, dataset, item_limit=100)
        ```
    """
    if score_type not in ["aggregate", "item", "all"]:
        raise ValueError("score_type must be 'aggregate', 'item', or 'all'")

    normalized_datasets = _normalize_datasets(datasets)
    hyper_param_configs: List[Dict[str, Any]] = expand_dict(hyperparameters or {})

    # [(dataset_name, {'outputs': [], 'labels': []}), 'hyperparameters': {}]
    dataset_results: List[Tuple[EvalDataset, Dict[str, List[Any]], Dict[str, Any]]] = []

    # Step 1 - Collect output from the inference function for each dataset.
    for eval_dataset in normalized_datasets:
        for hyperparameters in hyper_param_configs:

            # Collect all items and labels for this dataset
            items = eval_dataset.items
            if item_limit:
                items = items[:item_limit]

            labels = [item.get(eval_dataset.label) for item in items]

            # Call inference function with all items at once
            if asyncio.iscoroutinefunction(inference_fn):
                outputs = await inference_fn(items, hyperparameters)
            else:
                outputs = inference_fn(items, hyperparameters)

            inference_results: Dict[str, List[Any]] = {"outputs": outputs, "labels": labels}
            dataset_results.append((eval_dataset, inference_results, hyperparameters))

    # Step 2 - Calculate scores for each metric in each dataset and create eval results.
    eval_results: List[EvalResult] = []
    for eval_dataset, inference_results, hyperparameters in dataset_results:

        metric_scores = {}
        for metric in eval_dataset.metrics:
            aggregate_scores, item_scores = metric.score(
                inference_results["outputs"], inference_results["labels"]
            )
            metric_scores[metric.name] = {
                "aggregate_scores": aggregate_scores,
                "item_scores": item_scores,
            }

        eval_results.append(
            EvalResult(eval_dataset, inference_results["outputs"], metric_scores, hyperparameters)
        )

    # TODO: Implement experiment id
    if experiment_id:
        pass

    if return_type == "dict":
        if score_type == "all":
            # Combine results from all eval_results into single aggregate and per_sample lists
            combined_results: Dict[str, List[Dict[str, Any]]] = {"aggregate": [], "per_sample": []}
            for eval_result in eval_results:
                result_dict = eval_result.to_dict()
                combined_results["aggregate"].extend(result_dict["aggregate"])
                combined_results["per_sample"].extend(result_dict["per_sample"])
            return combined_results
        else:
            return_formats = {
                "aggregate": [eval_result.aggregate_scores for eval_result in eval_results],
                "item": [item for eval_result in eval_results for item in eval_result.item_scores],
            }
            return return_formats.get(score_type, {})

    else:
        return {eval_result.eval_dataset.name: eval_result for eval_result in eval_results}


def evaluate(
    inference_fn: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    hyperparameters: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
    return_type: str = "dict",
    score_type: str = "aggregate",
) -> Union[Dict, List]:
    """Wrap the async evaluate function for synchronous usage.

    This function provides backward compatibility for existing code while
    supporting both synchronous and asynchronous inference functions.

    Args:
        inference_fn: Function that takes a list of dataset items and returns a list of predictions.
                     Can be either synchronous or asynchronous.
        datasets: One or more evaluation datasets to run evaluation on.
        hyperparameters: Optional dictionary containing parameter sweep configuration.
        experiment_id: Optional string identifier for tracking multiple evaluation runs.
        item_limit: Optional integer limiting the number of items to evaluate per dataset.
        return_type: Format of the return value. Currently, only "dict" is supported.
        score_type: Type of score aggregation to return.

    Returns:
        Dictionary mapping dataset names to their evaluation results.
    """
    return asyncio.run(
        _evaluate_async(
            inference_fn=inference_fn,
            datasets=datasets,
            hyperparameters=hyperparameters,
            experiment_id=experiment_id,
            item_limit=item_limit,
            return_type=return_type,
            score_type=score_type,
        )
    )


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:

    if not isinstance(datasets, list):
        datasets = [datasets]

    # TODO: handle other types
    datasets = [d for d in datasets if isinstance(d, EvalDataset)]
    return datasets
