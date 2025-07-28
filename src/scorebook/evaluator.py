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

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Union

from scorebook.eval_dataset import EvalDataset
from scorebook.types import EvalResult, EvaluatedItem


def evaluate(
    inference_fn: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    sweep: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
    return_type: str = "dict",
    score_type: str = "aggregate",
) -> Dict:
    """
    Evaluate model predictions using specified metrics on given datasets.

    This function runs the provided inference function on one or more evaluation datasets,
    computes metric scores, and returns the evaluation results. It supports batch processing,
    parameter sweeping, and different result formatting options.

    Args:
        inference_fn: Function that takes a dataset item and returns a prediction.
        datasets: One or more evaluation datasets to run evaluation on. Can be:
                 - A single EvalDataset instance
                 - A list of EvalDataset instances
                 - A string identifier (for future dataset registry support)
                 - A list of string identifiers
        sweep: Optional dictionary containing parameter sweep configuration.
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
        def inference_fn(item):
            # Model inference logic here
            return prediction

        results = evaluate(inference_fn, dataset, item_limit=100)
        ```
    """
    if score_type not in ["aggregate", "item", "all"]:
        raise ValueError("score_type must be 'aggregate', 'item', or 'all'")

    normalized_datasets = {dataset.name: dataset for dataset in _normalize_datasets(datasets)}

    # TODO: Implement sweep
    if sweep:
        pass

    # First pass: collect predictions for all datasets
    eval_results: List[EvalResult] = []
    for eval_dataset_name, eval_dataset in normalized_datasets.items():

        evaluated_items: List[EvaluatedItem] = []
        for idx, item in enumerate(eval_dataset.items):

            if item_limit and idx >= item_limit:
                break

            output = inference_fn(item)
            label = item.get(eval_dataset.label)
            scores = {
                metric.name: metric.score(output=output, label=label)
                for metric in eval_dataset.metrics
            }
            evaluated_items.append(
                EvaluatedItem(item=item, output=output, label=label, scores=scores)
            )

        eval_results.append(
            EvalResult(
                dataset=eval_dataset_name, items=evaluated_items, metrics=eval_dataset.metrics
            )
        )

    # TODO: Implement experiment id
    if experiment_id:
        pass

    if return_type == "dict":
        results = {}
        for eval_result in eval_results:
            if score_type == "aggregate":
                results[eval_result.dataset] = {
                    "items": [asdict(item) for item in eval_result.items],
                    "scores": eval_result.aggregate_scores,
                }
            elif score_type == "item":
                results[eval_result.dataset] = {
                    "items": [asdict(item) for item in eval_result.items],
                    "scores": eval_result.item_scores,
                }
            elif score_type == "all":
                results[eval_result.dataset] = {
                    "items": [asdict(item) for item in eval_result.items],
                    "scores": {
                        "aggregate": eval_result.aggregate_scores,
                        "items": eval_result.item_scores,
                    },
                }
        return results

    else:
        return {eval_result.dataset: eval_result for eval_result in eval_results}


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:

    if not isinstance(datasets, list):
        datasets = [datasets]

    # TODO: handle other types
    datasets = [d for d in datasets if isinstance(d, EvalDataset)]
    return datasets
