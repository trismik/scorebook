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
from scorebook.types import DatasetResults, EvalResult


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
    results: Dict[str, DatasetResults] = {}
    for dataset_name, dataset in normalized_datasets.items():
        eval_results = []
        for idx, item in enumerate(dataset.items):
            if item_limit and idx >= item_limit:
                break

            prediction = inference_fn(item)
            eval_results.append(
                EvalResult(dataset_item=item, output=prediction, label=item.get(dataset.label))
            )

        results[dataset_name] = DatasetResults(items=eval_results, metrics={})

    # Second pass: compute metrics
    for dataset_name, dataset_results in results.items():
        outputs = [item.output for item in dataset_results.items]
        labels = [item.label for item in dataset_results.items]

        for metric in normalized_datasets[dataset_name].metrics:
            score = metric.score(outputs, labels, score_type=score_type)
            dataset_results.metrics[metric.name] = score

    # TODO: Implement experiment id
    if experiment_id:
        pass

    if return_type == "dict":
        results = {ds_name: asdict(ds_results) for ds_name, ds_results in results.items()}

    return results


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:

    if not isinstance(datasets, list):
        datasets = [datasets]

    # TODO: handle other types
    datasets = [d for d in datasets if isinstance(d, EvalDataset)]
    return datasets
