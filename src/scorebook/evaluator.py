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

from typing import Any, Callable, Dict, List, Optional, Union

from scorebook.eval_dataset import EvalDataset
from scorebook.metrics import MetricBase, get_metrics
from scorebook.types import DatasetResults, EvalResult


def evaluate(
    inference_fn: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    metrics: Union[str, Callable, List[Union[str, Callable]]],
    sweep: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    item_limit: Optional[int] = None,
) -> Dict:
    """
    Evaluate model predictions using specified metrics on given datasets.

    Args:
        inference_fn: Function that takes a dataset item and returns a prediction
        datasets: One or more datasets to evaluate on
        metrics: One or more metrics to compute
        sweep: Optional parameter sweep configuration
        experiment_id: Optional experiment identifier
        item_limit: Optional limit on number of items to evaluate per dataset

    Returns:
        Dictionary mapping dataset names to their evaluation results
    """

    normalized_datasets = _normalize_datasets(datasets)
    normalized_metrics = _normalize_metrics(metrics)

    # TODO: Implement sweep
    if sweep:
        pass

    # First pass: collect predictions for all datasets
    results: Dict[str, DatasetResults] = {}
    for dataset in normalized_datasets:
        eval_results = []
        for idx, item in enumerate(dataset.items):
            if item_limit and idx >= item_limit:
                break

            prediction = inference_fn(item)
            eval_results.append(
                EvalResult(dataset_item=item, output=prediction, label=item.get(dataset.label))
            )

        results[dataset.name] = DatasetResults(items=eval_results, metrics={})

    # Second pass: compute metrics
    for dataset_name, dataset_results in results.items():
        outputs = [item.output for item in dataset_results.items]
        labels = [item.label for item in dataset_results.items]

        for metric in normalized_metrics:
            score = metric.score(outputs, labels)
            dataset_results.metrics[metric.name] = score

    # TODO: Implement experiment id
    if experiment_id:
        pass

    return results


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:

    if not isinstance(datasets, list):
        datasets = [datasets]

    # TODO: handle other types
    datasets = [d for d in datasets if isinstance(d, EvalDataset)]
    return datasets


def _normalize_metrics(
    metrics: Union[str, MetricBase, List[Union[str, MetricBase]]]
) -> List[MetricBase]:

    if not isinstance(metrics, list):
        metrics = [metrics]

    # TODO: handle other types
    metric_types = get_metrics()
    metrics = [m for m in metrics if m in metric_types]
    return metrics
