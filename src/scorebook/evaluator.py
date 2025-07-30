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

from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult


def evaluate(
    inference_fn: Callable,
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]],
    sweep: Optional[Dict[str, Any]] = None,
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

    # Step 1 - Collect output from the inference function for each dataset item.
    dataset_results: Dict[str, Dict[str, List[Any]]] = (
        {}
    )  # {dataset_name: {'outputs': [], 'labels': []}}
    for eval_dataset_name, eval_dataset in normalized_datasets.items():

        inference_results: Dict[str, List[Any]] = {"outputs": [], "labels": []}
        for idx, item in enumerate(eval_dataset.items):

            if item_limit and idx >= item_limit:
                break

            output, label = inference_fn(item), item.get(eval_dataset.label)
            inference_results["outputs"].append(output)
            inference_results["labels"].append(label)

        dataset_results[eval_dataset_name] = inference_results

    # Step 2 - Calculate scores for each metric in each dataset and create eval results.
    eval_results: List[EvalResult] = []
    for eval_dataset_name, inference_results in dataset_results.items():

        metric_scores = {}
        for metric in normalized_datasets[eval_dataset_name].metrics:
            aggregate_scores, item_scores = metric.score(
                inference_results["outputs"], inference_results["labels"]
            )
            metric_scores[metric.name] = {
                "aggregate_scores": aggregate_scores,
                "item_scores": item_scores,
            }

        eval_dataset = normalized_datasets.get(eval_dataset_name)
        eval_results.append(EvalResult(eval_dataset, inference_results["outputs"], metric_scores))

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


def _normalize_datasets(
    datasets: Union[str, EvalDataset, List[Union[str, EvalDataset]]]
) -> List[EvalDataset]:

    if not isinstance(datasets, list):
        datasets = [datasets]

    # TODO: handle other types
    datasets = [d for d in datasets if isinstance(d, EvalDataset)]
    return datasets
