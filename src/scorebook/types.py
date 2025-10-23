"""Type definitions for scorebook evaluation framework."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from scorebook.eval_datasets import EvalDataset


@dataclass
class AdaptiveEvalDataset:
    """Represents a dataset configured for adaptive evaluation."""

    name: str


@dataclass
class EvalRunSpec:
    """Specification for a single evaluation run with dataset and hyperparameters."""

    dataset: EvalDataset
    dataset_index: int
    hyperparameter_config: Dict[str, Any]
    hyperparameters_index: int
    inputs: List[Any]
    labels: List[Any]

    def __str__(self) -> str:
        """Return string representation of EvalRunSpec."""
        return (
            f"EvalRunSpec(dataset={self.dataset.name}, "
            f"dataset_index={self.dataset_index}, "
            f"hyperparameter_config={self.hyperparameter_config}, "
            f"hyperparameters_index={self.hyperparameters_index})"
        )


@dataclass
class AdaptiveEvalRunSpec:
    """Specification for an adaptive evaluation run."""

    dataset: str
    dataset_index: int
    hyperparameter_config: Dict[str, Any]
    hyperparameters_index: int
    experiment_id: str
    project_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ClassicEvalRunResult:
    """Results from executing a classic evaluation run."""

    run_spec: EvalRunSpec
    run_completed: bool
    outputs: Optional[List[Any]]
    scores: Optional[Dict[str, Any]]
    run_id: Optional[str] = None

    @property
    def item_scores(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries containing scores for each evaluated item."""
        results = []

        if self.outputs:
            for idx, output in enumerate(self.outputs):
                if idx >= len(self.run_spec.inputs):
                    break

                result = {
                    "id": idx,
                    "dataset_name": self.run_spec.dataset.name,
                    "input": self.run_spec.inputs[idx],
                    "label": self.run_spec.labels[idx] if idx < len(self.run_spec.labels) else None,
                    "output": output,
                    **self.run_spec.hyperparameter_config,
                }

                # Add run_id if available
                if self.run_id is not None:
                    result["run_id"] = self.run_id

                # Add individual item scores if available
                if self.scores is not None:
                    for metric_name, metric_data in self.scores.items():
                        if isinstance(metric_data, dict) and "item_scores" in metric_data:
                            if idx < len(metric_data["item_scores"]):
                                result[metric_name] = metric_data["item_scores"][idx]
                        else:
                            # If scores is just a single value, replicate it for each item
                            result[metric_name] = metric_data

                results.append(result)

        return results

    @property
    def aggregate_scores(self) -> Dict[str, Any]:
        """Return the aggregated scores for this run."""
        result = {
            "dataset": self.run_spec.dataset.name,
            "run_completed": self.run_completed,
            **self.run_spec.hyperparameter_config,
        }

        # Add run_id if available
        if self.run_id is not None:
            result["run_id"] = self.run_id

        # Add aggregate scores from metrics
        if self.scores is not None:
            for metric_name, metric_data in self.scores.items():
                if isinstance(metric_data, dict) and "aggregate_scores" in metric_data:
                    # Flatten the aggregate scores from each metric
                    for key, value in metric_data["aggregate_scores"].items():
                        score_key = key if key == metric_name else f"{metric_name}_{key}"
                        result[score_key] = value
                else:
                    # If scores is just a single value, use it as is
                    result[metric_name] = metric_data

        return result


@dataclass
class AdaptiveEvalRunResult:
    """Results from executing an adaptive evaluation run."""

    run_spec: AdaptiveEvalRunSpec
    run_completed: bool
    scores: Dict[str, Any]

    @property
    def aggregate_scores(self) -> Dict[str, Any]:
        """Return the aggregated scores for this adaptive run."""
        result = {
            "dataset": self.run_spec.dataset,
            "experiment_id": self.run_spec.experiment_id,
            "project_id": self.run_spec.project_id,
        }

        # Safely unpack hyperparameter_config if it's not None
        if self.run_spec.hyperparameter_config:
            result.update(self.run_spec.hyperparameter_config)

        # Safely unpack metadata if it's not None
        if self.run_spec.metadata:
            result.update(self.run_spec.metadata)

        # Safely unpack scores if it's not None
        if self.scores:
            result.update(self.scores)

        return result


@dataclass
class EvalResult:
    """Container for evaluation results across multiple runs."""

    run_results: List[Union[ClassicEvalRunResult, AdaptiveEvalRunResult]]

    @property
    def item_scores(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries containing scores for each evaluated item."""
        results = []

        for run_result in self.run_results:
            if isinstance(run_result, ClassicEvalRunResult) and run_result.run_completed:
                results.extend(run_result.item_scores)

        return results

    @property
    def aggregate_scores(self) -> List[Dict[str, Any]]:
        """Return the aggregated scores across all evaluated runs."""
        results = []

        for run_result in self.run_results:
            results.append(run_result.aggregate_scores)

        return results
