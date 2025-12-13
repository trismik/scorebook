"""Type definitions for scorebook evaluation framework."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from scorebook.eval_datasets.eval_dataset import EvalDataset
from scorebook.metrics.core.metric_base import MetricBase

# Type alias for metrics parameter
Metrics = Union[
    str, MetricBase, Type[MetricBase], Sequence[Union[str, MetricBase, Type[MetricBase]]]
]


@dataclass
class AdaptiveEvalDataset:
    """Represents a dataset configured for adaptive evaluation."""

    name: str
    split: Optional[str] = None


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
    split: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ClassicEvalRunResult:
    """Results from executing a classic evaluation run."""

    run_spec: EvalRunSpec
    run_completed: bool
    outputs: Optional[List[Any]]
    scores: Optional[Dict[str, List[Dict[str, Any]]]]  # score_async format
    run_id: Optional[str] = None

    @property
    def item_scores(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries containing scores for each evaluated item."""
        if self.scores and "item_results" in self.scores:
            # score_async already built this in the exact format we need
            return self.scores["item_results"]
        return []

    @property
    def aggregate_scores(self) -> Dict[str, Any]:
        """Return the aggregated scores for this run."""
        if (
            self.scores
            and "aggregate_results" in self.scores
            and len(self.scores["aggregate_results"]) > 0
        ):
            result = self.scores["aggregate_results"][0].copy()
            # Add run_completed (not included in score_async format)
            result["run_completed"] = self.run_completed
            return result

        # Fallback if no scores available
        return {
            "dataset": self.run_spec.dataset.name,
            "run_completed": self.run_completed,
            **self.run_spec.hyperparameter_config,
        }


@dataclass
class AdaptiveEvalRunResult:
    """Results from executing an adaptive evaluation run."""

    run_spec: AdaptiveEvalRunSpec
    run_completed: bool
    scores: Dict[str, Any]
    run_id: Optional[str] = None

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


@dataclass
class MetricScore:
    """Container for metric scores across multiple runs."""

    metric_name: str
    aggregate_scores: Dict[str, Any]
    item_scores: List[Dict[str, Any]]
