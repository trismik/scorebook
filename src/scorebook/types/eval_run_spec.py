"""Evaluation run specification types for Scorebook."""

from typing import Any, Dict, List, NamedTuple

from scorebook.types import EvalDataset


class EvalRunSpec(NamedTuple):
    """Represents a single evaluation run configuration."""

    dataset_idx: int
    eval_dataset: EvalDataset
    items: List[Dict[str, Any]]
    labels: List[Any]
    hyperparams: Dict[str, Any]
    hp_idx: int

    def __str__(self) -> str:
        """Return a formatted string summary of the evaluation run specification."""
        hyperparams_str = ", ".join([f"{k}={v}" for k, v in self.hyperparams.items()])

        return (
            f"EvalRunSpec(dataset_idx={self.dataset_idx},"
            f"  hp_idx={self.hp_idx},"
            f"  dataset_name='{self.eval_dataset.name}',"
            f"  hyperparams=[{hyperparams_str}]"
            f")"
        )
