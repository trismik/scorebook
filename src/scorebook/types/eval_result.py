"""
This module defines the data structures used to represent evaluation results.

including individual prediction outcomes and aggregated dataset metrics.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from scorebook.types.eval_dataset import EvalDataset


@dataclass
class EvalResult:
    """
    Container for evaluation results from an entire dataset.

    Attributes:
        eval_dataset: The dataset used for evaluation.
        inference_outputs: A list of model predictions or outputs.
        metric_scores: A dictionary mapping metric names to their scores.
    """

    eval_dataset: EvalDataset
    inference_outputs: List[Any]
    metric_scores: Dict[str, Dict[str, Any]]
    hyperparams: Dict[str, Any]

    @property
    def item_scores(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries containing scores for each evaluated item."""
        results = []
        metric_names = list(self.metric_scores.keys()) if self.metric_scores else []

        for idx, item in enumerate(self.eval_dataset.items):
            if idx >= len(self.inference_outputs):
                break

            result = {
                "item_id": idx,
                "dataset_name": self.eval_dataset.name,
                **{
                    metric: self.metric_scores[metric]["item_scores"][idx]
                    for metric in metric_names
                },
            }
            results.append(result)

        return results

    @property
    def aggregate_scores(self) -> Dict[str, Any]:
        """Return the aggregated scores across all evaluated items."""
        result: Dict[str, Any] = {"dataset_name": self.eval_dataset.name}
        if not self.metric_scores:
            return result

        for metric, scores in self.metric_scores.items():
            # Flatten the aggregate scores from each metric into the result
            result.update(
                {
                    key if key == metric else f"{metric}_{key}": value
                    for key, value in scores["aggregate_scores"].items()
                }
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representing the evaluation results."""
        return {
            "aggregate": [
                {
                    **getattr(self.eval_dataset, "hyperparams", {}),
                    **self.aggregate_scores,
                }
            ],
            "per_sample": [item for item in self.item_scores],
        }

    def to_csv(self, file_path: str) -> None:
        """Save evaluation results to a CSV file.

        The CSV will contain item-level results.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write a header with all possible metric names
            item_fields = list(self.eval_dataset.items[0].keys()) if self.eval_dataset.items else []
            metric_names = list(self.metric_scores.keys()) if self.metric_scores else []
            header = ["item_id"] + item_fields + ["inference_output"] + metric_names
            writer.writerow(header)

            # Write item data
            for idx, item in enumerate(self.eval_dataset.items):
                if idx >= len(self.inference_outputs):
                    break

                row = (
                    [idx]
                    + list(item.values())
                    + [self.inference_outputs[idx]]
                    + [self.metric_scores[metric]["item_scores"][idx] for metric in metric_names]
                )
                writer.writerow(row)

    def to_json(self, file_path: str) -> None:
        """Save evaluation results to a JSON file in structured format (Option 2)."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        """Return a formatted string representation of the evaluation results."""
        result = [
            f"Eval Dataset: {self.eval_dataset.name}",
            "\nAggregate Scores:",
        ]
        for metric_name, score in self.aggregate_scores.items():
            result.append(f"\n  {metric_name}: {score:.4f}")
        return "".join(result)
