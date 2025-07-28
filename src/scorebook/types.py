"""
Type definitions for evaluation results in the Scorebook framework.

This module defines the data structures used to represent evaluation results,
including individual prediction outcomes and aggregated dataset metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvaluatedItem:
    """
    Container for a single evaluated item.

    Attributes:
        item: Original item from the eval dataset that was evaluated
        output: Model's inferred output for this item
        label: Ground truth label for this item
    """

    item: Dict[str, Any]
    output: Any
    label: Any
    scores: Dict


@dataclass
class EvalResult:
    """
    Container for evaluation results from an entire dataset.

    Attributes:
        dataset: The name of the dataset evaluated
        items: List of individual evaluation results for each dataset item
        scores: Dictionary mapping metric names to their computed scores
        metrics: list of all metrics used in the evaluation
    """

    dataset: str
    items: List[EvaluatedItem]
    scores: Dict[str, float]
    metrics: List

    def __init__(self, dataset: str, items: List[EvaluatedItem], metrics: List):
        """Initialize EvalResult Instance."""
        self.dataset = dataset
        self.items = items
        self.metrics = metrics
        self.scores = {metric.name: metric.score(evaluated_items=items) for metric in metrics}

    @property
    def item_scores(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries containing scores for each evaluated item."""
        return [{"label": item.label, "output": item.output} | item.scores for item in self.items]

    @property
    def aggregate_scores(self) -> Dict[str, Any]:
        """Return the aggregated scores across all evaluated items."""
        return self.scores

    def to_csv(self, file_path: str) -> None:
        """Save evaluation results to a CSV file.

        The CSV will contain both item-level results and aggregate scores.

        Args:
            file_path: Path where the CSV file will be written
        """
        import csv
        from pathlib import Path

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write dataset info and aggregate scores
            writer.writerow(["Dataset Name:", self.dataset])
            writer.writerow([])  # Empty row for readability
            writer.writerow(["Aggregate Scores:"])
            for metric_name, score in self.aggregate_scores.items():
                writer.writerow([metric_name, f"{score:.4f}"])

            writer.writerow([])  # Empty row for readability

            # Write item results
            writer.writerow(["Item Results:"])

            # Write header with all possible metric names
            metric_names = list(self.items[0].scores.keys()) if self.items else []
            item_keys = list(self.items[0].item.keys()) if self.items else []
            header = item_keys + ["Output", "Label"] + metric_names
            writer.writerow(header)

            # Write item data
            for item in self.items:
                row = (
                    [item.item.get(key, "") for key in item_keys]  # Get values for all item keys
                    + [item.output, item.label]
                    + [item.scores.get(metric, "") for metric in metric_names]
                )
                writer.writerow(row)

    def to_json(self, file_path: str) -> None:
        """Save evaluation results to a JSON file.

        The JSON will contain both item-level results and aggregate scores.

        Args:
            file_path: Path where the JSON file will be written
        """
        import json
        from pathlib import Path

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset": self.dataset,
            "aggregate_scores": self.aggregate_scores,
            "items": [
                {
                    "question": item.item.get("question", ""),
                    "output": item.output,
                    "label": item.label,
                    "scores": item.scores,
                }
                for item in self.items
            ],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
