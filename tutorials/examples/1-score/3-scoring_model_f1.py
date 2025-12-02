"""Tutorials - Score - Example 3 - F1 Metric Scoring."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging
from scorebook import score
from scorebook.metrics.f1 import F1


def main() -> Any:
    """Score pre-computed model predictions using F1 metric.

    This example demonstrates how to score NER (Named Entity Recognition)
    predictions using the F1 metric with different averaging methods.
    """

    # Sample NER predictions (in CoNLL format with BIO tags)
    model_predictions = [
        {"output": "O", "label": "O"},
        {"output": "B-PER", "label": "B-PER"},
        {"output": "I-PER", "label": "I-PER"},
        {"output": "O", "label": "O"},
        {"output": "B-LOC", "label": "B-LOC"},
        {"output": "O", "label": "O"},
        {"output": "B-ORG", "label": "B-LOC"},  # Misclassification
        {"output": "O", "label": "B-MISC"},     # Missed entity
        {"output": "B-PER", "label": "B-PER"},
        {"output": "O", "label": "O"},
    ]

    print(f"Scoring {len(model_predictions)} NER predictions\n")

    # Score with all averaging methods at once
    print("All averaging methods:")
    results_all = score(
        items=model_predictions,
        metrics=F1(average="all"),
        upload_results=False,
    )
    pprint(results_all["aggregate_results"])

    # Score with specific combination of methods
    print("\nMicro and weighted averaging:")
    results_combo = score(
        items=model_predictions,
        metrics=F1(average=["micro", "weighted"]),
        upload_results=False,
    )
    pprint(results_combo["aggregate_results"])

    return results_all


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-scoring_f1_metric", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "3-scoring_f1_metric_output.json")