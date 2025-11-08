"""Tutorials - Score - Example 1 - Scoring Models."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from scorebook.utils.tutorial_utils import save_results_to_json, setup_logging

from scorebook import score
from scorebook.metrics.accuracy import Accuracy


def main() -> Any:
    """Score pre-computed model predictions using Scorebook.

    This example demonstrates how to score generated model predictions.
    """

    # Prepare a list of items with generated outputs and labels
    model_predictions = [
        {"output": "4", "label": "4"},
        {"output": "Paris", "label": "Paris"},
        {"output": "George R. R. Martin", "label": "William Shakespeare"},
    ]

    # Score the predictions against labels using the accuracy metric
    results = score(
        items=model_predictions,
        metrics=Accuracy,
        upload_results=False,  # Disable uploading for this example
    )

    print("\nResults:")
    pprint(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="1-scoring_model_accuracy", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "1-scoring_model_accuracy_output.json")
