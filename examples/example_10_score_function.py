"""Example 10 - Scoring Pre-Computed Model Outputs."""

from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import score
from scorebook.metrics import Accuracy


def main() -> Any:
    """Score pre-computed model outputs using Scorebook.

    This example demonstrates how to score model outputs that have already been generated.

    Unlike evaluate(), which runs inference and then scores results, score() is used when you:
        - Already have model outputs from a previous run
        - Used a custom inference pipeline outside of Scorebook
        - Want to re-score existing outputs with different metrics

    The score() function requires items with 'output' and 'label' keys, and optionally 'input'.
    """

    # Create a list of items with pre-computed outputs and labels
    items = [
        {
            "input": "What is 2 + 2?",
            "output": "4",
            "label": "4",
        },
        {
            "input": "What is the capital of France?",
            "output": "Paris",
            "label": "Paris",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "output": "George R. R. Martin",
            "label": "William Shakespeare",
        },
    ]

    # Score the outputs against labels using the accuracy metric
    results = score(
        items=items,
        metrics=Accuracy,
        dataset="basic_questions",
        upload_results=False,  # Disable uploading for this example
    )

    print("\nScoring Results:")
    pprint(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_10")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_10_output.json")
