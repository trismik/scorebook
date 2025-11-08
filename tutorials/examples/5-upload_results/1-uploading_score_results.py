"""Tutorials - Upload Results - Example 1 - Uploading score() Results."""

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import login, score
from scorebook.metrics.accuracy import Accuracy


def main() -> Any:
    """Score pre-computed outputs and upload results to Trismik's dashboard.

    This example demonstrates how to upload score() results to Trismik.
    The score() function is used when you already have model outputs and
    want to score them against labels.

    Use score() when you want to:
        - Score pre-computed model outputs
        - Re-score existing results with different metrics
        - Upload scoring results without re-running inference

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
    """

    # Prepare items with pre-computed outputs and labels
    items = [
        {
            "input": "What is 2 + 2?",
            "output": "4",
            "label": "4"
        },
        {
            "input": "What is the capital of France?",
            "output": "Paris",
            "label": "Paris"
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "output": "William Shakespeare",
            "label": "William Shakespeare"
        },
        {
            "input": "What is 5 * 6?",
            "output": "30",
            "label": "30"
        },
        {
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter",
            "label": "Jupiter"
        },
    ]

    # Step 1: Log in with your Trismik API key
    login("TRISMIK_API_KEY") # TODO: ADD YOUR TRISMIK API KEY

    # Step 2: Score the outputs and upload results
    # When you provide experiment_id and project_id, results are automatically uploaded
    results = score(
        items=items,
        metrics=Accuracy,
        dataset_name="basic_questions",
        model_name="gpt-4o-mini",
        experiment_id="Score-Upload-Example",
        project_id="TRISMIK_PROJECT_ID", # TODO: ADD YOUR TRISMIK PROJECT ID
        metadata={
            "description": "Example demonstrating score() result uploading",
            "note": "These are pre-computed outputs",
        },
        upload_results=True,  # Explicitly enable uploading
    )

    print("\nResults uploaded successfully!")
    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="1-uploading_score_results", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "1-uploading_score_results_output.json")