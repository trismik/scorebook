"""Tutorials - Upload Results - Example 3 - Uploading Your Own Results."""

import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import login, score


def main() -> Any:
    """Upload your own pre-computed scoring results to Trismik's dashboard.

    This example demonstrates how to upload results that you've already
    computed or obtained from external sources. This is useful when:
        - You have results from a previous evaluation
        - You used a custom evaluation framework
        - You want to import historical data into Trismik
        - You computed scores using your own metrics

    The key is to format your data as items with outputs and labels,
    then use score() to compute standardized metrics and upload.

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
    """

    # Example: You have results from running your model externally
    # Format them as items with input, output, and label fields
    my_model_results = [
        {
            "input": "Translate 'hello' to Spanish",
            "output": "hola",
            "label": "hola",
        },
        {
            "input": "Translate 'goodbye' to Spanish",
            "output": "adiós",
            "label": "adiós",
        },
        {
            "input": "Translate 'thank you' to Spanish",
            "output": "gracias",
            "label": "gracias",
        },
        {
            "input": "Translate 'please' to Spanish",
            "output": "por favor",
            "label": "por favor",
        },
        {
            "input": "Translate 'good morning' to Spanish",
            "output": "buenos días",
            "label": "buenos días",
        },
    ]

    # Step 1: Log in with your Trismik API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        raise ValueError(
            "TRISMIK_API_KEY environment variable must be set. "
            "Get your API key from https://app.trismik.com/settings"
        )
    login(api_key)

    # Step 2: Get project ID from environment
    project_id = os.environ.get("TRISMIK_PROJECT_ID")
    if not project_id:
        raise ValueError(
            "TRISMIK_PROJECT_ID environment variable must be set. "
            "Find your project ID at https://app.trismik.com"
        )

    # Step 3: Score and upload your results
    # score() will compute metrics and upload to Trismik
    print("\nUploading your pre-computed results...")
    print("Scorebook will compute metrics and upload to Trismik dashboard.\n")

    results = score(
        items=my_model_results,
        metrics="accuracy",  # Let Scorebook compute standard metrics
        dataset_name="spanish_translation",
        model_name="my-custom-model-v1",
        experiment_id="Custom-Results-Upload",
        project_id=project_id,
        metadata={
            "description": "Results from external evaluation",
            "source": "Custom evaluation framework",
            "date": "2025-01-15",
        },
        upload_results=True,
    )

    print("\nYour results uploaded successfully!")
    print(f"View your results at: https://app.trismik.com/projects/{project_id}\n")

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-uploading_your_results")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "3-uploading_your_results_output.json")