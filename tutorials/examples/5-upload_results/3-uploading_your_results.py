"""Tutorials - Upload Results - Example 3 - Uploading Pre-Scored Results."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import login, upload_result


def main() -> Any:
    """Upload pre-scored results directly to Trismik's dashboard.

    This example demonstrates how to upload results where metrics are ALREADY computed.
    This is different from score() or evaluate() which compute metrics for you.

    Use upload_result() when you:
        - Already have metric scores calculated
        - Used a custom evaluation framework that computed metrics
        - Want to import historical evaluation data with existing scores
        - Have results from external tools (e.g., other eval frameworks)

    The key difference from Examples 1 & 2:
        - Example 1 (score): You have outputs/labels → Scorebook computes metrics
        - Example 2 (evaluate): Scorebook runs inference AND computes metrics
        - Example 3 (upload_result): You have EVERYTHING including metrics → Just upload

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
    """

    # Step 1: Log in with your Trismik API key
    # login() reads TRISMIK_API_KEY from environment variables or .env file
    login()

    # Step 2: Format your pre-scored results
    # This is the structure that upload_result() expects:
    # - aggregate_results: List with one dict containing overall metrics
    # - item_results: List of dicts with per-item data and metric scores

    # Example: You already ran an evaluation with your custom framework
    # and computed accuracy, f1_score, etc.
    my_pre_scored_results = {
        "aggregate_results": [
            {
                "dataset": "spanish_translation",
                "accuracy": 0.8,  # Your pre-computed aggregate accuracy
                "bleu_score": 0.75,  # Your pre-computed BLEU score
                # Add any hyperparameters used (optional)
                "temperature": 0.7,
                "max_tokens": 100,
            }
        ],
        "item_results": [
            {
                "id": 0,
                "dataset": "spanish_translation",
                "input": "Translate 'hello' to Spanish",
                "output": "hola",
                "label": "hola",
                "accuracy": 1.0,  # Item-level metric scores
                "bleu_score": 1.0,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            {
                "id": 1,
                "dataset": "spanish_translation",
                "input": "Translate 'goodbye' to Spanish",
                "output": "adiós",
                "label": "adiós",
                "accuracy": 1.0,
                "bleu_score": 0.95,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            {
                "id": 2,
                "dataset": "spanish_translation",
                "input": "Translate 'thank you' to Spanish",
                "output": "gracias",
                "label": "gracias",
                "accuracy": 1.0,
                "bleu_score": 1.0,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            {
                "id": 3,
                "dataset": "spanish_translation",
                "input": "Translate 'please' to Spanish",
                "output": "por favor",
                "label": "por favor",
                "accuracy": 1.0,
                "bleu_score": 1.0,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            {
                "id": 4,
                "dataset": "spanish_translation",
                "input": "Translate 'good morning' to Spanish",
                "output": "buenos dias",  # Missing accent - wrong answer
                "label": "buenos días",
                "accuracy": 0.0,
                "bleu_score": 0.85,
                "temperature": 0.7,
                "max_tokens": 100,
            },
        ],
    }

    # Step 3: Upload your pre-scored results directly
    print("\nUploading pre-scored results to Trismik...")
    print("Metrics are already computed - just uploading to dashboard.\n")

    run_id = upload_result(
        run_result=my_pre_scored_results,
        experiment_id="Pre-Scored-Results-Example",
        project_id="TRISMIK_PROJECT_ID", # TODO: ADD YOUR TRISMIK PROJECT ID
        dataset_name="spanish_translation",
        hyperparameters={
            "temperature": 0.7,
            "max_tokens": 100,
        },
        metadata={
            "description": "Results with pre-computed metrics from custom framework",
            "source": "Custom evaluation tool",
            "evaluation_date": "2025-01-15",
        },
        model_name="my-custom-translator-v2",
    )

    print(f"\nResults uploaded successfully with run_id: {run_id}")

    # Add run_id to results for reference
    my_pre_scored_results["run_id"] = run_id

    pprint(my_pre_scored_results)
    return my_pre_scored_results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-uploading_your_results", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "3-uploading_your_results_output.json")