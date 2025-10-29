"""
Google Cloud Vertex AI Model Inference Example.

This example demonstrates how to evaluate language models using Google Cloud
Vertex AI's Gemini models with Scorebook for real-time API calls.

Prerequisites: Google Cloud SDK (gcloud) authenticated and GOOGLE_CLOUD_PROJECT
environment variable set, or pass project ID as command line argument.
"""

import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from scorebook import EvalDataset, InferencePipeline, evaluate
from scorebook.inference.clients.vertex import responses
from scorebook.metrics import Accuracy


def main() -> None:
    """Run the Vertex AI inference example."""
    # Load environment variables from .env file for configuration
    load_dotenv()

    output_dir, model_name, project_id = setup_arguments()

    # Step 1: Load the evaluation dataset
    # Create an EvalDataset from local JSON file
    # - Uses 'answer' field as ground truth labels
    # - Configures Accuracy metric for evaluation
    # - Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function
    # Convert raw dataset items into Vertex AI API-compatible format
    # This function formats the question for the Gemini model
    def preprocessor(eval_item: Dict[str, str]) -> str:
        """Pre-process dataset items into Vertex AI string format."""
        return eval_item["question"]

    # Step 3: Define the postprocessing function
    # Extract the final answer from Vertex AI API response
    # Handles response parsing and returns the response text
    def postprocessor(response: Any) -> str:
        """Post-process Vertex AI response to extract the answer."""
        return str(response.text.strip())

    # Step 4: Create the inference pipeline for cloud-based evaluation
    # Combine preprocessing, Vertex AI inference, and postprocessing
    # Uses scorebook's built-in Vertex AI responses function for API calls

    # Create a system message with instructions for direct answers
    system_prompt = """
Answer the question directly and concisely.
Do not provide lengthy explanations unless specifically asked.
""".strip()

    async def inference_function(items: list, **hyperparams: Any) -> Any:
        return await responses(
            items,
            model=model_name,
            project_id=project_id,
            system_instruction=system_prompt,
            **hyperparams,
        )

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the cloud-based evaluation
    # Execute evaluation using Vertex AI with the inference pipeline
    # - Uses score_type="all" to get both aggregate and per-item results
    # - Limits to 10 items for quick demonstration and cost control
    print(f"Running Vertex AI evaluation with model: {model_name}")
    print(f"Project ID: {project_id}")
    print("Evaluating 10 items from local dataset...")

    results = evaluate(inference_pipeline, dataset, item_limit=10, score_type="all")
    print(results)

    # Step 6: Save results to file
    # Export evaluation results as JSON for later analysis
    output_file = output_dir / "vertex_messages_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_arguments() -> tuple[Path, str, str]:
    """Parse command line arguments."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run Vertex AI evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "results"),
        help="Directory to save evaluation outputs (JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Gemini model to use for inference (e.g., gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT env var)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle project ID fallback
    project_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "Project ID must be provided via --project-id or "
            "GOOGLE_CLOUD_PROJECT environment variable"
        )

    return output_dir, str(args.model), str(project_id)


if __name__ == "__main__":
    main()
