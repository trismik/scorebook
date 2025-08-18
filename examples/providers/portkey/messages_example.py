"""
Portkey Cloud Model Inference Example.

This example demonstrates how to evaluate language models using Portkey's inference
services with Scorebook for real-time API calls.

Prerequisites: PORTKEY_API_KEY environment variable and active Portkey account.
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.portkey import responses
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the Portkey inference example."""
    # Load environment variables from .env file for API keys
    load_dotenv()

    output_dir, model_name = setup_arguments()

    # Step 1: Load the evaluation dataset
    # Create an EvalDataset from local JSON file
    # - Uses 'answer' field as ground truth labels
    # - Configures Accuracy metric for evaluation
    # - Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function
    # Convert raw dataset items into Portkey API-compatible format
    # This function formats the question for the cloud model
    def preprocessor(eval_item: dict) -> list:
        """Pre-process dataset items into Portkey messages format."""
        prompt = eval_item["question"]

        # Create a system message with instructions for direct answers
        system_prompt = """
Answer the question directly and concisely.
Do not provide lengthy explanations unless specifically asked.
""".strip()

        # Format as messages for Portkey API
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    # Step 3: Define the postprocessing function
    # Extract the final answer from Portkey API response
    # Handles response parsing and returns the response text
    def postprocessor(response: Any) -> str:
        """Post-process Portkey response to extract the answer."""
        return str(response.choices[0].message.content.strip())

    # Step 4: Create the inference pipeline for cloud-based evaluation
    # Combine preprocessing, Portkey API inference, and postprocessing
    # Uses scorebook's built-in Portkey responses function for API calls

    async def inference_function(items: list, **hyperparams: Any) -> Any:
        return await responses(items, model=model_name, **hyperparams)

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the cloud-based evaluation
    # Execute evaluation using Portkey API with the inference pipeline
    # - Uses score_type="all" to get both aggregate and per-item results
    # - Limits to 10 items for quick demonstration and cost control
    print(f"Running Portkey evaluation with model: {model_name}")
    print("Evaluating 10 items from local dataset...")

    results = evaluate(inference_pipeline, dataset, item_limit=10, score_type="all")
    print(results)

    # Step 6: Save results to file
    # Export evaluation results as JSON for later analysis
    output_file = output_dir / "portkey_messages_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_arguments() -> tuple[Path, str]:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Portkey evaluation and save results.")
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
        help="Model to use for inference via Portkey (e.g., @openai/gpt-4.1-mini)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, str(args.model)


if __name__ == "__main__":
    main()
