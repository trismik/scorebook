"""
Cloud Batch Inference Example.

This example demonstrates how to leverage OpenAI's Batch API for cost-effective,
large-scale model evaluation using Scorebook. The Batch API offers significant
cost savings (50% off) and higher rate limits compared to standard API calls,
making it ideal for comprehensive evaluations.

Prerequisites:
- OpenAI API key set in environment variable OPENAI_API_KEY
- python-dotenv for environment variable management
- Active OpenAI account with API credits
- Model compatibility (batch API supports select models)

Compare with example_5_cloud_inference.py to understand the differences
between real-time and batch processing approaches.
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import batch
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the cloud batch inference example."""
    # Load environment variables from .env file for API keys
    load_dotenv()

    output_dir = setup_output_directory()

    # Step 1: Load the evaluation dataset
    # Create an EvalDataset from local JSON file
    # - Uses 'answer' field as ground truth labels
    # - Configures Accuracy metric for evaluation
    # - Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function for batch API
    # Convert raw dataset items into OpenAI Batch API-compatible format
    # The batch API requires a specific JSON structure with chat completions format
    def preprocessor(eval_item: dict, hyperparameters: dict) -> dict:
        """Pre-process dataset items into OpenAI Batch API format."""
        prompt = eval_item["question"]

        # Create the batch API request body format
        # This matches the structure expected by OpenAI's /v1/chat/completions endpoint
        batch_request = {
            "model": "gpt-4o-mini",  # Batch API compatible model
            "messages": [
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely as a single word",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
        }

        return batch_request

    # Step 3: Define the postprocessing function
    # Extract the final answer from OpenAI Batch API response
    # The batch API returns responses in a different format than standard API
    def postprocessor(response: Any, **hyperparameters: Any) -> str:
        """Post-process OpenAI batch response to extract the answer."""
        # The batch function returns the message content directly
        # after parsing the batch results file
        if isinstance(response, str):
            return response.strip()

        # Fallback for other response formats
        try:
            if hasattr(response, "choices") and response.choices:
                return str(response.choices[0].message.content.strip())
        except (AttributeError, IndexError):
            pass

        return str(response).strip() if response else ""

    # Step 4: Create the inference pipeline for batch processing
    # Uses scorebook's built-in OpenAI batch function for API calls
    # This pipeline handles the complete batch workflow:
    # 1. Upload JSONL file to OpenAI
    # 2. Create batch job
    # 3. Monitor progress
    # 4. Download and parse results
    model_name = setup_model_selection()
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=batch,  # Use batch instead of responses for batch processing
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    # Execute evaluation using OpenAI Batch API with the inference pipeline
    # - Uses score_type="all" to get both aggregate and per-item results
    # - Processes all items in dataset (remove item_limit for full evaluation)
    # - Batch processing is asynchronous and cost-effective
    print(f"Running OpenAI Batch API evaluation with model: {model_name}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    # For demonstration, limit to 25 items to manage cost and time
    # Remove item_limit parameter for full dataset evaluation
    results = evaluate(
        inference_pipeline,
        dataset,
        hyperparameters=[{"temperature": 0.6}, {"temperature": 0.7}, {"temperature": 0.8}],
        sample_size=25,
        return_aggregates=True,
        return_items=True,
    )
    print("\nBatch evaluation completed!")
    print(results)

    # Step 6: Save results to file
    # Export evaluation results as JSON for later analysis
    output_file = output_dir / "example_6_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OpenAI Batch API evaluation and save results."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/example_results"),
        help=(
            "Directory to save evaluation outputs (JSON). "
            "Defaults to ./examples/example_results in the current working directory."
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_model_selection() -> str:
    """Parse model selection from command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Select OpenAI model for batch evaluation.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "OpenAI model to use for batch inference. "
            "Note: Only select models support the Batch API. "
            "Supported models include: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo. "
            "Default: gpt-4o-mini"
        ),
    )
    args = parser.parse_args()
    return str(args.model)


if __name__ == "__main__":
    main()
