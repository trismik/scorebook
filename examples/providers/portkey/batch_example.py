"""
Portkey Cloud Batch Inference Example.

This example demonstrates how to leverage Portkey's Batch API for cost-effective,
large-scale model evaluation using Scorebook. The Batch API offers significant
cost savings and higher rate limits compared to standard API calls,
making it ideal for comprehensive evaluations.

Prerequisites:
- Portkey API key set in environment variable PORTKEY_API_KEY
- python-dotenv for environment variable management
- Active Portkey account with API credits
- portkey-ai package installed
- Model compatibility (batch API supports select models)

Compare with messages_example.py to understand the differences
between real-time and batch processing approaches.
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.portkey import batch
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the Portkey batch inference example."""
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

    # Step 2: Define the preprocessing function for batch API
    # Convert raw dataset items into Portkey Batch API-compatible format
    # The batch API requires a specific structure with messages format
    def preprocessor(eval_item: dict) -> list:
        """Pre-process dataset items into Portkey Batch API format."""
        prompt = eval_item["question"]

        # Create the batch API request messages format
        # This matches the structure expected by Portkey's chat completions endpoint
        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and concisely as a single word",
            },
            {"role": "user", "content": prompt},
        ]

        return messages

    # Step 3: Define the postprocessing function
    # Extract the final answer from Portkey Batch API response
    # The batch API returns responses in a different format than standard API
    def postprocessor(response: Any) -> str:
        """Post-process Portkey batch response to extract the answer."""
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
    # Uses scorebook's built-in Portkey batch function for API calls
    # This pipeline handles the complete batch workflow:
    # 1. Upload JSONL file to Portkey
    # 2. Create batch job
    # 3. Monitor progress
    # 4. Download and parse results

    async def inference_function(items: list, **hyperparams: Any) -> Any:  # noqa
        return await batch(items, model=model_name, **hyperparams)

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    # Execute evaluation using Portkey Batch API with the inference pipeline
    # - Uses score_type="all" to get both aggregate and per-item results
    # - Processes all items in dataset (remove item_limit for full evaluation)
    # - Batch processing is asynchronous and cost-effective
    print(f"Running Portkey Batch API evaluation with model: {model_name}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    # For demonstration, limit to 25 items to manage cost and time
    # Remove item_limit parameter for full dataset evaluation
    results = evaluate(inference_pipeline, dataset, item_limit=25, score_type="all")
    print("\nBatch evaluation completed!")
    print(results)

    # Step 6: Save results to file
    # Export evaluation results as JSON for later analysis
    output_file = output_dir / "portkey_batch_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_arguments() -> tuple[Path, str]:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Portkey Batch API evaluation and save results."
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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model to use for batch inference via Portkey. "
            "Note: Only select models support the Batch API. "
            "Example: @openai_bankmaker/gpt-4.1-mini"
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, str(args.model)


if __name__ == "__main__":
    main()
