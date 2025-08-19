"""
Portkey Cloud Batch Inference Example.

This example demonstrates how to leverage Portkey's Batch API for cost-effective,
large-scale model evaluation using Scorebook. The backend provider of choice for this example is
OpenAI, but it's easy to adapt to any other provider.

This example requires a portkey account linked to an OpenAI account and
a portkey API key set in environment variable PORTKEY_API_KEY .

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
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function for batch API
    def preprocessor(eval_item: dict) -> list:
        """Pre-process dataset items into Portkey Batch API format."""
        prompt = eval_item["question"]

        # Create the batch API request messages format
        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and concisely as a single word",
            },
            {"role": "user", "content": prompt},
        ]

        return messages

    # Step 3: Define the postprocessing function
    def postprocessor(response: str) -> str:
        """Post-process Portkey batch response to extract the answer."""
        # The batch function returns the message content directly
        return response.strip()

    # Step 4: Create the inference pipeline for batch processing

    async def inference_function(items: list, **hyperparams: Any) -> Any:  # noqa
        return await batch(items, model=model_name, **hyperparams)

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    print(f"Running Portkey Batch API evaluation with model: {model_name}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    # For demonstration, limit to 25 items
    results = evaluate(inference_pipeline, dataset, item_limit=25, score_type="all")
    print("\nBatch evaluation completed!")
    print(results)

    # Step 6: Save results to file
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
        default=str(Path.cwd() / "results"),
        help="Directory to save evaluation outputs (JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for batch inference via Portkey (e.g., @openai/gpt-4.1-mini)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, str(args.model)


if __name__ == "__main__":
    main()
