"""
Google Cloud Vertex AI Batch Inference Example.

This example demonstrates how to leverage Google Cloud Vertex AI's Batch API for
cost-effective, large-scale model evaluation using Scorebook. It uses Gemini models
for batch processing with automatic GCS upload/download and job management.

This example requires Google Cloud SDK (gsutil) to be installed and authenticated,
and a Google Cloud project with Vertex AI enabled. Set the project ID in the
GOOGLE_CLOUD_PROJECT environment variable or pass it as a command line argument.

Compare with the Portkey batch example to understand the differences
between different cloud providers' batch processing approaches.
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, InferencePipeline, evaluate
from scorebook.inference.clients.vertex import batch
from scorebook.metrics import Accuracy


def main() -> None:
    """Run the Vertex AI batch inference example."""
    # Load environment variables from .env file for configuration
    load_dotenv()

    output_dir, model_name, input_bucket, output_bucket, project_id = setup_arguments()

    # Step 1: Load the evaluation dataset
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function for Vertex AI Batch API
    def preprocessor(eval_item: dict) -> list:
        """Pre-process dataset items into Vertex AI Batch API format."""
        prompt = eval_item["question"]

        # Create the batch API request messages format for Vertex AI
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
        """Post-process Vertex AI batch response to extract the answer."""
        # The batch function returns the message content directly
        return response.strip()

    # Step 4: Create the inference pipeline for batch processing

    async def inference_function(items: list, **hyperparams: Any) -> Any:  # noqa
        return await batch(
            items,
            model=model_name,
            project_id=project_id,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            **hyperparams,
        )

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    print(f"Running Vertex AI Batch API evaluation with model: {model_name}")
    print(f"Project ID: {project_id}")
    print(f"Input bucket: {input_bucket}")
    print(f"Output bucket: {output_bucket}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    # For demonstration, limit to 25 items
    results = evaluate(inference_pipeline, dataset, item_limit=25, score_type="all")
    print("\nBatch evaluation completed!")
    print(results)

    # Step 6: Save results to file
    output_file = output_dir / "vertex_batch_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_arguments() -> tuple[Path, str, str, str, str]:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Vertex AI Batch API evaluation and save results."
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
        help="Gemini model to use for batch inference (e.g., gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=True,
        help="GCS bucket name for input data (without gs:// prefix)",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        required=True,
        help="GCS bucket name for output data (without gs:// prefix)",
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

    return (
        output_dir,
        str(args.model),
        str(args.input_bucket),
        str(args.output_bucket),
        str(project_id),
    )


if __name__ == "__main__":
    main()
