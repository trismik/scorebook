"""
AWS Bedrock Batch Inference Example.

This example demonstrates how to leverage AWS Bedrock's Model Invocation Jobs for
cost-effective, large-scale model evaluation using Scorebook. It uses Claude models
for batch processing with automatic S3 upload/download and job management.

This example requires AWS CLI to be configured with appropriate credentials and
permissions for Bedrock and S3. Set up your AWS profile and ensure you have
the necessary IAM roles configured.

Prerequisites:
- AWS CLI configured with appropriate profile
- S3 bucket with proper permissions
- IAM role for Bedrock execution with S3 access
- Minimum 100 items for batch processing (AWS requirement)
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

from scorebook import EvalDataset, InferencePipeline, evaluate
from scorebook.inference.clients.bedrock import batch
from scorebook.metrics import Accuracy


def main() -> None:
    """Run the AWS Bedrock batch inference example."""
    # Load environment variables from .env file for configuration
    load_dotenv()

    args = setup_arguments()

    # Step 1: Load the evaluation dataset
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json",
        metrics=[Accuracy],
        input="question",
        label="answer",
    )

    # Ensure minimum batch size requirement (AWS Bedrock requires 100+ items)
    if len(dataset) < 100:
        # Cycle through items to reach 100 - default minimum size of AWS batch jobs
        original_items = dataset.items
        expanded_items: List[Any] = []
        while len(expanded_items) < 100:
            items_needed = 100 - len(expanded_items)
            expanded_items.extend(original_items[: min(len(original_items), items_needed)])

        # Create new dataset with expanded items
        # Items already have "input" and "label" columns from the original dataset
        dataset = EvalDataset.from_list(
            name=dataset.name,
            metrics=[Accuracy],
            items=expanded_items,
            input="input",
            label="label",
        )

    # Step 2: Define the preprocessing function for AWS Bedrock Batch API
    def preprocessor(input_value: str) -> list:
        """Pre-process dataset inputs into AWS Bedrock Batch API format."""
        # Create the batch API request messages format for Bedrock
        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and concisely as a single word",
            },
            {"role": "user", "content": input_value},
        ]

        return messages

    # Step 3: Define the postprocessing function
    def postprocessor(response: str) -> str:
        """Post-process AWS Bedrock batch response to extract the answer."""
        # The batch function returns the message content directly
        return response.strip()

    # Step 4: Create the inference pipeline for batch processing
    async def inference_function(items: list, **hyperparams: Any) -> Any:
        return await batch(
            items,
            model=args.model,
            aws_region=args.aws_region,
            aws_profile=args.aws_profile,
            bucket=args.bucket,
            input_prefix=args.input_prefix,
            output_prefix=args.output_prefix,
            role_arn=args.role_arn,
            **hyperparams,
        )

    inference_pipeline = InferencePipeline(
        model=args.model or "claude-model",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    print("Running AWS Bedrock Batch API evaluation")
    print(f"Model: {args.model or 'Not specified'}")
    print(f"AWS Region: {args.aws_region or 'Not specified'}")
    print(f"AWS Profile: {args.aws_profile or 'Not specified'}")
    print(f"S3 Bucket: {args.bucket or 'Not specified'}")
    print(f"Role ARN: {args.role_arn or 'Not specified'}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    results = evaluate(inference_pipeline, dataset, return_items=True, return_output=True)
    print("\nBatch evaluation completed!")
    print(results)

    # Step 6: Save results to file
    output_file = args.output_dir / "bedrock_batch_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


class Args:
    """Simple container for parsed arguments."""

    def __init__(self) -> None:
        """Parse command line arguments."""
        self.output_dir: Path = Path(".")  # Will be overridden in setup_arguments
        self.model: Optional[str] = None
        self.aws_region: Optional[str] = None
        self.aws_profile: Optional[str] = None
        self.bucket: Optional[str] = None
        self.input_prefix: Optional[str] = None
        self.output_prefix: Optional[str] = None
        self.role_arn: Optional[str] = None


def setup_arguments() -> Args:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AWS Bedrock Batch API evaluation and save results."
    )

    # Required argument
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=str(Path.cwd() / "results"),
        help="Directory to save evaluation outputs (JSON).",
    )

    # All optional AWS parameters
    parser.add_argument(
        "--model",
        type=str,
        help="Bedrock model ID (e.g., 'us.anthropic.claude-3-5-sonnet-20241022-v2:0')",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        help="AWS region for Bedrock and S3 operations",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        help="AWS profile name for authentication",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket name for input/output data",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        help="S3 prefix for input data (e.g., 'batch/input/')",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        help="S3 prefix for output data (e.g., 'batch/output/')",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        help="IAM role ARN for Bedrock execution",
    )

    parsed_args = parser.parse_args()

    # Create Args object and populate
    args = Args()
    args.output_dir = Path(parsed_args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.model = parsed_args.model
    args.aws_region = parsed_args.aws_region
    args.aws_profile = parsed_args.aws_profile
    args.bucket = parsed_args.bucket
    args.input_prefix = parsed_args.input_prefix
    args.output_prefix = parsed_args.output_prefix
    args.role_arn = parsed_args.role_arn

    return args


if __name__ == "__main__":
    main()
