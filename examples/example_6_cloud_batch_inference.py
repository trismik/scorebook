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

from typing import Any

from dotenv import load_dotenv
from example_helpers import (
    save_results_to_json,
    setup_batch_model_parser,
    setup_logging,
    setup_output_directory,
)

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import batch
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main(model_name: str) -> Any:
    """Run the cloud batch inference example."""

    # Step 1: Load the evaluation dataset
    #    Create an EvalDataset from a local JSON file
    #    Uses 'answer' field as ground truth labels
    #    Configures Accuracy metric for evaluation
    #    Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function for batch API
    #    Convert raw dataset items into OpenAI Batch API-compatible format
    #    The batch API requires a specific JSON structure with chat completions format
    def preprocessor(eval_item: dict, **hyperparameters: Any) -> dict:
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
            "temperature": hyperparameters.get("temperature", 0.7),
        }

        return batch_request

    # Step 3: Define the postprocessing function
    #    Extract the final answer from OpenAI Batch API response
    #    The batch API returns responses in a different format than standard API
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
    #    Uses scorebook's built-in OpenAI batch function for API calls
    #    This pipeline handles the complete batch workflow:
    #        1. Upload a JSONL file to OpenAI
    #        2. Create a batch job
    #        3. Monitor progress
    #        4. Download and parse results
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=batch,  # Use batch instead of responses for batch processing
        postprocessor=postprocessor,
    )

    # Step 5: Run the batch evaluation
    #    Execute evaluation using OpenAI Batch API with the inference pipeline
    #    Uses score_type="all" to get both aggregate and per-item results
    #    Processes all items in dataset (remove item_limit for full evaluation)
    #    Batch processing is asynchronous and cost-effective
    print(f"Running OpenAI Batch API evaluation with model: {model_name}")
    print(f"Processing {len(dataset)} items using batch inference...")
    print("Note: Batch processing may take several minutes to complete.")

    # For demonstration, limit to 25 items to manage cost and time
    # Remove item_limit parameter for full dataset evaluation
    results = evaluate(
        inference_pipeline,
        dataset,
        hyperparameters=[{"temperature": 0.6}, {"temperature": 0.7}, {"temperature": 0.8}],
        parallel=True,
        sample_size=25,
        return_aggregates=True,
        return_items=True,
    )
    print("\nBatch evaluation completed!")
    print(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_6")
    output_dir = setup_output_directory()
    model_name = setup_batch_model_parser()
    results_dict = main(model_name)
    save_results_to_json(results_dict, output_dir, "example_6_output.json")
