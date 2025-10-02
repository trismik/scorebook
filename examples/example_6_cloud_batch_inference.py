"""Example 6 - Using Cloud Inference Providers with a Batch API."""

import asyncio
from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from example_helpers import (
    save_results_to_json,
    setup_batch_model_parser,
    setup_logging,
    setup_output_directory,
)

from scorebook import EvalDataset, evaluate_async
from scorebook.inference.openai import batch
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


async def main(model_name: str) -> Any:
    """Run the cloud batch inference example.

    This example demonstrates how to leverage OpenAI's Batch API for cost-effective,
    large-scale model evaluation within Scorebook.

    Prerequisites:
    - OpenAI API key set in environment variable OPENAI_API_KEY
    """

    # === Create Batch Inference Pipeline ===

    # Define a preprocessor function
    def preprocessor(eval_item: dict, **hyperparameters: Any) -> dict:
        """Pre-process dataset items into OpenAI's Batch API format."""
        prompt = eval_item["question"]

        # Create the batch API request body format
        # This matches the structure expected by OpenAI's /v1/chat/completions endpoint
        batch_request = {
            "model": "gpt-4o-mini",  # Batch API compatible model
            "messages": [
                {"role": "system", "content": hyperparameters.get("system_message")},
                {"role": "user", "content": prompt},
            ],
            "temperature": hyperparameters.get("temperature", 0.7),
        }

        return batch_request

    # Define a postprocessor function
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

    # Create an InferencePipeline using the openai.batch function for batch processing
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=batch,
        postprocessor=postprocessor,
    )

    # === Evaluate With Batched Inference ===

    dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/basic_questions.json", label="answer", metrics=Accuracy
    )

    print(f"\nRunning OpenAI Batch API evaluation with model: {model_name}")
    print("Note: Batch processing may take several minutes to complete.\n")

    results = await evaluate_async(
        inference_pipeline,
        dataset,
        hyperparameters={
            "temperature": 0.7,
            "system_message": "Answer the question directly and concisely",
        },
        return_aggregates=True,
        return_items=True,
        return_output=True,
        upload_results=False,
    )
    print("\nBatch evaluation completed:\n")

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_6")
    output_dir = setup_output_directory()
    model = setup_batch_model_parser()
    results_dict = asyncio.run(main(model))
    save_results_to_json(results_dict, output_dir, "example_6_output.json")
