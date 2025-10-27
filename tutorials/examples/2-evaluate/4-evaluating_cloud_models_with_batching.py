"""Tutorials - Evaluate - Example 4 - Evaluating Cloud Models with Batching."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import EvalDataset, evaluate_async


async def main() -> Any:
    """Run evaluation using OpenAI's Batch API.

    This example demonstrates how to use OpenAI's Batch API for cost-effective,
    large-scale model evaluation. The Batch API offers 50% cost savings compared
    to standard API calls, with results typically delivered within 24 hours.

    Prerequisites:
        - OpenAI API key set in environment variable OPENAI_API_KEY
    """

    # Initialize OpenAI client
    client = AsyncOpenAI()
    model_name = "gpt-4o-mini"

    # Define an async batch inference function
    async def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through OpenAI's Batch API.

        Args:
            inputs: Input values from an EvalDataset.
            hyperparameters: Model hyperparameters including system_message and temperature.

        Returns:
            List of model outputs for all inputs.
        """
        # Step 1: Create batch requests in JSONL format
        batch_requests = []
        for idx, input_val in enumerate(inputs):
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": hyperparameters.get(
                                "system_message", "You are a helpful assistant."
                            ),
                        },
                        {"role": "user", "content": str(input_val)},
                    ],
                    "temperature": hyperparameters.get("temperature", 0.7),
                },
            }
            batch_requests.append(request)

        # Step 2: Write requests to a temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
            temp_file_path = f.name

        try:
            # Step 3: Upload the batch file
            print(f"Uploading batch file with {len(inputs)} requests...")
            with open(temp_file_path, "rb") as f:
                batch_file = await client.files.create(file=f, purpose="batch")

            # Step 4: Create the batch job
            print(f"Creating batch job...")
            batch_job = await client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            # Step 5: Wait for batch completion (with polling)
            print(f"Waiting for batch to complete (ID: {batch_job.id})...")
            while batch_job.status not in ["completed", "failed", "cancelled"]:
                await asyncio.sleep(10)  # Poll every 10 seconds
                batch_job = await client.batches.retrieve(batch_job.id)
                print(f"Status: {batch_job.status}")

            if batch_job.status != "completed":
                raise Exception(f"Batch job failed with status: {batch_job.status}")

            # Step 6: Download and parse results
            print("Batch completed! Downloading results...")
            result_file_id = batch_job.output_file_id
            result_content = await client.files.content(result_file_id)
            result_text = result_content.text

            # Step 7: Parse results and extract outputs
            results_by_id = {}
            for line in result_text.strip().split("\n"):
                result = json.loads(line)
                custom_id = result["custom_id"]
                try:
                    output = result["response"]["body"]["choices"][0]["message"]["content"]
                    results_by_id[custom_id] = output.strip()
                except (KeyError, IndexError):
                    results_by_id[custom_id] = "Error: Failed to extract response"

            # Step 8: Return outputs in original order
            outputs = []
            for idx in range(len(inputs)):
                custom_id = f"request-{idx}"
                outputs.append(results_by_id.get(custom_id, "Error: Missing response"))

            return outputs

        finally:
            # Clean up the temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    # Create a list of evaluation items
    evaluation_items = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    # Create an evaluation dataset
    evaluation_dataset = EvalDataset.from_list(
        name="basic_questions",
        metrics="accuracy",
        items=evaluation_items,
        input="question",
        label="answer",
    )

    print(f"\nRunning OpenAI Batch API evaluation with model: {model_name}")
    print("Note: Batch processing may take several minutes to complete.\n")

    # Run evaluation
    results = await evaluate_async(
        inference,
        evaluation_dataset,
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
    log_file = setup_logging(experiment_id="4-evaluating_cloud_models_with_batching", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main())
    save_results_to_json(results_dict, output_dir, "4-evaluating_cloud_models_with_batching_output.json")
