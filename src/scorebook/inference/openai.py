"""
OpenAI inference implementation for Scorebook.

This module provides utilities for running inference using OpenAI's models,
supporting both single response and batch inference operations. It handles
API communication, request formatting, and response processing.
"""

import asyncio
import json
import tempfile
from typing import Any, List

from openai import OpenAI
from tqdm.asyncio import tqdm


async def responses(
    items: List[Any], model: str = "gpt-4.1-nano", client: Any = None, **hyperparameters: Any
) -> List[Any]:
    """Process multiple inference requests using OpenAI's API.

    This asynchronous function handles multiple inference requests,
    manages the API communication, and processes the responses.

    Args:
        items: List of preprocessed items to process.
        model: OpenAI model to use.
        client: Optional OpenAI client instance.
        hyperparameters: Dictionary of hyperparameters for inference.

    Returns:
        List of raw model responses.

    Raises:
        NotImplementedError: Currently not implemented.
    """
    if client is None:
        client = OpenAI()

    results = []
    for item in items:
        response = client.responses.create(model=model, input=item)
        results.append(response)

    return results


async def batch(
    items: List[Any],
    model: str = "gpt-4.1-nano",
    client: Any = None,
    **hyperparameters: Any,
) -> List[Any]:
    """Process multiple inference requests in batch using OpenAI's API.

    This asynchronous function handles batch processing of inference requests,
    optimizing for throughput while respecting API rate limits.

    Args:
        items: List of preprocessed items to process.
        model: OpenAI model to use.
        client: Optional OpenAI client instance.
        hyperparameters: Dictionary of hyperparameters for inference.

    Returns:
        A list of raw model responses.

    Raises:
        NotImplementedError: Currently not implemented.
    """
    if client is None:
        client = OpenAI()

    file_id = _upload_batch(items, client)
    batch_id = _start_batch(file_id, client)

    # Initialize progress bar
    pbar = tqdm(total=len(items), desc="Batch processing", unit="requests")

    awaiting_batch = True
    while awaiting_batch:
        batch_object = await _get_batch(batch_id, client)
        batch_status = batch_object.status

        if hasattr(batch_object, "request_counts") and batch_object.request_counts:
            completed = batch_object.request_counts.completed
            total = batch_object.request_counts.total
            pbar.n = completed
            pbar.set_postfix(status=batch_status, completed=f"{completed}/{total}")
        else:
            pbar.set_postfix(status=batch_status)

        pbar.refresh()

        if batch_status == "completed":
            awaiting_batch = False
            pbar.n = pbar.total
            pbar.set_postfix(status="completed")
        elif batch_status == "failed":
            raise Exception("Batch processing failed")
        else:
            await asyncio.sleep(60)

    pbar.close()

    # Get the final batch object to access output_file_id
    final_batch_object = await _get_batch(batch_id, client)
    output_file_id = final_batch_object.output_file_id

    batch_result = await _get_results_file(output_file_id, client)
    return batch_result


def _upload_batch(items: List[Any], client: Any) -> str:
    """Create a .jsonl file from preprocessed items and upload to OpenAI for batch processing.

    Args:
        items: A list of preprocessed items, each representing a single dataset eval item.

    Returns:
        The file ID returned by OpenAI after uploading.
    """
    print("Uploading batch...")
    # Instantiate OpenAI client
    if client is None:
        client = OpenAI()

    # Create temp .jsonl file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        for i, item in enumerate(items):
            # Construct each batch line
            payload = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": item,
            }
            f.write(json.dumps(payload) + "\n")
        file_path = f.name

    # Upload file to OpenAI
    with open(file_path, "rb") as upload_file:
        response = client.files.create(file=upload_file, purpose="batch")

    return str(response.id)


def _start_batch(file_id: str, client: Any) -> str:
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return str(batch_response.id)


async def _get_batch(batch_id: str, client: Any) -> Any:
    batch_object = client.batches.retrieve(batch_id)
    return batch_object


async def _get_results_file(output_file_id: str, client: Any) -> List[str]:
    """Download and parse the batch results file from OpenAI."""
    response = client.files.content(output_file_id)

    # Parse the JSONL content
    content = response.content.decode("utf-8")
    results = []

    for line in content.strip().split("\n"):
        if line.strip():
            result_obj = json.loads(line)
            # Extract the response from the batch result structure
            if "response" in result_obj and "body" in result_obj["response"]:
                response_body = result_obj["response"]["body"]
                if "choices" in response_body and len(response_body["choices"]) > 0:
                    message_content = response_body["choices"][0]["message"]["content"]
                    results.append(message_content)
                else:
                    results.append("")
            else:
                results.append("")

    return results
