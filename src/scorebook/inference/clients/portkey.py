"""
Portkey inference implementation for Scorebook.

This module provides utilities for running inference using Portkey's API,
supporting both single response and batch inference operations. It handles
API communication, request formatting, and response processing.
"""

import asyncio
import json
import os
import tempfile
from typing import Any, List, Optional

from portkey_ai import AsyncPortkey
from tqdm.auto import tqdm


async def responses(
    items: List[Any], model: str, client: Optional[AsyncPortkey] = None, **hyperparameters: Any
) -> List[Any]:
    """Process multiple inference requests using Portkey's API.

    This asynchronous function handles multiple inference requests,
    manages the API communication, and processes the responses.

    Args:
        items: List of preprocessed items to process.
        model: Model to use via Portkey.
        client: Optional Portkey client instance.
        hyperparameters: Dictionary of hyperparameters for inference.

    Returns:
        List of raw model responses.
    """

    if client is None:
        client = AsyncPortkey(api_key=os.getenv("PORTKEY_API_KEY"))

    results = []
    for item in items:
        response = await client.chat.completions.create(
            model=model,
            messages=item if isinstance(item, list) else [{"role": "user", "content": str(item)}],
        )
        results.append(response)

    return results


async def batch(
    items: List[Any],
    model: str,
    client: Optional[AsyncPortkey] = None,
    **hyperparameters: Any,
) -> List[Any]:
    """Process multiple inference requests in batch using Portkey's API.

    This asynchronous function handles batch processing of inference requests,
    optimizing for throughput while respecting API rate limits.

    Args:
        items: List of preprocessed items to process.
        model: Model to use via Portkey.
        client: Optional Portkey client instance.
        hyperparameters: Dictionary of hyperparameters for inference.

    Returns:
        A list of raw model responses.
    """

    provider, model = model.split("/")

    if client is None:
        client = AsyncPortkey(provider=provider, api_key=os.getenv("PORTKEY_API_KEY"))

    file_id = await _upload_batch(items, client, model, **hyperparameters)
    batch_id = await _start_batch(file_id, client)

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

    # Use the final batch object to access output_file_id
    output_file_id = batch_object.output_file_id

    batch_result = await _get_results_file(output_file_id, client)
    return batch_result


async def _upload_batch(
    items: List[Any], client: AsyncPortkey, model: str, **hyperparameters: Any
) -> str:
    """Create a .jsonl file from preprocessed items and upload to Portkey for batch processing.

    Args:
        items: A list of preprocessed items, each representing a single dataset eval item.
        client: Portkey client instance.
        model: Model to use for batch processing.
        hyperparameters: Additional parameters for the batch requests.

    Returns:
        The file ID returned by Portkey after uploading.
    """
    print("Uploading batch...")

    # Create temp .jsonl file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i, item in enumerate(items):
            # Construct each batch line
            payload = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": (
                        item if isinstance(item, list) else [{"role": "user", "content": str(item)}]
                    ),
                    **hyperparameters,
                },
            }
            f.write(json.dumps(payload) + "\n")
        file_path = f.name

    # Upload file to Portkey
    with open(file_path, "rb") as upload_file:
        response = await client.files.create(file=upload_file, purpose="batch")

    return str(response.id)


async def _start_batch(file_id: str, client: Any) -> str:
    batch_response = await client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return str(batch_response.id)


async def _get_batch(batch_id: str, client: Any) -> Any:
    batch_object = await client.batches.retrieve(batch_id)
    return batch_object


async def _get_results_file(output_file_id: str, client: Any) -> List[str]:
    """Download and parse the batch results file from Portkey."""
    response = await client.files.content(output_file_id)

    # Parse the JSONL content
    content = response.content.decode("utf-8")
    results = []

    for line in content.strip().split("\n"):
        result_obj = json.loads(line)
        message_content = result_obj["response"]["body"]["choices"][0]["message"]["content"]
        results.append(message_content)

    return results
