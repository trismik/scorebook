"""
OpenAI inference implementation for Scorebook.

This module provides utilities for running inference using OpenAI's models,
supporting both single response and batch inference operations. It handles
API communication, request formatting, and response processing.
"""

import asyncio
import atexit
import json
import logging
import tempfile
from typing import Any, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Global singleton OpenAI client for resource management
_openai_client: Optional[AsyncOpenAI] = None
_cleanup_registered: bool = False


def _register_openai_cleanup() -> None:
    """Register cleanup handlers for proper OpenAI client shutdown."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_openai_client)
        _cleanup_registered = True


def _cleanup_openai_client() -> None:
    """Clean up OpenAI client instance."""
    global _openai_client
    if _openai_client and hasattr(_openai_client, "close"):
        try:
            # Try to close if event loop is still running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_openai_client.close())
            else:
                asyncio.run(_openai_client.close())
        except Exception as e:
            logger.debug(f"Error closing OpenAI client: {e}")
        finally:
            _openai_client = None


async def cleanup_openai_client() -> None:
    """Perform async cleanup for proper OpenAI client closure."""
    global _openai_client
    if _openai_client and hasattr(_openai_client, "close"):
        try:
            await _openai_client.close()
            logger.debug("Successfully closed OpenAI client")
        except Exception as e:
            logger.debug(f"Error closing OpenAI client: {e}")
        finally:
            _openai_client = None


def get_openai_client() -> AsyncOpenAI:
    """Get singleton OpenAI client with proper resource management."""
    global _openai_client

    # Register cleanup handlers on first use
    _register_openai_cleanup()

    if _openai_client is None:
        logger.debug("Creating new singleton AsyncOpenAI client")
        _openai_client = AsyncOpenAI()

    return _openai_client


async def responses(
    items: List[Any], model: str = "gpt-4.1-nano", client: Any = None, **hyperparameters: Any
) -> List[Any]:
    """Process multiple inference requests using OpenAI's Async API.

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
    logger.debug("OpenAI responses function called with %d items", len(items))
    logger.debug("Using model: %s", model)
    logger.debug("Hyperparameters: %s", hyperparameters)

    if client is None:
        logger.debug("Using singleton AsyncOpenAI client")
        client = get_openai_client()

    # Create all tasks concurrently for true parallelism
    tasks = []
    for i, item in enumerate(items):
        logger.debug(
            "Processing item %d: %s",
            i,
            str(item)[:100] + "..." if len(str(item)) > 100 else str(item),
        )

        # Handle string input from preprocessor - convert to proper messages format
        if isinstance(item, str):
            # Convert the string format to proper OpenAI messages array
            messages = [{"role": "user", "content": item}]
            logger.debug(
                "Converted string to messages format: %s",
                (
                    messages[0]["content"][:100] + "..."
                    if len(messages[0]["content"]) > 100
                    else messages[0]["content"]
                ),
            )
        elif isinstance(item, list):
            # Already in proper messages format
            messages = item
            logger.debug("Item %d already in messages format", i)
        else:
            # Fallback: treat as user message
            messages = [{"role": "user", "content": str(item)}]
            logger.debug("Item %d converted to fallback format", i)

        logger.debug("Creating OpenAI task %d with messages: %s", i, messages)
        # Filter to only include valid OpenAI chat completions parameters
        valid_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "logit_bias",
            "user",
            "seed",
            "tools",
            "tool_choice",
            "response_format",
            "n",
            "logprobs",
            "top_logprobs",
        }
        filtered_hyperparameters = {k: v for k, v in hyperparameters.items() if k in valid_params}
        task = client.chat.completions.create(
            model=model, messages=messages, **filtered_hyperparameters
        )
        tasks.append(task)

    logger.debug("Created %d tasks, waiting for OpenAI responses...", len(tasks))
    # Wait for all requests to complete in parallel
    results = await asyncio.gather(*tasks)
    logger.debug("Received %d responses from OpenAI", len(results))

    for i, result in enumerate(results):
        logger.debug("Response %d type: %s", i, type(result))
        try:
            if hasattr(result, "choices") and result.choices:
                content = result.choices[0].message.content
                logger.debug(
                    "Response %d content: %s",
                    i,
                    content[:100] + "..." if content and len(content) > 100 else content,
                )
            else:
                logger.debug("Response %d has no choices or unexpected format", i)
        except Exception as e:
            logger.error("Error logging response %d: %s", i, e)

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
        client = get_openai_client()

    file_id = await _upload_batch(items, client)
    batch_id = await _start_batch(file_id, client)

    awaiting_batch = True
    while awaiting_batch:
        batch_object = await _get_batch(batch_id, client)
        batch_status = batch_object.status

        if batch_status == "completed":
            awaiting_batch = False
        elif batch_status == "failed":
            raise Exception("Batch processing failed")
        else:
            await asyncio.sleep(60)

    # Get the final batch object to access output_file_id
    final_batch_object = await _get_batch(batch_id, client)
    output_file_id = final_batch_object.output_file_id

    batch_result = await _get_results_file(output_file_id, client)
    return batch_result


async def _upload_batch(items: List[Any], client: Any) -> str:
    """Create a .jsonl file from preprocessed items and upload to OpenAI for batch processing.

    Args:
        items: A list of preprocessed items, each representing a single dataset eval item.

    Returns:
        The file ID returned by OpenAI after uploading.
    """
    # Instantiate OpenAI client
    if client is None:
        client = get_openai_client()

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
    """Download and parse the batch results file from OpenAI."""
    response = await client.files.content(output_file_id)

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
