"""
Google Cloud Vertex AI batch inference implementation for Scorebook.

This module provides utilities for running batch inference using Google Cloud
Vertex AI Gemini models, supporting large-scale asynchronous processing. It handles
API communication, request formatting, response processing, and Cloud Storage operations.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import fsspec
import pandas as pd
from google import genai
from google.cloud import storage
from google.genai import types
from tqdm.asyncio import tqdm


async def responses(
    items: List[
        Union[
            str,
            List[str],
            types.Content,
            List[types.Content],
            types.FunctionCall,
            List[types.FunctionCall],
            types.Part,
            List[types.Part],
        ]
    ],
    model: str,
    client: Optional[genai.Client] = None,
    project_id: Optional[str] = None,
    location: str = "us-central1",
    system_instruction: Optional[str] = None,
    **hyperparameters: Any,
) -> List[types.GenerateContentResponse]:
    """Process multiple inference requests using Google Cloud Vertex AI.

    This asynchronous function handles multiple inference requests,
    manages the API communication, and processes the responses.

    Args:
        items: List of preprocessed items to process.
        model: Gemini model ID to use (e.g., 'gemini-2.0-flash-001').
        client: Optional Vertex AI client instance.
        project_id: Google Cloud Project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
        location: Google Cloud region (default: 'us-central1').
        system_instruction: Optional system instruction to guide model behavior.
        hyperparameters: Additional parameters for the requests.

    Returns:
        List of raw model responses.
    """
    if client is None:
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError(
                    "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT "
                    "environment variable"
                )

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(api_version="v1"),
        )

    # Create config if system_instruction or hyperparameters are provided
    config = None
    if system_instruction or hyperparameters:
        config_dict = {}
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        if hyperparameters:
            config_dict.update(hyperparameters)
        config = types.GenerateContentConfig(**config_dict)

    results = []
    for item in items:
        response = client.models.generate_content(
            model=model,
            contents=item,
            config=config,
        )
        results.append(response)

    return results


async def batch(
    items: List[Any],
    model: str,
    project_id: Optional[str] = None,
    location: str = "us-central1",
    input_bucket: Optional[str] = None,
    output_bucket: Optional[str] = None,
    **hyperparameters: Any,
) -> List[Any]:
    """Process multiple inference requests in batch using Google Cloud Vertex AI.

    This asynchronous function handles batch processing of inference requests,
    optimizing for cost and throughput using Google Cloud's batch prediction API.

    Args:
        items: List of preprocessed items to process.
        model: Gemini model ID to use (e.g., 'gemini-2.0-flash-001').
        project_id: Google Cloud Project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
        location: Google Cloud region (default: 'us-central1').
        input_bucket: GCS bucket for input data (required).
        output_bucket: GCS bucket for output data (required).
        hyperparameters: Additional parameters for the batch requests.

    Returns:
        A list of raw model responses.
    """
    # Set up project ID
    if project_id is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError(
                "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT " "environment variable"
            )

    if not input_bucket or not output_bucket:
        raise ValueError("Both input_bucket and output_bucket must be provided")

    # Initialize client
    client = genai.Client(vertexai=True, project=project_id, location=location)

    # Upload batch data
    input_uri = await _upload_batch(items, input_bucket, model, project_id, **hyperparameters)

    # Start batch job
    batch_job = await _start_batch_job(client, model, input_uri, output_bucket)

    # Wait for completion with progress tracking
    await _wait_for_completion(client, batch_job, len(items))

    # Retrieve results
    results = await _get_batch_results(batch_job)

    return results


async def _upload_batch(
    items: List[Any], input_bucket: str, model: str, project_id: str, **hyperparameters: Any
) -> str:
    """Create a JSONL file from preprocessed items and upload to GCS for batch processing."""

    # Create temp JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in items:
            # Construct batch request in Vertex AI format
            request_data: Dict[str, Any] = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        str(item)
                                        if not isinstance(item, list)
                                        else item[0]["content"]
                                    )
                                }
                            ],
                        }
                    ]
                }
            }

            # Only add generationConfig if hyperparameters are provided
            if hyperparameters:
                request_data["request"]["generationConfig"] = hyperparameters
            f.write(json.dumps(request_data) + "\n")
        file_path = f.name

    # Upload to GCS using Python client
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Parse bucket name and path from input_bucket
    if input_bucket.startswith("gs://"):
        bucket_path = input_bucket[5:]  # Remove 'gs://' prefix
    else:
        bucket_path = input_bucket

    # Split bucket name and path
    bucket_name = bucket_path.split("/")[0]
    bucket_prefix = "/".join(bucket_path.split("/")[1:]) if "/" in bucket_path else ""

    # Create blob path
    blob_name = (
        f"{bucket_prefix}/batch_input_{timestamp}.jsonl"
        if bucket_prefix
        else f"batch_input_{timestamp}.jsonl"
    )

    # Upload using Google Cloud Storage client
    try:
        gcs_client = storage.Client(project=project_id)
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with open(file_path, "rb") as f:
            blob.upload_from_file(f)

        input_uri = f"gs://{bucket_name}/{blob_name}"

    except Exception as e:
        raise Exception(f"Failed to upload file to GCS: {e}")
    finally:
        # Clean up temp file
        os.unlink(file_path)

    return input_uri


async def _start_batch_job(
    client: genai.Client, model: str, input_uri: str, output_bucket: str
) -> Any:
    """Start a batch prediction job."""
    batch_job = client.batches.create(
        model=model,
        src=input_uri,
        config=types.CreateBatchJobConfig(dest=output_bucket),
    )
    return batch_job


async def _wait_for_completion(client: genai.Client, batch_job: Any, total_items: int) -> None:
    """Wait for batch job completion with progress tracking."""
    # Initialize progress bar
    pbar = tqdm(total=total_items, desc="Batch processing", unit="requests")

    while True:
        # Refresh job status
        batch_job = client.batches.get(name=batch_job.name)
        state = batch_job.state

        # Update progress bar
        pbar.set_postfix(status=str(state).replace("JobState.JOB_STATE_", ""))
        pbar.refresh()

        if state.name == "JOB_STATE_SUCCEEDED":
            pbar.n = pbar.total
            pbar.set_postfix(status="COMPLETED")
            break
        elif state.name == "JOB_STATE_FAILED":
            pbar.close()
            error_msg = getattr(batch_job, "error", "Unknown error")
            raise Exception(f"Batch job failed: {error_msg}")
        elif state.name in ["JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
            pbar.close()
            raise Exception(f"Batch job was {state.name}")

        # Wait before checking again
        await asyncio.sleep(30)

    pbar.close()


async def _get_batch_results(batch_job: Any) -> List[str]:
    """Download and parse batch results from GCS."""

    # Set up GCS filesystem
    fs = fsspec.filesystem("gcs")

    # Find predictions file - the pattern is: dest_uri/prediction-model-*/predictions.jsonl
    output_uri = batch_job.dest.gcs_uri.rstrip("/")
    file_paths = fs.glob(f"{output_uri}/prediction-model-*/predictions.jsonl")

    if not file_paths:
        raise Exception("No predictions file found in output bucket")

    # Load and parse results
    df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

    results = []
    for _, row in df.iterrows():
        # Extract text content from successful responses
        response = row["response"]
        text_content = response["candidates"][0]["content"]["parts"][0]["text"]
        results.append(text_content)

    return results
