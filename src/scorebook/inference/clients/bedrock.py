"""
AWS Bedrock batch inference implementation for Scorebook.

This module provides utilities for running batch inference using AWS Bedrock's
Model Invocation Jobs, supporting large-scale asynchronous processing. It handles
API communication, request formatting, response processing, and S3 operations.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm.auto import tqdm


async def batch(
    items: List[Any],
    model: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
    bucket: Optional[str] = None,
    input_prefix: Optional[str] = None,
    output_prefix: Optional[str] = None,
    role_arn: Optional[str] = None,
    **hyperparameters: Any,
) -> List[Any]:
    """Process multiple inference requests in batch using AWS Bedrock.

    This asynchronous function handles batch processing of inference requests,
    optimizing for cost and throughput using AWS Bedrock's Model Invocation Jobs.

    Args:
        items: List of preprocessed items to process.
        model: Bedrock model ID (e.g., 'us.anthropic.claude-3-5-sonnet-20241022-v2:0').
        aws_region: AWS region for Bedrock and S3.
        aws_profile: AWS profile name for authentication.
        bucket: S3 bucket name for input/output data.
        input_prefix: S3 prefix for input data.
        output_prefix: S3 prefix for output data.
        role_arn: IAM role ARN for Bedrock execution.
        hyperparameters: Additional parameters for the batch requests.

    Returns:
        A list of raw model responses.
    """
    # Set up AWS session and clients
    session_kwargs = {}
    if aws_profile:
        session_kwargs["profile_name"] = aws_profile
    if aws_region:
        session_kwargs["region_name"] = aws_region

    session = boto3.Session(**session_kwargs)

    boto_config = Config(region_name=aws_region, retries={"max_attempts": 10, "mode": "adaptive"})

    s3_client = session.client("s3", config=boto_config)
    bedrock_client = session.client("bedrock", config=boto_config)

    # Upload batch data
    input_uri = await _upload_batch(
        items, s3_client, bucket, input_prefix, model, **hyperparameters
    )

    # Start batch job
    job_arn = await _start_batch_job(
        bedrock_client, model, input_uri, bucket, output_prefix, role_arn
    )

    # Wait for completion with progress tracking
    await _wait_for_completion(bedrock_client, job_arn, len(items))

    # Retrieve results
    results = await _get_batch_results(s3_client, bedrock_client, job_arn)

    return results


async def _upload_batch(
    items: List[Any],
    s3_client: Any,
    bucket: Optional[str],
    input_prefix: Optional[str],
    model: Optional[str],
    **hyperparameters: Any,
) -> str:
    """Create a JSONL file from preprocessed items and upload to S3 for batch processing."""

    # Generate unique run ID and key
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]

    if input_prefix:
        input_key = f"{input_prefix.rstrip('/')}/inputs-{run_id}.jsonl"
    else:
        input_key = f"inputs-{run_id}.jsonl"

    # Create temp JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i, item in enumerate(items):
            # Construct batch request in Bedrock format
            record = {
                "recordId": f"rec-{i:04d}",
                "modelInput": _build_claude_messages_payload(item, **hyperparameters),
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        file_path = f.name

    # Upload to S3
    try:
        body = open(file_path, "rb").read()
        s3_client.put_object(
            Bucket=bucket,
            Key=input_key,
            Body=body,
            StorageClass="INTELLIGENT_TIERING",
            ContentType="application/json",
        )
        input_uri = f"s3://{bucket}/{input_key}"
    except Exception as e:
        raise Exception(f"Failed to upload file to S3: {e}")
    finally:
        # Clean up temp file
        os.unlink(file_path)

    return input_uri


def _build_claude_messages_payload(item: Any, **hyperparameters: Any) -> Dict[str, Any]:
    """Build Claude messages payload for Bedrock batch processing."""

    # item is a list of messages from our preprocessor
    messages = item

    # Convert to Bedrock format and extract system message
    bedrock_messages = []
    system_content = None

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            bedrock_messages.append(
                {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
            )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": bedrock_messages,
    }

    if system_content:
        payload["system"] = system_content

    payload.update(hyperparameters)
    return payload


async def _start_batch_job(
    bedrock_client: Any,
    model: Optional[str],
    input_uri: str,
    bucket: Optional[str],
    output_prefix: Optional[str],
    role_arn: Optional[str],
) -> str:
    """Start a Bedrock Model Invocation Job."""

    # Generate unique job name and output URI
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job_name = f"bedrock-batch-{run_id}"

    if output_prefix:
        output_uri = f"s3://{bucket}/{output_prefix.rstrip('/')}/job-{run_id}/"
    else:
        output_uri = f"s3://{bucket}/job-{run_id}/"

    try:
        response = bedrock_client.create_model_invocation_job(
            jobName=job_name,
            modelId=model,
            roleArn=role_arn,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}},
            tags=[{"key": "project", "value": "scorebook-batch"}],
        )
        job_arn: str = response["jobArn"]
        return job_arn
    except ClientError as e:
        error_info = e.response.get("Error", {})
        raise Exception(f"Failed to create batch job: {error_info}")


async def _wait_for_completion(bedrock_client: Any, job_arn: str, total_items: int) -> None:
    """Wait for batch job completion with progress tracking."""

    # Initialize progress bar
    pbar = tqdm(total=total_items, desc="Batch processing", unit="requests")

    terminal_states = {"Completed", "Failed", "Stopped"}
    sleep_time = 15

    while True:
        try:
            desc = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = desc["status"]

            # Get progress if available
            job_state = desc.get("jobState", {})
            progress = job_state.get("percentComplete")

            # Update progress bar
            if progress is not None:
                completed = int((progress / 100) * total_items)
                pbar.n = completed
                pbar.set_postfix(status=status, progress=f"{progress}%")
            else:
                pbar.set_postfix(status=status)

            pbar.refresh()

            if status in terminal_states:
                if status == "Completed":
                    pbar.n = pbar.total
                    pbar.set_postfix(status="COMPLETED")
                else:
                    pbar.close()
                    error_msg = desc.get("failureMessage", f"Job ended with status {status}")
                    raise Exception(f"Batch job failed: {error_msg}")
                break

            # Wait before checking again
            await asyncio.sleep(sleep_time)

        except Exception as e:
            pbar.close()
            raise e

    pbar.close()


async def _get_batch_results(s3_client: Any, bedrock_client: Any, job_arn: str) -> List[str]:
    """Download and parse batch results from S3."""

    # Get job details to find output location
    desc = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
    output_uri = desc["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]

    bucket_name, prefix = s3_uri_to_bucket_and_prefix(output_uri)

    # Find the output JSONL file
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get("Contents", [])

        # Look for the output JSONL file
        output_key = None
        for obj in contents:
            if obj["Key"].endswith(".jsonl.out"):
                output_key = obj["Key"]
                break

        if not output_key:
            raise Exception("No output JSONL file found in S3")

        # Download and parse results
        obj_response = s3_client.get_object(Bucket=bucket_name, Key=output_key)
        content = obj_response["Body"].read().decode("utf-8")

        results = []
        for line in content.strip().split("\n"):
            if line.strip():
                result_obj = json.loads(line)
                # Extract text from Claude response format
                model_output = result_obj.get("modelOutput", {})
                content_list = model_output.get("content", [])
                if content_list and len(content_list) > 0:
                    text = content_list[0].get("text", "")
                    results.append(text)
                else:
                    results.append("")

        return results

    except Exception as e:
        raise Exception(f"Failed to retrieve batch results: {e}")


def s3_uri_to_bucket_and_prefix(s3_uri: str) -> Tuple[str, str]:
    """Parse S3 URI to bucket and prefix."""
    # Parse S3 URI
    if s3_uri.startswith("s3://"):
        uri_parts = s3_uri[5:].split("/", 1)
        bucket_name = uri_parts[0]
        prefix = uri_parts[1] if len(uri_parts) > 1 else ""
    else:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return bucket_name, prefix
