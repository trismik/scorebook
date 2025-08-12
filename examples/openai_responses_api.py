"""OpenAI inference example using MMLU-Pro dataset."""

import argparse
import json
import string
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import responses
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def preprocessor(eval_item: dict) -> str:
    """Pre-process MMLU-Pro dataset items into OpenAI prompt format."""
    prompt = f"{eval_item['question']}\nOptions:\n" + "\n".join(
        [
            f"{letter}: {choice}"
            for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
        ]
    )

    # Create a system message with instructions
    system_prompt = """
Answer the question you are given using only a single letter (for example, 'A').
Do not use punctuation.
Do not show your reasoning.
Do not provide any explanation.
Follow the instructions exactly and always answer using a single uppercase letter.

For example, if the question is "What is the capital of France?" and the
choices are "A: Paris", "B: London", "C: Rome", "D: Madrid",
- the answer should be "A"
- the answer should NOT be "Paris" or "A. Paris" or "A: Paris"

Please adhere strictly to the instructions.
""".strip()

    # Format as a conversation for OpenAI
    return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"


def postprocessor(response: Any) -> str:
    """Post-process OpenAI response to extract the answer letter."""
    # Extract the text from the OpenAI response object
    try:
        # Access the first choice's message content
        raw_response = response.output[0].content[0].text
    except (KeyError, IndexError, AttributeError):
        raw_response = ""

    # Extract single letter from response
    # Look for uppercase letters A-Z
    for char in raw_response:
        if char in string.ascii_uppercase:
            return str(char)

    # Fallback: return first character if it's a letter
    if raw_response and raw_response[0].upper() in string.ascii_uppercase:
        return str(raw_response[0].upper())

    # Last resort: return an empty string as default
    return ""


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run OpenAI evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/demo_results"),
        help=(
            "Directory to save evaluation outputs (JSON). "
            "Defaults to ./results in the current working directory."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for inference (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MMLU-Pro dataset
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )

    # Create inference pipeline using the responses function directly
    inference_pipeline = InferencePipeline(
        model=args.model,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    print(f"Running OpenAI evaluation with model: {args.model}")
    print(f"Evaluating {10} items from MMLU-Pro dataset...")

    # Evaluate using OpenAI with inference pipeline
    results = evaluate(inference_pipeline, mmlu_pro, item_limit=10, score_type="all")
    print(results)

    # Save results to a JSON file
    output_file = output_dir / "openai_responses_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")
