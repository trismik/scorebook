"""OpenAI inference example using MMLU-Pro dataset."""

import argparse
import json
import string
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import batch
from scorebook.metrics import Accuracy

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run OpenAI evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "results"),
        help=(
            "Directory to save evaluation outputs (JSON). "
            "Defaults to ./results in the current working directory."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for batch inference (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_mmlu_item(eval_item: dict) -> dict:
        """Pre-process MMLU-Pro dataset items into OpenAI batch API format."""
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

        # Return properly formatted OpenAI chat completion request
        return {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

    def postprocess_openai_response(response: Any) -> str:
        """Post-process OpenAI response to extract the answer letter."""
        # Extract the text from the OpenAI response object
        try:
            # Access the first choice's message content
            raw_response = response.output[0].content[0].text
        except (KeyError, IndexError, AttributeError):
            raw_response = ""

        # Extract a single letter from response
        # Look for uppercase letters A-Z
        for char in raw_response:
            if char in string.ascii_uppercase:
                return str(char)

        # Fallback: return the first character if it's a letter
        if raw_response and raw_response[0].upper() in string.ascii_uppercase:
            return str(raw_response[0].upper())

        # Last resort: return an empty string as default
        return ""

    # Load MMLU-Pro dataset
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )

    async def openai_batch_inference_function(eval_items: List, **hyperparameters: Any) -> Any:
        """Async inference function that uses OpenAI API."""
        return await batch(
            items=eval_items,
            pre_processor=preprocess_mmlu_item,
            post_processor=postprocess_openai_response,
            model=args.model,
        )

    print(f"Running OpenAI batch evaluation with model: {args.model}")
    print(f"Evaluating {10} items from MMLU-Pro dataset...")

    # Evaluate using OpenAI with the batch inference function
    results = evaluate(openai_batch_inference_function, mmlu_pro, item_limit=2, score_type="all")
    print(results)

    # Save results to a JSON file
    output_file = output_dir / "openai_batch_responses_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")
