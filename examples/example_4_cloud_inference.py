"""
Cloud Model Inference Example.

This example demonstrates how to evaluate language models using cloud-based inference
services with Scorebook. It showcases integration with OpenAI's API for large-scale
model evaluation without requiring local model hosting.

Key Features Demonstrated:
1. **Cloud Integration**: Use OpenAI's API for model inference
2. **Flexible Model Selection**: Support for different OpenAI models (GPT-4, GPT-3.5, etc.)
3. **Inference Pipeline**: Modular preprocessing and postprocessing for cloud APIs
4. **Cost-Effective Evaluation**: Leverage powerful cloud models without local resources
5. **Environment Configuration**: Secure API key management using environment variables

Cloud Inference Benefits:
- **No Local Resources**: No need for powerful GPUs or model downloads
- **Latest Models**: Access to state-of-the-art models like GPT-5
- **Scalability**: Handle large evaluations without memory constraints
- **Consistency**: Reproducible results across different environments
- **Cost Control**: Pay-per-use pricing model

Prerequisites:
- OpenAI API key set in environment variable OPENAI_API_KEY
- python-dotenv for environment variable management
- Active OpenAI account with API credits

Compare with local model examples to understand the tradeoffs between
cloud and local inference approaches.
"""

import json
import string
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import responses
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the cloud inference example."""
    # Load environment variables from .env file for API keys
    load_dotenv()

    output_dir = setup_output_directory()

    # Step 1: Load the evaluation dataset
    # Create an EvalDataset from Hugging Face Hub using the MMLU-Pro benchmark
    # - Uses 'answer' field as ground truth labels
    # - Configures Accuracy metric for evaluation
    # - Uses validation split for evaluation
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )

    # Step 2: Define the preprocessing function
    # Convert raw dataset items into OpenAI API-compatible format
    # This function formats the question and options for the cloud model
    def preprocessor(eval_item: dict) -> str:
        """Pre-process MMLU-Pro dataset items into OpenAI prompt format."""
        prompt = f"{eval_item['question']}\nOptions:\n" + "\n".join(
            [
                f"{letter}: {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            ]
        )

        # Create a system message with strict instructions for single-letter responses
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

        # Format as a conversation for OpenAI API
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    # Step 3: Define the postprocessing function
    # Extract the final answer letter from OpenAI API response
    # Handles response parsing and extracts single letter answers
    def postprocessor(response: Any) -> str:
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

    # Step 4: Create the inference pipeline for cloud-based evaluation
    # Combine preprocessing, OpenAI API inference, and postprocessing
    # Uses scorebook's built-in OpenAI responses function for API calls
    model_name = setup_model_selection()
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    # Step 5: Run the cloud-based evaluation
    # Execute evaluation using OpenAI API with the inference pipeline
    # - Uses score_type="all" to get both aggregate and per-item results
    # - Limits to 10 items for quick demonstration and cost control
    print(f"Running OpenAI evaluation with model: {model_name}")
    print("Evaluating 10 items from MMLU-Pro dataset...")

    results = evaluate(inference_pipeline, mmlu_pro, item_limit=10, score_type="all")
    print(results)

    # Step 6: Save results to file
    # Export evaluation results as JSON for later analysis
    output_file = output_dir / "example_4_output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved in {output_file}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenAI evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/example_results"),
        help=(
            "Directory to save evaluation outputs (JSON). "
            "Defaults to ./examples/example_results in the current working directory."
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_model_selection() -> str:
    """Parse model selection from command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Select OpenAI model for evaluation.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for inference (default: gpt-4o-mini)",
    )
    args = parser.parse_args()
    return str(args.model)


if __name__ == "__main__":
    main()
