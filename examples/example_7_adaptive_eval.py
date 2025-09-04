"""
Trismik Adaptive Evaluation with Cloud Inference.

This example demonstrates how to use Trismik's adaptive evaluations for faster evaluation of models
using cloud-based OpenAI inference. It combines the benefits of adaptive testing with powerful
cloud models.

Prerequisites:
- TRISMIK_API_KEY environment variable must be set
- OPENAI_API_KEY environment variable must be set
- python-dotenv for environment variable management
- Active OpenAI account with API credits
"""

import json
import os
import string
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scorebook import EvalDataset, evaluate, login
from scorebook.inference.openai import responses
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the adaptive evaluation example."""
    output_dir = setup_output_directory()

    load_dotenv()

    # Step 1: Log in with your Trismik API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        raise ValueError("TRISMIK_API_KEY environment variable must be set")
    login(api_key)

    # Step 2: Define preprocessing and postprocessing functions for OpenAI API
    def preprocessor(eval_item: dict, hyperparameters: dict) -> str:
        """Pre-process dataset items into OpenAI prompt format."""
        prompt = eval_item["question"]

        if "options" in eval_item:
            prompt += "\nOptions:\n" + "\n".join(
                f"{letter}: {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            )

        # Create a system message with instructions for direct answers
        system_prompt = """
Answer the question with a single letter representing the correct answer from the list of choices.
Do not provide any additional explanation or output beyond the single letter.
""".strip()

        # Format as a conversation for OpenAI API
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    def postprocessor(response: Any, hyperparameters: dict) -> str:
        """Post-process OpenAI response to extract the answer."""
        # Extract the text from the OpenAI response object
        try:
            # Access the first choice's message content
            raw_response = response.output[0].content[0].text
        except (KeyError, IndexError, AttributeError):
            raw_response = ""

        # Return the response text, stripping whitespace
        return str(raw_response.strip())

    # Step 3: Create the inference pipeline for cloud-based evaluation
    # Combine preprocessing, OpenAI API inference, and postprocessing
    # Uses scorebook's built-in OpenAI responses function for API calls
    model_name = setup_model_selection()
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    print("Loading MMLU-Pro dataset from Hugging Face Hub...")
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )
    print(f"Loaded {len(mmlu_pro)} items from MMLU-Pro dataset")

    # Step 4: Run the adaptive evaluation
    print(f"Running OpenAI adaptive evaluation with model: {model_name}")
    eval_results = evaluate(
        inference_pipeline,
        ["MMLUPro2025:adaptive", mmlu_pro],
        experiment_id="scorebook-example",
        project_id="c48419ff38b70f2b79265312a236b594a616f74c",
        return_dict=True,
        return_items=True,
        return_output=True,
    )

    print(eval_results)

    # Step 4: Save results to file
    # Export evaluation results as JSON for later analysis
    with open(output_dir / "example_7_output.json", "w") as output_file:
        json.dump(eval_results, output_file, indent=4)
        print(f"Results saved in {output_dir / 'example_7_output.json'}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/example_results"),
        help=(
            "Directory to save evaluation outputs (CSV and JSON). "
            "Defaults to ./results in the current working directory."
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
        default="gpt-5",
        help="OpenAI model to use for inference (default: gpt-4o)",
    )
    args = parser.parse_args()
    return str(args.model)


if __name__ == "__main__":
    main()
