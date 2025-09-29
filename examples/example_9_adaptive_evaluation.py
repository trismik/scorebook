"""Example 9 - Using Trismik's Adaptive Evaluation."""

import os
import string
from typing import Any

from dotenv import load_dotenv
from example_helpers import (
    save_results_to_json,
    setup_logging,
    setup_openai_model_parser,
    setup_output_directory,
)

from scorebook import evaluate, login
from scorebook.inference.openai import responses
from scorebook.inference_pipeline import InferencePipeline


def main(model_name: str) -> Any:
    """Run a Trismik adaptive evaluation example.

    This example demonstrates how to use Trismik's adaptive evaluations.

    Firstly, a basic InferencePipline using OpenAI's responses API is created
    Secondly, the adaptive MMLU-Pro dataset is used to evaluate the model

    Prerequisites:
    - Valid Trismik authentication credentials
    - A Trismik project id
    """

    # === Setup InferencePipeline ===

    def preprocessor(eval_item: dict, **hyperparameters: Any) -> str:
        """Pre-process dataset items into OpenAI prompt format."""
        prompt = eval_item["question"]

        if "options" in eval_item:
            prompt += "\nOptions:\n" + "\n".join(
                f"{letter}: {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            )

        # Create a system message with instructions for direct answers
        system_prompt = """
        Answer the question with a single letter
        representing the correct answer from the list of choices.
        Do not provide any additional explanation or output beyond the single letter.
        """.strip()

        # Format as a conversation for OpenAI API
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    def postprocessor(response: Any, **hyperparameters: Any) -> str:
        """Post-process OpenAI response to extract the answer."""
        # Extract the text from the OpenAI response object
        try:
            # Access the first choice's message content (correct OpenAI ChatCompletion format)
            raw_response = response.choices[0].message.content
        except (KeyError, IndexError, AttributeError):
            raw_response = ""

        # Return the response text, stripping whitespace
        return str(raw_response.strip())

    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    # === Run Adaptive Evaluation ===

    # Step 1: Log in with your Trismik API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        raise ValueError("TRISMIK_API_KEY environment variable must be set")
    login(api_key)

    # Step 2: Run evaluation with a Trismik adaptive dataset
    results = evaluate(
        inference_pipeline,
        datasets="MMLUPro2025:adaptive",  # Adaptive datasets have the suffix ":adaptive"
        experiment_id="Scorebook-Example-9-Adaptive-Evaluation",
        project_id="YOUR_PROJECT_ID",
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )

    print(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_9")
    output_dir = setup_output_directory()
    model = setup_openai_model_parser()
    results_dict = main(model)
    save_results_to_json(results_dict, output_dir, "example_9_output.json")
