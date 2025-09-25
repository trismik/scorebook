"""Example 9 - Using Trismik's Adaptive Evaluation."""

import os
import string
from typing import Any, Dict, List

import transformers
from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import evaluate, login
from scorebook.inference_pipeline import InferencePipeline


def main() -> Any:
    """Run a Trismik adaptive evaluation example.

    This example demonstrates how to use Trismik's adaptive evaluations.

    Firstly, a basic InferencePipeline using phi-4 is created
    Secondly, the adaptive MMLU-Pro dataset is used to evaluate the model

    Prerequisites:
    - Valid Trismik authentication credentials
    - A Trismik project id
    """

    # === Setup InferencePipeline ===

    def preprocessor(eval_item: Dict, **hyperparameters: Any) -> List[Any]:
        """Convert an evaluation item to a valid model input."""
        prompt = eval_item["question"]

        if "options" in eval_item:
            prompt += "\nOptions:\n" + "\n".join(
                f"{letter}: {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            )

        # Create a system message with instructions for direct answers
        system_prompt = """Answer the question with a single letter
        representing the correct answer from the list of choices.
        Do not provide any additional explanation or output beyond the single letter."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return messages

    # Setup transformers pipeline for phi-4
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(preprocessed_items: List[Any], **hyperparameters: Any) -> List[Any]:
        """Run model inference on preprocessed eval items."""
        return [
            pipeline(model_input, temperature=hyperparameters.get("temperature", 0.7))
            for model_input in preprocessed_items
        ]

    def postprocessor(model_output: Any, **hyperparameters: Any) -> str:
        """Extract the final parsed answer from the model output."""
        return str(model_output[0]["generated_text"][-1]["content"])

    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference,
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
        project_id="1b25c494472209e23f20a3cfcc9da9c60000fb8e",
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
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_9_output.json")
