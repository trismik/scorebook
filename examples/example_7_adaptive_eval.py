"""
Trismik Adaptive Evaluation with Cloud Inference.

This example demonstrates how to use Trismik's adaptive evaluations for faster evaluation of models
using cloud-based OpenAI inference. It combines the benefits of adaptive testing with powerful
cloud models.

Key Features Demonstrated:
1. **Trismik Adaptive Evaluation**: Intelligent test selection for efficient evaluation
2. **Cloud Integration**: OpenAI API for powerful model inference
3. **Hybrid Evaluation**: Both adaptive and classic evaluation in the same run
4. **Inference Pipeline**: Modular preprocessing and postprocessing for cloud APIs
5. **MMLU-Pro Dataset**: Large-scale academic benchmark for comprehensive testing

Adaptive Evaluation Benefits:
- **Efficiency**: Fewer evaluations needed to reach statistical significance
- **Cost Savings**: Reduced API calls while maintaining evaluation quality
- **Speed**: Faster turnaround for model evaluation
- **Intelligence**: Adaptive selection of most informative test cases

Prerequisites:
- TRISMIK_API_KEY environment variable must be set
- OPENAI_API_KEY environment variable must be set
- python-dotenv for environment variable management
- Active OpenAI account with API credits
- Active Trismik account with API credits

Compare with cloud inference examples to understand the efficiency gains
of adaptive evaluation approaches.
"""

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

from scorebook import EvalDataset, evaluate, login
from scorebook.inference.openai import responses
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main(model_name: str) -> Any:
    """Run the adaptive evaluation example."""

    # Step 1: Log in with your Trismik API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        raise ValueError("TRISMIK_API_KEY environment variable must be set")
    login(api_key)

    # Step 2: Define preprocessing and postprocessing functions for OpenAI API
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
Answer the question with a single letter representing the correct answer from the list of choices.
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

    # Step 3: Create the inference pipeline for cloud-based evaluation
    # Combine preprocessing, OpenAI API inference, and postprocessing
    # Uses scorebook's built-in OpenAI responses function for API calls
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    # Step 4: Load the evaluation dataset
    print("Loading MMLU-Pro dataset from Hugging Face Hub...")
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )
    print(f"Loaded {len(mmlu_pro)} items from MMLU-Pro dataset")

    # Step 5: Run the adaptive evaluation
    # Execute evaluation using both adaptive and classic approaches
    # Adaptive evaluation uses intelligent test selection for efficiency
    # Classic evaluation provides comprehensive baseline comparison
    print(f"Running OpenAI adaptive evaluation with model: {model_name}")
    print("Running hybrid evaluation: adaptive + classic approaches...")

    results = evaluate(
        inference_pipeline,
        ["MMLUPro2025:adaptive", mmlu_pro],
        experiment_id="scorebook-example",
        project_id="c48419ff38b70f2b79265312a236b594a616f74c",
        sample_size=10,  # Limit for demonstration and cost control
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )

    print(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_7")
    output_dir = setup_output_directory()
    model_name = setup_openai_model_parser()
    results_dict = main(model_name)
    save_results_to_json(results_dict, output_dir, "example_7_output.json")
