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

from typing import Any

from dotenv import load_dotenv
from example_helpers import (
    save_results_to_json,
    setup_logging,
    setup_openai_model_parser,
    setup_output_directory,
)

from scorebook import EvalDataset, evaluate
from scorebook.inference.openai import responses
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main(model_name: str) -> Any:
    """Run the cloud inference example."""

    # Step 1: Load the evaluation dataset
    #    Create an EvalDataset from local JSON file
    #    Uses 'answer' field as ground truth labels
    #    Configures Accuracy metric for evaluation
    #    Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Define the preprocessing function
    #    Convert raw dataset items into OpenAI API-compatible format
    #    This function formats the question for the cloud model
    def preprocessor(eval_item: dict, **hyperparameters: Any) -> str:
        """Pre-process dataset items into OpenAI prompt format."""
        prompt = eval_item["question"]

        # Create a system message with instructions for direct answers
        system_prompt = """
Answer the question directly and concisely.
Do not provide lengthy explanations unless specifically asked.
""".strip()

        # Format as a conversation for OpenAI API
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    # Step 3: Define the postprocessing function
    #    Extract the final answer from OpenAI API response
    #    Handles response parsing and returns the response text
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

    # Step 4: Create the inference pipeline for cloud-based evaluation
    #    Combine preprocessing, OpenAI API inference, and postprocessing
    #    Uses scorebook's built-in OpenAI responses function for API calls
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=responses,
        postprocessor=postprocessor,
    )

    # Step 5: Run the cloud-based evaluation
    #    Execute evaluation using OpenAI API with the inference pipeline
    #    Uses score_type="all" to get both aggregate and per-item results
    #    Limits to 10 items for quick demonstration and cost control
    print(f"Running OpenAI evaluation with model: {model_name}")
    print("Evaluating 10 items from local dataset...")

    results = evaluate(
        inference_pipeline,
        dataset,
        hyperparameters=[{"temperature": 0.6}, {"temperature": 0.7}, {"temperature": 0.8}],
        parallel=True,
        sample_size=10,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )
    print(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_5")
    output_dir = setup_output_directory()
    model_name = setup_openai_model_parser()
    results_dict = main(model_name)
    save_results_to_json(results_dict, output_dir, "example_5_output.json")
