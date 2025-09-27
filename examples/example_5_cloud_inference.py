"""Example 5 - Using Cloud Inference Providers."""

from typing import Any, List

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
    """Run an evaluation using a model hosted in the cloud by an inference provider.

    This example demonstrates how to evaluate cloud hosted large language models
    It showcases integration with OpenAI's API.

    Prerequisites:
        - OpenAI API key set in environment variable OPENAI_API_KEY
    """

    # === Cloud-Based Inference Pipeline Creation ===

    # Define a preprocessor mapping to openAI's message format
    def openai_preprocessor(eval_item: dict, **hyperparameter_config: Any) -> List[dict]:
        """Pre-process dataset items into OpenAI's message format."""
        messages = [
            {
                "role": "system",
                "content": hyperparameter_config.get(
                    "system_message", "You are a helpful assistant."
                ),
            },
            {"role": "user", "content": eval_item["question"]},
        ]
        return messages

    # Define a postprocessor parsing responses and handling exceptions
    def openai_postprocessor(response: Any, **hyperparameter_config: Any) -> str:
        """Post-process OpenAI response to extract the answer."""
        try:
            raw_response = response.choices[0].message.content
        except (KeyError, IndexError, AttributeError):
            raw_response = "Error: Failed to extract response from OpenAI API"

        return str(raw_response.strip())

    # Create an inference pipeline using openai.responses as an inference function
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=openai_preprocessor,
        inference_function=responses,
        postprocessor=openai_postprocessor,
    )

    # === Evaluation With Cloud-Based Inference ===

    dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/basic_questions.json", label="answer", metrics=Accuracy
    )

    results = evaluate(
        inference_pipeline,
        dataset,
        hyperparameters={
            "system_message": (
                "Answer the question directly. Provide only the answer, without context."
            ),
            "temperature": 0.7,
        },
        return_items=True,
        return_output=True,
        parallel=True,  # Enable to run inference and evaluations simultaneously
        upload_results=False,
    )

    print(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_5")
    output_dir = setup_output_directory()
    model = setup_openai_model_parser()
    results_dict = main(model)
    save_results_to_json(results_dict, output_dir, "example_5_output.json")
