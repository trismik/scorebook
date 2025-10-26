"""Tutorials - Upload Results - Example 2 - Uploading evaluate() Results."""

import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import EvalDataset, evaluate, login


def main() -> Any:
    """Run an evaluation and upload results to Trismik's dashboard.

    This example demonstrates how to upload evaluate() results to Trismik.
    The evaluate() function runs inference on a dataset and automatically
    uploads the results when you provide experiment_id and project_id.

    Use evaluate() when you want to:
        - Run inference AND score in one step
        - Track full evaluation runs with hyperparameters
        - Compare different models on the same dataset

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
    """

    # Initialize HuggingFace model pipeline
    model_name = "microsoft/Phi-4-mini-instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Define an inference function
    def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through the model.

        Args:
            inputs: Input values from an EvalDataset.
            hyperparameters: Model hyperparameters.

        Returns:
            List of model outputs for all inputs.
        """
        outputs = []
        for input_val in inputs:
            # Build messages
            messages = [
                {"role": "system", "content": hyperparameters["system_message"]},
                {"role": "user", "content": str(input_val)},
            ]

            # Run inference
            result = pipeline(messages)

            # Extract the answer
            output = str(result[0]["generated_text"][-1]["content"])
            outputs.append(output)

        return outputs

    # Load evaluation dataset
    dataset = EvalDataset.from_json(
        path="../../2-evaluate/example_datasets/basic_questions.json",
        metrics="accuracy",
        input="question",
        label="answer",
    )

    # Step 1: Log in with your Trismik API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    if not api_key:
        raise ValueError(
            "TRISMIK_API_KEY environment variable must be set. "
            "Get your API key from https://app.trismik.com/settings"
        )
    login(api_key)

    # Step 2: Get project ID from environment
    project_id = os.environ.get("TRISMIK_PROJECT_ID")
    if not project_id:
        raise ValueError(
            "TRISMIK_PROJECT_ID environment variable must be set. "
            "Find your project ID at https://app.trismik.com"
        )

    # Step 3: Run evaluation with result uploading
    # When you provide experiment_id and project_id, results are automatically uploaded
    print(f"\nRunning evaluation with model: {model_name}")
    print("Results will be uploaded to Trismik dashboard.\n")

    results = evaluate(
        inference,
        dataset,
        hyperparameters={
            "system_message": "Answer the question directly and concisely.",
        },
        experiment_id="Uploading-Results-Example",  # Creates/uses this experiment
        project_id=project_id,
        metadata={
            "model": model_name,
            "description": "Example evaluation demonstrating result uploading",
        },
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )

    print("\nResults uploaded successfully!")
    print(f"View your results at: https://app.trismik.com/projects/{project_id}\n")

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="2-uploading_evaluate_results")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "2-uploading_evaluate_results_output.json")