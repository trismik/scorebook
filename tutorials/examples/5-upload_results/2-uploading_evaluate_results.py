"""Tutorials - Upload Results - Example 2 - Uploading evaluate() Results."""

from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv

from scorebook.utils.tutorial_utils import save_results_to_json, setup_logging

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
    dataset_path = Path(__file__).parent.parent / "3-evaluation_datasets" / "example_datasets" / "basic_questions.json"
    dataset = EvalDataset.from_json(
        path=str(dataset_path),
        metrics="accuracy",
        input="question",
        label="answer",
    )

    # Step 1: Log in with your Trismik API key
    login("TRISMIK_API_KEY") # TODO: ADD YOUR TRISMIK API KEY


    # Step 2: Run evaluation with result uploading
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
        project_id="TRISMIK_PROJECT_ID", # TODO: ADD YOUR TRISMIK PROJECT ID
        metadata={
            "model": model_name,
            "description": "Example evaluation demonstrating result uploading",
        },
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )

    print("\nResults uploaded successfully!")
    pprint(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="2-uploading_evaluate_results", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "2-uploading_evaluate_results_output.json")