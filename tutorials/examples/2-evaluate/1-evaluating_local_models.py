"""Tutorials - Evaluate - Example 1 - Evaluating Local Models."""

from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import EvalDataset, evaluate


def main() -> Any:
    """Run a simple Scorebook evaluation on a local model.

    This example demonstrates the fundamental workflow for evaluating a model using Scorebook.

    It shows how to:
        1. Create an evaluation dataset from a list of evaluation items
        2. Define an inference function using Hugging Face's transformers library
        3. Run the evaluation and collect results

    This serves as a starting point for understanding Scorebook's core evaluation capabilities.
    """

    # Create a list of evaluation items
    evaluation_items = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    # Create an evaluation dataset
    evaluation_dataset = EvalDataset.from_list(
        name="basic_questions",    # Dataset name
        metrics="accuracy",        # Metric/Metrics used to calculate scores
        items=evaluation_items,    # List of evaluation items
        input="question",          # Key for the input field in evaluation items
        label="answer",            # Key for the label field in evaluation items
    )

    # Create a model
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Define an inference function
    def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Return a list of model outputs for a list of inputs.

        Args:
            inputs: Input values from an EvalDataset.
            hyperparameters: Model hyperparameters.

        Returns:
            The model outputs for a list of inputs.
        """
        inference_outputs = []
        for model_input in inputs:

            # Wrap inputs in the model's message format
            messages = [
                {
                    "role": "system",
                    "content": hyperparameters.get("system_message"),
                },
                {"role": "user", "content": model_input},
            ]

            # Run inference on the item
            output = pipeline(messages, temperature=hyperparameters.get("temperature"))

            # Extract and collect the output generated from the model's response
            inference_outputs.append(output[0]["generated_text"][-1]["content"])

        return inference_outputs

    # Evaluate a model against an evaluation dataset
    results = evaluate(
        inference,
        evaluation_dataset,
        hyperparameters={
            "temperature": 0.7,
            "system_message": "Answer the question directly and concisely.",
        },
        return_items=True,
        upload_results=False,  # Disable uploading for this example
    )

    print("\nEvaluation Results:")
    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="1-evaluating_local_models", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "1-evaluating_local_models_output.json")
