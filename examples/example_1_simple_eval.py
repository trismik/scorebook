"""Example 1 - A Simple Scorebook Evaluation."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a simple Scorebook evaluation.

    This example demonstrates the fundamental workflow for evaluating a model using Scorebook.

    It shows how to:
        1. Define an inference function using Hugging Face's transformers library
        2. Load an evaluation dataset from a local JSON file
        3. Run the evaluation and collect results

    This serves as a starting point for understanding Scorebook's core evaluation capabilities.
    """

    # Step 1: Define an inference function

    # For this example, we use Hugging Face's transformer library with Microsoft's Phi-4-mini
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Inference function can be defined with the following signature:
    def inference(eval_items: List[Dict], **hyperparameter_config: Any) -> List[Any]:
        """Return a list of model outputs for a list of evaluation items.

        Args:
            eval_items: Evaluation items from an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            The model outputs for a list of evaluation items.
        """
        inference_results = []
        for eval_item in eval_items:

            # Prepare eval items into valid model input
            messages = [
                {
                    "role": "system",
                    "content": hyperparameter_config["system_message"],
                },
                {"role": "user", "content": eval_item["question"]},
            ]

            # Run inference on the item
            output = pipeline(messages, temperature=hyperparameter_config["temperature"])

            # Extract and collect the output generated from the model's response
            inference_results.append(output[0]["generated_text"][-1]["content"])

        return inference_results

    # Step 2: Load an evaluation dataset, defining a label field and metric for scoring
    eval_dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/dataset.json", label="answer", metrics=Accuracy
    )

    # Step 3: Run the evaluation using the inference function and dataset
    results = evaluate(
        inference,
        eval_dataset,
        hyperparameters={  # Optionally pass a hyperparameter configuration
            "temperature": 0.7,
            "system_message": "Answer the question directly and concisely.",
        },
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_1")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_1_output.json")
