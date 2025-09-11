"""
Submitting Evaluation Results Example.

This example demonstrates how to automatically submit evaluation results to Trismik
using Scorebook's auto upload functionality. When upload_results="auto" and both
experiment_id and project_id are provided, results are automatically uploaded.

Key Features Demonstrated:
1. **Auto Upload**: Automatic result submission when experiment/project IDs are provided
2. **Experiment Tracking**: Organize evaluations within experiments and projects
3. **Metadata Inclusion**: Add custom metadata to evaluation runs

Prerequisites:
- Valid Trismik authentication (use: trismik login)
- Active Trismik project and experiment IDs

The auto upload logic:
- If upload_results="auto" and no experiment_id/project_id � upload_results becomes False
- If upload_results="auto" and both experiment_id/project_id provided � upload_results becomes True
"""

from typing import Any

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run the submitting eval results example."""

    # Step 1: Load the evaluation dataset
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Initialize the language model
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Step 3: Define the inference function
    def inference_function(eval_items: list[dict], **hyperparameters: Any) -> list[Any]:
        """Pre-processes dataset items, runs inference and post-processes results."""
        results = []
        for eval_item in eval_items:
            prompt = eval_item["question"]

            messages = [
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely.",
                },
                {"role": "user", "content": prompt},
            ]

            output = pipeline(messages)
            response = output[0]["generated_text"][-1]["content"]
            results.append(response)

        return results

    # Step 4: Run evaluation with auto upload
    # Replace with your actual Trismik experiment and project IDs
    results = evaluate(
        inference_function,
        dataset,
        sample_size=10,
        experiment_id="scorebook-example-7",
        project_id="c48419ff38b70f2b79265312a236b594a616f74c",
        upload_results="auto",  # Will automatically upload since IDs are provided
        metadata={
            "model_name": "microsoft/Phi-4-mini-instruct",
            "evaluation_type": "auto_upload_example",
        },
    )

    print("Evaluation completed!")
    # print(f"Accuracy: {results.aggregate_scores}")
    print("Results automatically uploaded to Trismik (if authentication configured)")
    # print(f"Trismik URL: https://app.trismik.com/projects/ID/experiments/ID")

    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_7")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_7_output.json")
