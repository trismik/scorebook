"""Example 6 - Uploading Evaluation Results to Trismik."""

import os
from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate, login
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a scorebook evaluation and submit the results to Trismik.

    This example demonstrates how to submit evaluation results to your Trismik dashboard.

    Prerequisites:
    - Valid Trismik authentication credentials
    - A Trismik project id
    """

    # === Inference Function Setup ===

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(eval_items: List[Dict], **hyperparameter_config: Any) -> List[Any]:
        """Pre-processes dataset items, runs inference and post-processes results."""
        results = []
        for eval_item in eval_items:
            messages = [
                {"role": "system", "content": "Answer the question directly and concisely."},
                {"role": "user", "content": eval_item["question"]},
            ]
            results.append(pipeline(messages)[0]["generated_text"][-1]["content"])
        return results

    # === Evaluation With Result Uploading ===

    dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/dataset.json", label="answer", metrics=Accuracy
    )

    # Login to Trismik with a valid API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    login(api_key)

    # If logged into trismik, evaluate will expect an experiment_id and project_id
    results = evaluate(
        inference,
        dataset,
        experiment_id="YOUR-EXPERIMENT-ID",  # New experiments can be created at runtime
        project_id="YOUR-PROJECT-ID",  # You must create a project on Trismik's dashboard
        return_items=True,
        metadata={
            "model_name": "microsoft/Phi-4-mini-instruct",
        },
    )

    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_6")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_6_output.json")
