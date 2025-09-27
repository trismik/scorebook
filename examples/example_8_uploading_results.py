"""Example 8 - Uploading Evaluation Results to Trismik's dashboard."""

import os
from pprint import pprint
from typing import Any, Dict, List

import transformers
from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate, login
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a scorebook evaluation and submit the results to Trismik's dashboard.

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
                {"role": "system", "content": hyperparameter_config["system_message"]},
                {"role": "user", "content": eval_item["question"]},
            ]
            results.append(pipeline(messages)[0]["generated_text"][-1]["content"])
        return results

    # === Evaluation With Result Uploading ===

    dataset = EvalDataset.from_json(
        path="examples/example_datasets/basic_questions.json", label="answer", metrics=Accuracy
    )

    # Login to Trismik with a valid API key
    api_key = os.environ.get("TRISMIK_API_KEY")
    login(api_key)

    # If logged into trismik, evaluate will expect an experiment_id and project_id
    results = evaluate(
        inference,
        dataset,
        hyperparameters={
            "system_message": "Answer the question directly and concisely.",
        },
        # New experiments can be created at runtime
        experiment_id="Scorebook-Example-8-Uploading-Results",
        # You must create a project on dashboard
        project_id="YOUR_PROJECT_ID",
        return_items=True,
        metadata={
            "model": "microsoft/Phi-4-mini-instruct",
        },
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_8")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_8_output.json")
