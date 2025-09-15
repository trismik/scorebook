"""Example 4 - Evaluations with Hugging Face Datasets."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a simple Scorebook evaluation using datasets from Hugging.

    This example demonstrates how to evaluate models with the following datasets from Hugging Face:
        - MMLU
        - MMLU-Pro
        - TODO: ADD ONE MORE EXAMPLE DATASET

    Firstly, a basic inference function is defined.
    Secondly, the datasets are loaded using the EvalDataset.from_huggingface() method.
    Finally, the evaluation is run using all loaded datasets simultaneously.
    """

    # === Inference Function Setup ===

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(eval_items: List[Dict], **hyperparameter_config: Any) -> list[Any]:
        """Return a list of model outputs for a list of evaluation items."""

        inference_results = []
        for eval_item in eval_items:
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely.",
                },
                {"role": "user", "content": eval_item["question"]},
            ]

            output = pipeline(messages)
            inference_results.append(output[0]["generated_text"][-1]["content"])

        return inference_results

    # === Hugging Face Dataset Loading ===

    mmlu = EvalDataset.from_huggingface(
        path="cais/mmlu",
        label="answer",
        metrics=Accuracy,
        split="test",
        name="all",
    )
    print(f"Loaded {mmlu.name} from Hugging Face.")

    mmlu_pro = EvalDataset.from_huggingface(
        path="TIGER-Lab/MMLU-Pro",
        label="answer",
        metrics=Accuracy,
        split="validation",
        name="default",
    )
    print(f"Loaded {mmlu_pro.name} from Hugging Face.")

    # === Evaluation With Hugging Face Datasets ===

    results = evaluate(
        inference,
        datasets=[mmlu, mmlu_pro],  # Evaluate can be used with lists datasets
        sample_size=10,  # Sample size can be used for quick testing on large datasets.
        return_items=True,
        return_output=True,
        parallel=False,
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_4")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_4_output.json")
