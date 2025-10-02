"""Example 2 - Evaluation Datasets."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a simple Scorebook using datasets from local data and HuggingFace datasets.

    This example demonstrates how to use Scorebook with local datasets and Hugging Face datasets.
        - from_list
        - from_json
        - from_csv
        - from_huggingface
        - from_yaml

    The following datasets from Hugging Face are loaded:
        - MMLU
        - MMLU-Pro

    Firstly, a basic inference function is defined.
    Secondly, the datasets are created using the EvalDataset's from_* class methods.
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

    # === Creating Evaluation Datasets ===

    evaluation_items = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    # Create an EvalDataset from a list
    dataset_1 = EvalDataset.from_list(
        name="basic_questions",  # Dataset name
        label="answer",  # Key for the label value in evaluation items
        metrics=Accuracy,  # Metric/Metrics used to calculate scores
        data=evaluation_items,  # List of evaluation items
    )
    print(f"Loaded {dataset_1.name} from a list.")

    # Create an EvalDataset from a JSON file
    dataset_2 = EvalDataset.from_json(
        name="basic_questions_2",
        file_path="examples/example_datasets/basic_questions.json",
        label="answer",
        metrics=Accuracy,
    )
    print(f"Loaded {dataset_2.name} from a JSON file.")

    # Create an EvalDataset from a CSV file
    dataset_3 = EvalDataset.from_csv(
        name="basic_questions_3",
        file_path="examples/example_datasets/basic_questions.csv",
        label="answer",
        metrics=Accuracy,
    )
    print(f"Loaded {dataset_3.name} from a CSV file.")

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
        datasets=[  # Evaluate can be used with a list of datasets
            dataset_1,
            dataset_2,
            dataset_3,
            mmlu,
            mmlu_pro,
        ],
        sample_size=3,  # Sample size can be used for quick testing on large datasets
        return_items=True,  # Include the scores for individual items evaluated in results
        return_output=True,  # Include the model responses for each evaluated item in item results
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_2")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_2_output.json")
