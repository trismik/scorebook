"""Example 2.1 - Evaluation Datasets."""

from pprint import pprint
from typing import Any, List

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

    def inference(inputs: List[Any], **hyperparameter_config: Any) -> list[Any]:
        """Return a list of model outputs for a list of inputs."""

        inference_results = []
        for input_text in inputs:
            messages = [
                {
                    "role": "system",
                    "content": """
                        Answer the question directly and concisely.
                        Provide only the answer no additional context ot text.
                    """,
                },
                {
                    "role": "user",
                    "content": input_text,
                },
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
        metrics=Accuracy,  # Metric/Metrics used to calculate scores
        items=evaluation_items,  # List of evaluation items
        input="question",  # Key for the input field in evaluation items
        label="answer",  # Key for the label field in evaluation items
    )
    print(f"Loaded {dataset_1.name} from a list.")

    # Create an EvalDataset from a JSON file
    dataset_2 = EvalDataset.from_json(
        name="basic_questions_2",
        path="examples/example_datasets/basic_questions.json",
        metrics=Accuracy,
        input="question",
        label="answer",
    )
    print(f"Loaded {dataset_2.name} from a JSON file.")

    # Create an EvalDataset from a CSV file
    dataset_3 = EvalDataset.from_csv(
        name="basic_questions_3",
        path="examples/example_datasets/basic_questions.csv",
        metrics=Accuracy,
        input="question",
        label="answer",
    )
    print(f"Loaded {dataset_3.name} from a CSV file.")

    # Load an EvalDatasets from a HF Dataset
    simple_qa = EvalDataset.from_huggingface(
        path="basicv8vc/SimpleQA",
        metrics=Accuracy,
        input="problem",
        label="answer",
        split="test",
    )
    print(f"Loaded {simple_qa.name} from Hugging Face.")

    # === Evaluation With Hugging Face Datasets ===

    results = evaluate(
        inference,
        datasets=[  # Evaluate can be used with a list of datasets
            dataset_1,
            dataset_2,
            dataset_3,
            simple_qa,
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
    log_file = setup_logging(experiment_id="example_2.1")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_2.1_output.json")
