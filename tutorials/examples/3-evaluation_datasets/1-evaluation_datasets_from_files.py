"""Tutorials - Evaluation Datasets - Example 1 - Loading Datasets from Files."""

import sys
from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import EvalDataset, evaluate


def main() -> Any:
    """Run evaluations using datasets loaded from local files.

    This example demonstrates how to load evaluation datasets from files:
        - from_json: Load datasets from JSON files
        - from_csv: Load datasets from CSV files

    Both methods support loading data from local files with custom field mappings.
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
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely. Provide only the answer, no additional context or text.",
                },
                {"role": "user", "content": str(input_val)},
            ]

            # Run inference
            result = pipeline(messages, temperature=0.7)

            # Extract the answer
            output = str(result[0]["generated_text"][-1]["content"])
            outputs.append(output)

        return outputs

    # Construct paths to example data files
    example_datasets_dir = Path(__file__).parent / "example_datasets"
    json_path = example_datasets_dir / "basic_questions.json"
    csv_path = example_datasets_dir / "basic_questions.csv"

    # Load dataset from JSON file
    json_dataset = EvalDataset.from_json(
        name="basic_questions_json",
        path=str(json_path),
        metrics="accuracy",
        input="question",
        label="answer",
    )
    print(f"Loaded {json_dataset.name} from JSON file: {len(json_dataset.items)} items")

    # Load dataset from CSV file
    csv_dataset = EvalDataset.from_csv(
        name="basic_questions_csv",
        path=str(csv_path),
        metrics="accuracy",
        input="question",
        label="answer",
    )
    print(f"Loaded {csv_dataset.name} from CSV file: {len(csv_dataset.items)} items")

    # Run evaluation on both datasets
    results = evaluate(
        inference,
        datasets=[json_dataset, csv_dataset],
        return_aggregates=True,
        return_items=True,
        return_output=True,
        upload_results=False,
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="1-evaluation_datasets_from_files", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "1-evaluation_datasets_from_files_output.json")
