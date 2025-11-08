"""Tutorials - Evaluate - Example 5 - Hyperparameter Sweeps."""

from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv

from scorebook.utils.tutorial_utils import save_results_to_json, setup_logging

from scorebook import EvalDataset, evaluate


def main() -> Any:
    """Run a Scorebook evaluation with a hyperparameter sweep.

    This example demonstrates how Scorebook can automatically test multiple
    hyperparameter configurations in a single evaluation.

    How Hyperparameter Sweeping Works:
    - Define hyperparameters with lists of values to test
    - Scorebook generates all possible combinations (Cartesian product)
    - Each configuration is evaluated separately on the same dataset

    Example Hyperparameters:
    - system_message: "Answer the question directly and concisely." (1 value)
    - temperature: [0.6, 0.7, 0.8] (3 values)
    - top_p: [0.7, 0.8, 0.9] (3 values)
    - top_k: [10, 20, 30] (3 values)

    Total configurations = 1 × 3 × 3 × 3 = 27 hyperparameter configurations
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
            hyperparameters: Model hyperparameters including system_message, temperature, top_p, top_k.

        Returns:
            List of model outputs for all inputs.
        """
        outputs = []
        for input_val in inputs:
            # Preprocess: Build messages
            messages = [
                {"role": "system", "content": hyperparameters["system_message"]},
                {"role": "user", "content": str(input_val)},
            ]

            # Run inference
            result = pipeline(
                messages,
                temperature=hyperparameters["temperature"],
                top_p=hyperparameters.get("top_p"),
                top_k=hyperparameters.get("top_k"),
            )

            # Postprocess: Extract the answer
            output = str(result[0]["generated_text"][-1]["content"])
            outputs.append(output)

        return outputs

    # Create a list of evaluation items
    evaluation_items = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    # Create an evaluation dataset
    evaluation_dataset = EvalDataset.from_list(
        name="basic_questions",
        metrics="accuracy",
        items=evaluation_items,
        input="question",
    label="answer",
    )

    # Define hyperparameters with lists of values to create a sweep
    hyperparameters = {
        "system_message": "Answer the question directly and concisely.",
        "temperature": [0.6, 0.7, 0.8],
        "top_p": [0.7, 0.8, 0.9],
        "top_k": [10, 20, 30],
    }

    # Run evaluation across all hyperparameter combinations
    results = evaluate(
        inference,
        evaluation_dataset,
        hyperparameters=hyperparameters,
        return_aggregates=True,
        return_items=True,
        return_output=True,
        upload_results=False,
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="5-hyperparameter_sweeps", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "5-hyperparameter_sweeps_output.json")
