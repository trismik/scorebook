"""Example 7 - Using Hyperparameter Sweeps."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from dotenv import load_dotenv
from example_helpers import (
    save_results_to_json,
    setup_logging,
    setup_openai_model_parser,
    setup_output_directory,
)

from scorebook import EvalDataset, InferencePipeline, evaluate
from scorebook.metrics import Accuracy


def main(model_name: str) -> Any:
    """Run a Scorebook evaluation with a hyperparameter sweep.

    This example demonstrates Scorebook can create a grid of hyperparameter configurations.
    Instead of manually testing different parameter configurations, Scorebook can automatically
    evaluate your models across grids of hyperparameter values to find optimal configurations.

    How Hyperparameter Sweeping Works:
    - Define a hyperparameter dictionary with lists of values to test
    - Scorebook generates all possible configurations (Cartesian product)
    - Each configuration is run separately on the same dataset

    Example Hyperparameters:
    - system_message: "Answer the question directly and concisely." (1 value)
    - max_new_tokens: [50, 75, 100] (3 values)
    - temperature: [0.6, 0.7, 0.8] (3 values)
    - top_p: [0.7, 0.8, 0.9] (3 values)

    Total configuration = 1 × 3 × 3 × 3 = 27 hyperparameter configurations
    """

    # === Inference Pipeline Setup ===

    def preprocessor(eval_item: Dict, **hyperparameter_config: Any) -> List[Any]:
        """Convert an evaluation item to a valid model input."""
        messages = [
            {"role": "system", "content": hyperparameter_config["system_message"]},
            {"role": "user", "content": eval_item["question"]},
        ]
        return messages

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(preprocessed_items: List[Any], **hyperparameter_config: Any) -> List[Any]:
        """Run model inference on preprocessed eval items."""
        return [
            pipeline(
                model_input,
                temperature=hyperparameter_config["temperature"],
                top_p=hyperparameter_config.get("top_p"),
                top_k=hyperparameter_config.get("top_k"),
            )
            for model_input in preprocessed_items
        ]

    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output."""
        return str(model_output[0]["generated_text"][-1]["content"])

    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference,
        postprocessor=postprocessor,
    )

    # === Evaluation Across Hyperparameters ===

    dataset = EvalDataset.from_json(
        path="examples/example_datasets/basic_questions.json", label="answer", metrics=Accuracy
    )

    # Define hyperparameters, using lists of values to generate configurations
    hyperparameters = {
        "system_message": "Answer the question directly and concisely.",
        "temperature": [0.6, 0.7, 0.8],
        "top_p": [0.7, 0.8, 0.9],
        "top_k": [10, 20, 30],
    }

    results = evaluate(
        inference=inference_pipeline,
        datasets=dataset,
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
    log_file = setup_logging(experiment_id="example_7")
    output_dir = setup_output_directory()
    model = setup_openai_model_parser()
    results_dict = main(model)
    save_results_to_json(results_dict, output_dir, "example_7_output.json")
