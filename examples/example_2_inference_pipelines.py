"""
Example 2 - Using Inference Pipelines.

This example demonstrates how to use Scorebook's InferencePipeline in evaluations.

Inference pipelines separate the evaluation workflow into three distinct stages:
    1. Pre-processing: Convert raw dataset items into model-ready input format
    2. Inference: Execute model predictions on preprocessed data
    3. Post-processing: Extract final answers from raw model outputs

The logic for these stages can be encapsulated in reusable functions and used to create pipelines.
An inference pipeline can be passed into the evaluate function's inference parameter.
"""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a simple Scorebook evaluation using an InferencePipeline."""

    # === Pre-Processing ===

    def preprocessor(eval_item: Dict, **hyperparameter_config: Any) -> List[Any]:
        """Convert an evaluation item to a valid model input.

        Args:
            eval_item: An evaluation item from an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            A structured representation of an evaluation item for model input.
        """
        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and concisely.",
            },
            {"role": "user", "content": eval_item["question"]},
        ]

        return messages

    # === Inference ===

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(preprocessed_items: List[Any], **hyperparameter_config: Any) -> List[Any]:
        """Run model inference on preprocessed eval items.

        Args:
            preprocessed_items: The list of evaluation items for an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            A list of model outputs for an EvalDataset.
        """
        return [pipeline(model_input) for model_input in preprocessed_items]

    # === Post-Processing ===

    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output.

        Args:
            model_output: An evaluation item from an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            Parsed answer from the model output to be used for scoring.
        """
        return str(model_output[0]["generated_text"][-1]["content"])

    # === Evaluation ===

    # Step 1: Load the evaluation dataset
    eval_dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Create an inference pipeline, using the functions defined
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference,
        postprocessor=postprocessor,
    )

    # Step 3: Run the evaluation using the inference pipeline and dataset
    results = evaluate(inference_pipeline, eval_dataset)

    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_2")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_2_output.json")
