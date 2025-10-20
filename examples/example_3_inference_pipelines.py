"""Example 3 - Using Inference Pipelines."""

from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, InferencePipeline, evaluate
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a simple Scorebook evaluation using an InferencePipeline.

    This example demonstrates how to use Scorebook's InferencePipeline in evaluations.

    Inference pipelines separate the evaluation workflow into three distinct stages:
        1. Pre-processing: Convert raw dataset items into model-ready input format
        2. Inference: Execute model predictions on preprocessed data
        3. Post-processing: Extract final answers from raw model outputs

    These stages can be encapsulated in reusable functions and used to create pipelines.
    An inference pipeline can be passed into the evaluate function's inference parameter.
    """

    # === Pre-Processing ===

    # The preprocessor function is responsible for mapping items in an Eval Dataset to model inputs
    def preprocessor(input_value: str, **hyperparameter_config: Any) -> List[Any]:
        """Convert an evaluation input to a valid model input.

        Args:
            input_value: The input value from the dataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            A structured representation of an evaluation item for model input.
        """
        messages = [
            {
                "role": "system",
                "content": hyperparameter_config["system_message"],
            },
            {"role": "user", "content": input_value},
        ]

        return messages

    # === Inference ===

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # An inference function for an InferencePipeline that returns a list of raw outputs
    def inference(preprocessed_items: List[Any], **hyperparameter_config: Any) -> List[Any]:
        """Run model inference on preprocessed eval items.

        Args:
            preprocessed_items: The list of evaluation items for an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            A list of model outputs for an EvalDataset.
        """
        return [
            pipeline(model_input, temperature=hyperparameter_config["temperature"])
            for model_input in preprocessed_items
        ]

    # === Post-Processing ===

    # The postprocessor function parses model output for metric scoring
    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output.

        Args:
            model_output: An evaluation item from an EvalDataset.
            hyperparameter_config: Model hyperparameters.

        Returns:
            Parsed answer from the model output to be used for scoring.
        """
        return str(model_output[0]["generated_text"][-1]["content"])

    # === Evaluation With An InferencePipeline ===

    # Step 1: Create an inference pipeline, using the 3 functions defined
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference,
        postprocessor=postprocessor,
    )

    # Step 2: Load the evaluation dataset
    eval_dataset = EvalDataset.from_json(
        path="examples/example_datasets/basic_questions.json",
        metrics=Accuracy,
        input="question",
        label="answer",
    )

    # Step 3: Run the evaluation using the inference pipeline and dataset
    results = evaluate(
        inference_pipeline,
        eval_dataset,
        hyperparameters={
            "temperature": 0.7,
            "system_message": "Answer the question directly and concisely.",
        },
        return_items=True,  # Enable to include results for individual items in the dict returned.
        return_output=True,  # Enable to include the model's output for individual items.
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_3")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_3_output.json")
