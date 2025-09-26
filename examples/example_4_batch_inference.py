"""Example 4 - Local Batch Inference."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a Scorebook evaluation using local batch inference.

    This example demonstrates how to perform batch inference locally

    This approach offers several benefits:
        1. Improved throughput by processing multiple items in parallel
        2. Better GPU utilization through batched tensor operations
        3. More efficient memory usage compared to sequential processing
    """

    # === Pre-Processing ===

    def preprocessor(eval_item: Dict, **hyperparameter_config: Any) -> Dict[str, Any]:
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
                "content": hyperparameter_config["system_message"],
            },
            {"role": "user", "content": eval_item["question"]},
        ]

        return {"messages": messages}

    # === Batch Inference ===

    # Initialize the pipeline with appropriate settings for batch processing
    model_name = "microsoft/Phi-4-mini-instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
        device="cpu",
    )
    pipeline.tokenizer.padding_side = "left"

    def batch_inference(
        preprocessed_items: List[Dict[str, Any]], **hyperparameter_config: Any
    ) -> List[Any]:
        """Run batch inference on multiple preprocessed eval items simultaneously.

        Args:
            preprocessed_items: List of preprocessed evaluation items.
            hyperparameter_config: Model hyperparameters.
        Returns:
            A list of model outputs for all evaluation items.
        """

        results = pipeline(
            [item["messages"] for item in preprocessed_items],
            temperature=hyperparameter_config["temperature"],
            batch_size=5,  # Set batch size
            do_sample=True,
            max_new_tokens=128,
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )

        return list(results)

    # === Post-Processing ===

    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output.

        Args:
            model_output: Raw model output from inference.
            hyperparameter_config: Model hyperparameters.
        Returns:
            Parsed answer from the model output to be used for scoring.
        """
        # Extract the assistant's response (last message in the conversation)
        return str(model_output[0]["generated_text"][-1]["content"])

    # === Evaluation With Batch InferencePipeline ===

    # Step 1: Create a batch inference pipeline
    inference_pipeline = InferencePipeline(
        model=model_name,
        preprocessor=preprocessor,
        inference_function=batch_inference,
        postprocessor=postprocessor,
    )

    # Step 2: Load the evaluation dataset
    eval_dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/basic_questions.json", label="answer", metrics=Accuracy
    )

    # Step 3: Run the evaluation

    # Define hyperparameters
    hyperparameters = {
        "system_message": "Answer the question directly. Provide no additional context",
        "temperature": 0.7,
        "max_new_tokens": 128,
    }

    results = evaluate(
        inference_pipeline,
        eval_dataset,
        hyperparameters=hyperparameters,
        return_aggregates=True,  # Include aggregate results for each configuration
        return_items=True,  # Include results for individual items
        return_output=True,  # Include model outputs for debugging
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_4")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_4_output.json")
