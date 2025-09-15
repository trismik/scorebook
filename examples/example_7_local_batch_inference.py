"""Example 7 - Local Batch Inference."""

from pprint import pprint
from typing import Any, Dict, List

import torch
import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a Scorebook evaluation using local batch inference.

    This example demonstrates how to perform batch inference locally, where multiple
    evaluation items are processed simultaneously rather than one at a time.

    This approach offers several benefits:
        1. Improved throughput by processing multiple items in parallel
        2. Better GPU utilization through batched tensor operations
        3. More efficient memory usage compared to sequential processing

    The key difference from sequential inference is that the inference function
    receives all preprocessed items at once and returns all results together.
    """

    # === Pre-Processing ===

    def preprocessor(eval_item: Dict, **hyperparameter_config: Any) -> Dict[str, Any]:
        """Convert an evaluation item to a valid model input for batching.

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
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        batch_size=4,  # Process 4 items at once
    )

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
        # Extract messages from all preprocessed items
        all_messages = [item["messages"] for item in preprocessed_items]

        # Perform batch inference - the pipeline will process multiple inputs together
        batch_outputs = pipeline(
            all_messages,
            temperature=hyperparameter_config["temperature"],
            max_new_tokens=hyperparameter_config.get("max_new_tokens", 256),
            do_sample=True,
            batch_size=hyperparameter_config.get("batch_size", 4),
        )

        return list(batch_outputs)

    # === Post-Processing ===

    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output.

        Args:
            model_output: Raw model output from batch inference.
            hyperparameter_config: Model hyperparameters.

        Returns:
            Parsed answer from the model output to be used for scoring.
        """
        # Handle both single outputs and batch outputs
        if isinstance(model_output, list) and len(model_output) > 0:
            # For batch outputs, extract the generated text
            generated_text = model_output[0]["generated_text"]
            if isinstance(generated_text, list):
                # Extract the assistant's response (last message)
                return str(generated_text[-1]["content"])
            else:
                # Handle string format
                return str(generated_text).split("assistant\n")[-1].strip()

        return str(model_output)

    # === Evaluation With Batch InferencePipeline ===

    # Step 1: Create a batch inference pipeline
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=batch_inference,
        postprocessor=postprocessor,
    )

    # Step 2: Load the evaluation dataset
    eval_dataset = EvalDataset.from_json(
        file_path="examples/example_datasets/dataset.json", label="answer", metrics=Accuracy
    )

    # Step 3: Run the evaluation using batch inference
    print("Running local batch inference evaluation...")
    print(f"Processing {len(eval_dataset)} items with batch size 4")

    results = evaluate(
        inference_pipeline,
        eval_dataset,
        hyperparameters={
            "temperature": 0.7,
            "system_message": "Answer the question directly and concisely.",
            "max_new_tokens": 256,
            "batch_size": 4,
        },
        parallel=True,  # Enable parallel processing of batches
        return_items=True,  # Include results for individual items
        return_output=True,  # Include model outputs for debugging
        upload_results=False,  # Disable uploading for this example
    )

    print("\nBatch evaluation completed:")
    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_7")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_7_output.json")
