"""Example 4 - Local Batch Inference."""

from pprint import pprint
from typing import Any, Dict, List

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run a Scorebook evaluation using local batch inference with hyperparameter sweeps.

    This example demonstrates how to perform batch inference locally combined with
    hyperparameter sweeps, where multiple evaluation items are processed simultaneously
    across different hyperparameter configurations.

    This approach offers several benefits:
        1. Improved throughput by processing multiple items in parallel
        2. Better GPU utilization through batched tensor operations
        3. More efficient memory usage compared to sequential processing
        4. Systematic hyperparameter optimization across multiple configurations

    The key differences from sequential inference:
        - The inference function receives all preprocessed items at once
        - Multiple hyperparameter configurations are tested systematically
        - Results include performance metrics for each configuration
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
    # Using the same model as example 1
    model_name = "microsoft/Phi-4-mini-instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
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
        print("\n=== Batch Inference Debug ===")
        print(f"Processing {len(preprocessed_items)} items in batch")
        print(f"Temperature: {hyperparameter_config['temperature']}")

        # Extract messages from all preprocessed items for batch processing
        all_messages = [item["messages"] for item in preprocessed_items]
        print(f"Sample message structure: {all_messages[0] if all_messages else 'None'}")

        # Run batch inference using the same approach as example 1
        print("Starting batch inference...")
        batch_outputs = pipeline(
            all_messages, temperature=hyperparameter_config["temperature"], batch_size=1
        )

        print(f"Raw batch outputs type: {type(batch_outputs)}")
        print(
            f"Number of batch outputs: "
            f"{len(batch_outputs) if hasattr(batch_outputs, '__len__') else 'N/A'}"
        )
        if batch_outputs:
            print(f"First output structure: {type(batch_outputs[0])}")
            print(
                f"First output keys: "
                f"{batch_outputs[0].keys() if hasattr(batch_outputs[0], 'keys') else 'N/A'}"
            )

        # Extract the generated content from each output
        results = []
        for i, output in enumerate(batch_outputs):
            try:
                content = output[0]["generated_text"][-1]["content"]
                results.append(content)
                print(f"Item {i}: Extracted content: {content}")
            except Exception as e:
                print(f"Item {i}: Error extracting content: {e}")
                print(f"Item {i}: Output structure: {output}")
                results.append(str(output))

        print(f"Final results: {results}")
        print("=== End Batch Inference Debug ===\n")
        return results

    # === Post-Processing ===

    def postprocessor(model_output: Any, **hyperparameter_config: Any) -> str:
        """Extract the final parsed answer from the model output.

        Args:
            model_output: Raw model output from batch inference.
            hyperparameter_config: Model hyperparameters.
        Returns:
            Parsed answer from the model output to be used for scoring.
        """
        # The batch inference already extracts the content, so just return it
        return str(model_output)

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

    # Step 3: Run the evaluation using batch inference with hyperparameter sweep
    print("Running local batch inference evaluation with hyperparameter sweep...")
    print(f"Processing {len(eval_dataset)} items with multiple configurations")

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

    print("\nBatch evaluation completed:")
    pprint(results)
    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_4")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_4_output.json")
