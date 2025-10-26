"""Tutorials - Evaluate - Example 2 - Evaluating Local Models with Batching."""

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
    """Run a Scorebook evaluation using local batch inference.

    This example demonstrates how to perform batch inference locally.

    This approach offers several benefits:
        1. Improved throughput by processing multiple items in parallel
        2. Better GPU utilization through batched tensor operations
        3. More efficient memory usage compared to sequential processing
    """

    # Initialize the pipeline with appropriate settings for batch processing
    model_name = "google/flan-t5-small"

    # Task is text2text-generation for seq2seq models
    pipeline = transformers.pipeline(
        "text2text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",  # will pick up gpu if available
    )

    # Define a batch inference function
    def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process multiple inputs through the model in batches.

        Args:
            inputs: Input values from an EvalDataset.
            hyperparameters: Model hyperparameters including batch_size and max_new_tokens.

        Returns:
            List of model outputs for all inputs.
        """
        # Preprocess: Convert inputs to strings
        preprocessed_inputs = [str(input_val) for input_val in inputs]

        # Run batch inference
        raw_results = pipeline(
            preprocessed_inputs,
            batch_size=hyperparameters["batch_size"],
            max_new_tokens=hyperparameters["max_new_tokens"],
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )

        # Postprocess: Extract and clean the generated text
        final_outputs = [str(result["generated_text"]).strip() for result in raw_results]

        return final_outputs

    # Create a list of evaluation items
    evaluation_items = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    # Create an evaluation dataset
    evaluation_dataset = EvalDataset.from_list(
        name="basic_questions",    # Dataset name
        metrics="accuracy",        # Metric/Metrics used to calculate scores
        items=evaluation_items,    # List of evaluation items
        input="question",          # Key for the input field in evaluation items
        label="answer",            # Key for the label field in evaluation items
    )


    # Define hyperparameters
    hyperparameters = {
        "max_new_tokens": 128,
        "batch_size": 2,
    }

    # Run the evaluation with batch inference
    results = evaluate(
        inference,
        evaluation_dataset,
        hyperparameters=hyperparameters,
        return_aggregates=True,  # Include aggregate results for each configuration
        return_items=True,       # Include results for individual items
        return_output=True,      # Include model outputs for debugging
        upload_results=False,    # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="2-evaluating_local_models_with_batching")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "2-evaluating_local_models_with_batching_output.json")
