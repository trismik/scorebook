"""
Hyperparameter Sweep Evaluation Example.

This example demonstrates Scorebook's automated hyperparameter sweeping capabilities.
Instead of manually testing different parameter combinations, Scorebook automatically
evaluates your model across a grid of hyperparameter values to find optimal configurations.

How Hyperparameter Sweeping Works:
- Define a hyperparameters dictionary with lists of values to test
- Scorebook generates all possible combinations (Cartesian product)
- Each combination is evaluated separately on the same dataset
- Results include performance metrics for every configuration

Example Configuration:
- max_new_tokens: [50, 75] (2 values)
- temperature: [0.6, 0.7] (2 values)
- Total combinations: 2 × 2 = 4 evaluations

Compare with example_1_simple_eval.py and example_2_inference_pipelines.py
to see the evolution from basic evaluation to advanced hyperparameter optimization.
"""

import json
from pathlib import Path
from typing import Any

import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> None:
    """Run the hyperparameter sweeps example."""
    output_dir = setup_output_directory()

    # Step 1: Load the evaluation dataset
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Initialize the language model
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
    )

    # Step 3: Define the preprocessing function
    def preprocessor(eval_item: dict) -> list:
        """Convert evaluation item to model input format."""
        prompt = eval_item["question"]

        messages = [
            {
                "role": "system",
                "content": "Answer the question directly and concisely.",
            },
            {"role": "user", "content": prompt},
        ]
        return messages

    # Step 4: Define the inference function with hyperparameter support
    # Execute model inference on preprocessed items
    # IMPORTANT: This function accepts **hyperparameters and passes them to the model
    # This enables the hyperparameter sweep functionality
    def inference_function(processed_items: list[list], **hyperparameters: Any) -> list[Any]:
        """Run model inference on preprocessed items."""
        outputs = []
        for messages in processed_items:
            # Pass hyperparameters directly to the model pipeline
            output = pipeline(messages, **hyperparameters)
            outputs.append(output)
        return outputs

    # Step 5: Define the postprocessing function
    def postprocessor(model_output: Any) -> str:
        """Extract the final answer from the model's output."""
        return str(model_output[0]["generated_text"][-1]["content"])

    # Step 6: Define hyperparameter grid
    # Specify multiple values for each hyperparameter to test
    # Scorebook will automatically test all combinations:
    # - max_new_tokens: [50, 75] (2 values)
    # - temperature: [0.6, 0.7] (2 values)
    # Total combinations: 2 × 2 = 4 different configurations
    hyperparameters = {
        "max_new_tokens": [50, 75],
        "temperature": [0.6, 0.7],
    }

    # Step 7: Create the inference pipeline
    # This pipeline will be used across all hyperparameter combinations
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 8: Run hyperparameter sweep evaluation
    # Execute evaluation across all hyperparameter combinations
    # - hyperparameters: Grid of parameters to sweep
    # - score_type="all": Returns both aggregate and per-item scores
    # - item_limit=10: Limits to 10 items per configuration for quick demonstration
    # Results will contain separate evaluations for each parameter combination
    results = evaluate(
        inference_pipeline,
        dataset,
        hyperparameters=hyperparameters,
        score_type="all",
        item_limit=10,
    )

    # Step 9: Save results to file
    # Export comprehensive results including all hyperparameter combinations
    # The JSON will contain results for each of the 4 parameter combinations
    with open(output_dir / "example_3_output.json", "w") as output_file:
        json.dump(results, output_file, indent=4)
        print(f"Results saved in {output_dir / 'example_3_output.json'}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation with hyperparam sweep and save results."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/example_results"),
        help=(
            "Directory to save evaluation outputs (CSV and JSON). "
            "Defaults to ./results in the current working directory."
        ),
    )
    args = parser.parse_args()
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


if __name__ == "__main__":
    main()
