"""
Simple Scorebook Evaluation Example.

This example demonstrates the fundamental workflow for evaluating a language model using Scorebook.
It shows how to:

1. Load an evaluation dataset from local JSON file
2. Set up a language model using Hugging Face transformers
3. Define a custom inference function with preprocessing and postprocessing
4. Run the evaluation and collect results
5. Save results for analysis

The example uses:
- Dataset: Local JSON dataset (question/answer pairs)
- Model: Microsoft Phi-4-mini-instruct
- Metric: Accuracy
- Sample size: 10 items (for quick demonstration)

This serves as a starting point for understanding Scorebook's core evaluation capabilities
before exploring more advanced features like inference pipelines and hyperparameter sweeps.
"""

import json
from pathlib import Path
from typing import Any

import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy


def main() -> None:
    """Run the simple evaluation example."""
    output_dir = setup_output_directory()

    # Step 1: Load the evaluation dataset
    # Create an EvalDataset from local JSON file
    # - Uses 'answer' field as ground truth labels
    # - Configures Accuracy metric for evaluation
    # - Loads from examples/example_datasets/dataset.json
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Initialize the language model
    # Set up a Hugging Face transformers pipeline with Phi-4-mini-instruct model
    # - Uses automatic torch dtype selection for optimal performance
    # - Automatically distributes model across available devices
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Step 3: Define the inference function
    # This function handles the complete inference workflow:
    # preprocessing → model inference → postprocessing
    def inference_function(eval_items: list[dict], **hyperparameters: Any) -> list[Any]:
        """Pre-processes dataset items, inference and post-processing result."""
        results = []
        for eval_item in eval_items:
            # Preprocess: Use the question directly from the local dataset
            prompt = eval_item["question"]

            # Create chat messages
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely.",
                },
                {"role": "user", "content": prompt},
            ]

            # Run inference and extract the model's response
            output = pipeline(messages)
            output = output[0]["generated_text"][-1]["content"]
            results.append(output)

        return results

    # Step 4: Run the evaluation
    # Execute the evaluation using scorebook's evaluate function
    # - Limits to 10 items for quick demonstration
    # - Returns structured results with metrics and per-item scores
    results = evaluate(inference_function, dataset, return_sample_size=10)
    print(results)

    # Step 5: Save results to file
    # Export evaluation results as JSON for later analysis
    with open(output_dir / "example_1_output.json", "w") as output_file:
        json.dump(results, output_file, indent=4)
        print(f"Results saved in {output_dir / 'example_1_output.json'}")


# ============================================================================
# Utility Functions
# ============================================================================


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation and save results.")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    main()
