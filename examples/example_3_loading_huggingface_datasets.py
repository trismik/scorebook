"""
Loading Hugging Face Datasets Example.

This example demonstrates how to load and evaluate datasets directly from Hugging Face Hub
using Scorebook. It shows how to work with real benchmark datasets like MMLU-Pro, which
have different formats and preprocessing requirements compared to simple local datasets.

Key Features Demonstrated:
1. **Hugging Face Integration**: Load datasets directly from the Hugging Face Hub
2. **Multiple Choice Processing**: Handle datasets with multiple choice questions
3. **Benchmark Evaluation**: Work with established academic benchmarks
4. **Format Adaptation**: Adapt preprocessing for different dataset structures

The example uses:
- Dataset: MMLU-Pro (multi-choice academic questions from Hugging Face Hub)
- Model: Microsoft Phi-4-mini-instruct
- Metric: Accuracy
- Sample size: 10 items (for quick demonstration)

Compare with examples 1-2 to see how preprocessing differs between simple question-answer
datasets and more complex multiple choice benchmarks.
"""

import string
from typing import Any

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline


def main() -> Any:
    """Run the Hugging Face dataset loading example."""

    # Step 1: Load the evaluation dataset from Hugging Face Hub
    #    Create an EvalDataset from Hugging Face Hub using the MMLU-Pro benchmark
    #    Uses 'answer' field as ground truth labels
    #    Configures Accuracy metric for evaluation
    #    Uses validation split for evaluation
    print("Loading MMLU-Pro dataset from Hugging Face Hub...")
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )
    print(f"Loaded {len(mmlu_pro)} items from MMLU-Pro dataset")

    # Step 2: Initialize the language model
    #    Set up a Hugging Face transformers pipeline with Phi-4-mini-instruct model
    #    Uses automatic torch dtype selection for optimal performance
    #    Automatically distributes model across available devices
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Step 3: Define the preprocessing function
    #    Convert raw MMLU-Pro dataset items into model-ready input format
    #    MMLU-Pro has a different structure than simple Q&A datasets
    def preprocessor(eval_item: dict) -> list:
        """Convert MMLU-Pro evaluation item to model input format."""
        # Format the question with multiple choice options (A, B, C, D...)
        prompt = f"{eval_item['question']}\nOptions:\n" + "\n".join(
            [
                f"{letter}: {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            ]
        )

        messages = [
            {
                "role": "system",
                "content": "Answer the question using only a single letter (for example, 'A').",
            },
            {"role": "user", "content": prompt},
        ]

        return messages

    # Step 4: Define the inference function
    #    Execute model inference on preprocessed items
    #    Takes a batch of preprocessed items and returns raw model outputs
    def inference_function(processed_items: list[list], **hyperparameters: Any) -> list[Any]:
        """Run model inference on preprocessed items."""
        outputs = []
        for messages in processed_items:
            output = pipeline(messages)
            outputs.append(output)
        return outputs

    # Step 5: Define the postprocessing function
    #    Extract the final prediction from raw model output
    #    For MMLU-Pro, we need to extract single letter answers (A, B, C, D, etc.)
    def postprocessor(model_output: Any) -> str:
        """Extract the letter answer from model output."""
        response = str(model_output[0]["generated_text"][-1]["content"])

        # Extract the first uppercase letter from the response
        for char in response:
            if char in string.ascii_uppercase:
                return char

        # Fallback: return the first character if it's a letter
        if response and response[0].upper() in string.ascii_uppercase:
            return response[0].upper()

        # Default fallback
        return ""

    # Step 6: Create the inference pipeline
    #    Combine all three stages into a modular InferencePipeline object
    #    This demonstrates how to adapt Scorebook for different dataset formats
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 7: Run the evaluation
    #    Execute the evaluation using the Hugging Face dataset
    #    Limits to 10 items for quick demonstration
    #    Returns structured results with metrics and per-item scores
    print("\nRunning evaluation on MMLU-Pro dataset...")
    results = evaluate(inference_pipeline, mmlu_pro, sample_size=10)
    print(results)

    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_3")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_3_output.json")
