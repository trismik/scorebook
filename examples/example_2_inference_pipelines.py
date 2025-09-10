"""
Modular Inference Pipeline Example.

This example demonstrates Scorebook's modular InferencePipeline approach for model evaluation.
Unlike a single inference function, pipelines separate the evaluation workflow into three
distinct, reusable stages:

1. **Preprocessing**: Convert raw dataset items into model-ready input format
2. **Inference**: Execute model predictions on preprocessed data
3. **Postprocessing**: Extract final answers from raw model outputs

Key Benefits of Inference Pipelines:
- **Modularity**: Each stage can be developed, tested, and reused independently
- **Flexibility**: Easy to swap different preprocessors, models, or postprocessors
- **Maintainability**: Clear separation of concerns makes code easier to understand
- **Reusability**: Components can be shared across different evaluation setups

This approach is particularly useful when:
- Working with complex preprocessing requirements
- Switching between different models or formats
- Building reusable evaluation components
- Collaborating with teams on evaluation workflows

Compare with example_1_simple_eval.py to see the difference between monolithic
inference functions and modular pipeline approaches.
"""

from typing import Any

import transformers
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy


def main() -> Any:
    """Run the inference pipelines example."""

    # Step 1: Load the evaluation dataset
    dataset = EvalDataset.from_json(
        "examples/example_datasets/dataset.json", label="answer", metrics=[Accuracy]
    )

    # Step 2: Initialize the language model
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Step 3: Define the preprocessing function
    #    Convert raw dataset items into model-ready input format
    #    This function handles formatting the question for the model
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
    #    Converts model output into the format needed for metric calculation
    def postprocessor(model_output: Any) -> str:
        """Extract the final answer from model output."""
        return str(model_output[0]["generated_text"][-1]["content"])

    # Step 6: Create the inference pipeline
    #    Combine all three stages into a modular InferencePipeline object
    #    This demonstrates Scorebook's modular approach to model evaluation
    #    Separates concerns: preprocessing, inference, and postprocessing
    #    Makes components reusable across different models or datasets
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Step 7: Run the evaluation
    results = evaluate(inference_pipeline, dataset, sample_size=10)
    print(results)

    return results


if __name__ == "__main__":
    log_file = setup_logging(experiment_id="example_2")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_2_output.json")
