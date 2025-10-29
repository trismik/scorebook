"""Tutorials - Evaluate - Example 3 - Evaluating Cloud Models."""

import asyncio
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import EvalDataset, evaluate_async


async def main() -> Any:
    """Run an evaluation using a cloud-hosted model.

    This example demonstrates how to evaluate cloud-hosted models using OpenAI's API directly.

    Prerequisites:
        - OpenAI API key set in environment variable OPENAI_API_KEY
    """

    # Initialize OpenAI client
    client = AsyncOpenAI()
    model_name = "gpt-4o-mini"

    # Define an async inference function
    async def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through OpenAI's API.

        Args:
            inputs: Input values from an EvalDataset.
            hyperparameters: Model hyperparameters including system_message and temperature.

        Returns:
            List of model outputs for all inputs.
        """
        outputs = []
        for input_val in inputs:
            # Build messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": hyperparameters.get(
                        "system_message", "You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": str(input_val)},
            ]

            # Call OpenAI API
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=hyperparameters.get("temperature", 0.7),
                )
                output = response.choices[0].message.content.strip()
            except Exception as e:
                output = f"Error: {str(e)}"

            outputs.append(output)

        return outputs

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

    # Run evaluation
    results = await evaluate_async(
        inference,
        evaluation_dataset,
        hyperparameters={
            "system_message": (
                "Answer the question directly. Provide only the answer, without context."
            ),
            "temperature": 0.7,
        },
        return_items=True,
        return_output=True,
        upload_results=False,
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-evaluating_cloud_models", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main())
    save_results_to_json(results_dict, output_dir, "3-evaluating_cloud_models_output.json")
