"""Tutorials - Adaptive Evaluations - Example 2 - Adaptive Dataset Splits."""

import asyncio
import string
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / ".example_utils"))

from output import save_results_to_json
from setup import setup_logging

from scorebook import evaluate_async, login


async def main() -> Any:
    """

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
        - OpenAI API key set in OPENAI_API_KEY environment variable
    """

    # Initialize OpenAI client
    client = AsyncOpenAI()
    model_name = "gpt-4o-mini"

    # Define an async inference function
    async def inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through OpenAI's API.

        Args:
            inputs: Input values from an EvalDataset. For adaptive MMLU-Pro,
                   each input is a dict with 'question' and 'options' keys.
            hyperparameters: Model hyperparameters.

        Returns:
            List of model outputs for all inputs.
        """
        outputs = []
        for input_val in inputs:
            # Handle dict input from adaptive dataset
            if isinstance(input_val, dict):
                prompt = input_val.get("question", "")
                if "options" in input_val:
                    prompt += "\nOptions:\n" + "\n".join(
                        f"{letter}: {choice}"
                        for letter, choice in zip(string.ascii_uppercase, input_val["options"])
                    )
            else:
                prompt = str(input_val)

            # Build messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question with a single letter representing the correct answer from the list of choices. Do not provide any additional explanation or output beyond the single letter.",
                },
                {"role": "user", "content": prompt},
            ]

            # Call OpenAI API
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                )
                output = response.choices[0].message.content.strip()
            except Exception as e:
                output = f"Error: {str(e)}"

            outputs.append(output)

        return outputs

    # Step 1: Log in with your Trismik API key
    login("TRISMIK_API_KEY")

    # Step 2: Run adaptive evaluation
    results = await evaluate_async(
        inference,
        datasets="trismik/headQA:adaptive:test",  # Adaptive datasets have the ":adaptive" suffix
        experiment_id="Adaptive-Head-QA-Test-Evaluation",
        project_id='TRISMIK-PROJECT-ID',
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="2-adaptive_dataset_splits", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main())
    save_results_to_json(results_dict, output_dir, "2-adaptive_dataset_splits_output.json")