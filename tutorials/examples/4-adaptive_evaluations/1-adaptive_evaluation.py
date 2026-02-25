"""Tutorials - Adaptive Evaluations - Example 1 - Adaptive Evaluation."""

import argparse
import asyncio
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import evaluate_async, login


async def main(project_id: str) -> Any:
    """Run adaptive evaluations using Trismik's adaptive testing.

    This example demonstrates how to use Trismik's adaptive evaluation feature
    with both multiple-choice and open-ended datasets.

    Adaptive evaluations use Item Response Theory (IRT) to efficiently estimate
    model capabilities by selecting questions based on previous responses.

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
        - OpenAI API key set in OPENAI_API_KEY environment variable
    """

    # Initialize OpenAI client
    client = AsyncOpenAI()
    model_name = "gpt-4o-mini"

    # Multiple-choice inference function
    async def mc_inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process multiple-choice inputs through OpenAI's API.

        Args:
            inputs: Each input is a dict with 'question' and 'choices' keys.
            hyperparameters: Model hyperparameters.

        Returns:
            List of model outputs (single letter answers).
        """
        outputs = []
        for input_val in inputs:
            choices = input_val.get("choices", [])
            prompt = (
                str(input_val.get("question", ""))
                + "\nOptions:\n"
                + "\n".join(f"{choice['id']}: {choice['text']}" for choice in choices)
            )

            messages = [
                {
                    "role": "system",
                    "content": "Answer with only the letter of the correct option.",
                },
                {"role": "user", "content": prompt},
            ]

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

    # Open-ended inference function
    async def open_ended_inference(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process open-ended inputs through OpenAI's API.

        Args:
            inputs: Each input is a dict with a 'question' key.
            hyperparameters: Model hyperparameters.

        Returns:
            List of model outputs (free-text answers).
        """
        outputs = []
        for input_val in inputs:
            prompt = str(input_val.get("question", ""))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer the question. Place your final answer "
                        "between <answer> and </answer> tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                )
                output = response.choices[0].message.content.strip()

                # Extract from <answer> tags if present
                start = output.rfind("<answer>")
                end = output.rfind("</answer>")
                if start != -1 and end > start:
                    output = output[start + len("<answer>") : end].strip()

            except Exception as e:
                output = f"Error: {str(e)}"

            outputs.append(output)

        return outputs

    # Step 1: Log in with your Trismik API key
    login()

    # Step 2: Run multiple-choice adaptive evaluation
    print("=== Multiple-Choice Adaptive Evaluation ===")
    mc_results = await evaluate_async(
        mc_inference,
        datasets="trismik/headQA:adaptive",
        split="test",
        experiment_id="Adaptive Evaluation Tutorial",
        project_id=project_id,
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )
    pprint(mc_results)

    # Step 3: Run open-ended adaptive evaluation
    print("\n=== Open-Ended Adaptive Evaluation ===")
    oe_results = await evaluate_async(
        open_ended_inference,
        datasets="trismik/fingpt_convfinqa_test:adaptive",
        experiment_id="Adaptive Evaluation Tutorial",
        project_id=project_id,
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        return_output=True,
    )
    pprint(oe_results)

    return {"multiple_choice": mc_results, "open_ended": oe_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive evaluation tutorial")
    parser.add_argument("--project-id", required=True, help="Trismik project ID")
    args = parser.parse_args()

    load_dotenv()
    log_file = setup_logging(experiment_id="1-adaptive_evaluation", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main(args.project_id))
    save_results_to_json(results_dict, output_dir, "1-adaptive_evaluation_output.json")
