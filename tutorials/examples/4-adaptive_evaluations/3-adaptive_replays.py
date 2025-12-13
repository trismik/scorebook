"""Tutorials - Adaptive Evaluations - Example 3 - Adaptive Replays."""

import asyncio
import string
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import evaluate_async, login, replay_async


async def main() -> Any:
    """Run an adaptive evaluation and then replay it with a different model.

    This example demonstrates how to use Trismik's adaptive replay feature.
    Replays allow you to re-run the exact same test questions from a previous
    adaptive run with a different model or configuration, enabling fair
    model-to-model comparisons.

    Use cases for adaptive replays:
        - Model comparison: Compare two models on identical questions
        - Hyperparameter tuning: Test different temperatures on the same questions
        - Regression testing: Verify a new model version performs similarly
        - A/B testing: Compare model variants under identical conditions

    How replays differ from regular adaptive runs:
        - Regular runs: Questions adapt based on model responses (CAT algorithm)
        - Replays: Questions are fixed to match the original run exactly

    Prerequisites:
        - Valid Trismik API key set in TRISMIK_API_KEY environment variable
        - A Trismik project ID
        - OpenAI API key set in OPENAI_API_KEY environment variable
    """

    # Initialize OpenAI client
    client = AsyncOpenAI()

    # Define inference functions for two different models
    async def inference_model_a(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through gpt-4o-mini (Model A)."""
        return await run_inference(client, inputs, model="gpt-4o-mini", **hyperparameters)

    async def inference_model_b(inputs: List[Any], **hyperparameters: Any) -> List[Any]:
        """Process inputs through gpt-4o (Model B)."""
        return await run_inference(client, inputs, model="gpt-4o", **hyperparameters)

    # Step 1: Log in with your Trismik API key
    login()

    # Step 2: Run the original adaptive evaluation with Model A
    print("=" * 60)
    print("Running original adaptive evaluation with gpt-4o-mini...")
    print("=" * 60)

    original_results = await evaluate_async(
        inference_model_a,
        datasets="trismik/headQA:adaptive",
        split="test",
        experiment_id="Adaptive-Replay-Comparison",
        project_id="TRISMIK-PROJECT-ID",
        return_dict=False,  # Return EvalResult object to access run_id
    )

    # Extract the run_id from the original evaluation
    original_run_id = original_results.run_results[0].run_id
    original_scores = original_results.aggregate_scores[0]

    print(f"\nOriginal run completed!")
    print(f"Run ID: {original_run_id}")
    print(f"Model A (gpt-4o-mini) theta: {original_scores.get('score', {}).get('theta', 'N/A')}")

    # Step 3: Replay the same questions with Model B
    print("\n" + "=" * 60)
    print("Replaying with gpt-4o (same questions, different model)...")
    print("=" * 60)

    replay_results = await replay_async(
        inference_model_b,
        previous_run_id=original_run_id,
        experiment_id="Adaptive-Replay-Comparison",
        project_id="TRISMIK-PROJECT-ID",
        metadata={"model": "gpt-4o", "comparison_type": "model_upgrade"},
        return_dict=False,  # Return AdaptiveReplayRunResult object
    )

    replay_scores = replay_results.aggregate_scores

    print(f"\nReplay completed!")
    print(f"New Run ID: {replay_results.run_id}")
    print(f"Replay of: {replay_results.replay_of_run}")
    print(f"Model B (gpt-4o) theta: {replay_scores.get('score', {}).get('theta', 'N/A')}")

    # Step 4: Compare results
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    results_summary = {
        "original_run": {
            "run_id": original_run_id,
            "model": "gpt-4o-mini",
            "scores": original_scores,
        },
        "replay_run": {
            "run_id": replay_results.run_id,
            "replay_of": replay_results.replay_of_run,
            "model": "gpt-4o",
            "scores": replay_scores,
        },
    }

    pprint(results_summary)
    return results_summary


async def run_inference(
    client: AsyncOpenAI,
    inputs: List[Any],
    model: str = "gpt-4o-mini",
    **hyperparameters: Any,
) -> List[Any]:
    """Process inputs through OpenAI's API.

    Args:
        client: OpenAI async client instance.
        inputs: Input values from an EvalDataset. For adaptive headQA,
               each input is a dict with 'question' and 'options' keys.
        model: OpenAI model name to use.
        hyperparameters: Model hyperparameters (e.g., temperature).

    Returns:
        List of model outputs for all inputs.
    """
    temperature = hyperparameters.get("temperature", 0.7)
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
                model=model,
                messages=messages,
                temperature=temperature,
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            output = f"Error: {str(e)}"

        outputs.append(output)

    return outputs


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-adaptive_replays", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main())
    save_results_to_json(results_dict, output_dir, "3-adaptive_replays_output.json")