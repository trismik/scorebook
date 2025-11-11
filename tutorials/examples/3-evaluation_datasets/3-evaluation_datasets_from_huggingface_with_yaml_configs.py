"""Tutorials - Evaluation Datasets - Example 3 - Loading from YAML Config."""

import asyncio
from pathlib import Path
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import EvalDataset, evaluate_async


async def main() -> Any:
    """Run evaluations using datasets loaded from YAML configuration files.

    This example demonstrates how to use YAML configuration files to define
    dataset loading parameters. YAML configs are useful for:
        - Storing dataset configurations in version control
        - Reusing the same dataset configuration across projects
        - Defining complex prompt templates and field mappings

    The YAML files contain:
        - HuggingFace dataset path and split information
        - Metrics to use for evaluation
        - Jinja2 templates for input and label formatting
        - Metadata about the dataset

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
            hyperparameters: Model hyperparameters.

        Returns:
            List of model outputs for all inputs.
        """
        outputs = []
        for input_val in inputs:
            # Build messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": "Answer the multiple choice question by selecting the correct letter (A, B, C, D, etc.). Provide ONLY the letter of your answer, no additional text or explanation.",
                },
                {"role": "user", "content": str(input_val)},
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

    # Construct paths to YAML config files
    yaml_configs_dir = Path(__file__).parent / "example_yaml_configs"
    cais_mmlu_yaml = yaml_configs_dir / "Cais-MMLU.yaml"
    tiger_mmlu_pro_yaml = yaml_configs_dir / "TIGER-Lab-MMLU-Pro.yaml"

    # Load Cais-MMLU dataset from YAML configuration
    cais_mmlu = EvalDataset.from_yaml(str(cais_mmlu_yaml))
    print(f"Loaded {cais_mmlu.name} from YAML config: {len(cais_mmlu.items)} items")

    # Load TIGER-Lab MMLU-Pro dataset from YAML configuration
    tiger_mmlu_pro = EvalDataset.from_yaml(str(tiger_mmlu_pro_yaml))
    print(f"Loaded {tiger_mmlu_pro.name} from YAML config: {len(tiger_mmlu_pro.items)} items")

    # Run evaluation on both datasets
    results = await evaluate_async(
        inference,
        datasets=[cais_mmlu, tiger_mmlu_pro],
        sample_size=5,  # Sample 5 items from each dataset for quick testing
        return_aggregates=True,
        return_items=True,
        return_output=True,
        upload_results=False,
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="3-evaluation_datasets_from_yaml", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = asyncio.run(main())
    save_results_to_json(results_dict, output_dir, "3-evaluation_datasets_from_yaml_output.json")