"""Example 2.2 - Evaluation Datasets from YAML Configuration."""

from pprint import pprint
from typing import Any, List

import transformers
from dotenv import load_dotenv
from example_helpers import save_results_to_json, setup_logging, setup_output_directory

from scorebook import EvalDataset, evaluate


def main() -> Any:
    """Run a simple Scorebook using datasets from YAML configuration files.

    This example demonstrates how to use Scorebook with EvalDataset.from_yaml
    to load dataset configurations from YAML files. The YAML files contain
    all necessary configuration including dataset path, metrics, and prompt templates.

    The example loads two MMLU datasets:
        - Cais-MMLU: Standard MMLU questions from Cais
        - TIGER-Lab/MMLU-Pro: Enhanced MMLU questions from TIGER-Lab

    Firstly, a basic inference function is defined.
    Secondly, the datasets are created using EvalDataset.from_yaml.
    Finally, the evaluation is run using both datasets simultaneously.
    """

    # === Inference Function Setup ===

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference(inputs: List[Any], **hyperparameter_config: Any) -> list[Any]:
        """Return a list of model outputs for a list of inputs."""

        inference_results = []
        for input_text in inputs:
            messages = [
                {
                    "role": "system",
                    "content": """
                        Answer the multiple choice question by selecting the correct letter
                        (A, B, C, D, etc.). Provide ONLY the letter of your answer,
                        no additional text or explanation.
                    """,
                },
                {"role": "user", "content": input_text},
            ]

            output = pipeline(messages)
            inference_results.append(output[0]["generated_text"][-1]["content"])

        return inference_results

    # === Creating Evaluation Datasets from YAML ===

    # Load Cais-MMLU dataset from YAML configuration
    cais_mmlu = EvalDataset.from_yaml("examples/example_yaml_configs/Cais-MMLU.yaml")
    print(f"Loaded {cais_mmlu.name} from YAML configuration.")

    # Load TIGER-Lab MMLU-Pro dataset from YAML configuration
    tiger_mmlu_pro = EvalDataset.from_yaml("examples/example_yaml_configs/TIGER-Lab-MMLU-Pro.yaml")
    print(f"Loaded {tiger_mmlu_pro.name} from YAML configuration.")

    # === Evaluation With YAML-Configured Datasets ===

    results = evaluate(
        inference,
        datasets=[
            cais_mmlu,
            tiger_mmlu_pro,
        ],
        sample_size=5,  # Sample size for quick testing on large datasets
        return_items=True,  # Include the scores for individual items evaluated in results
        return_output=True,  # Include the model responses for each evaluated item in item results
        upload_results=False,  # Disable uploading for this example
    )

    pprint(results)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="example_2.2")
    output_dir = setup_output_directory()
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "example_2.2_output.json")
