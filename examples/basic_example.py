"""Example run."""

import argparse
import json
import string
from pathlib import Path
from typing import Any

import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy
from scorebook.types.inference_pipeline import InferencePipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "demo_results"),
        help=(
            "Directory to save evaluation outputs (CSV and JSON). "
            "Defaults to ./results in the current working directory."
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def preprocessor(eval_item: dict) -> list:
        """Convert evaluation item to model input format."""
        prompt = f"{eval_item['question']}\nOptions:\n" + "\n".join(
            [
                f"{letter} : {choice}"
                for letter, choice in zip(string.ascii_uppercase, eval_item["options"])
            ]
        )

        # The system message contains the instructions for the model. We ask the
        # model to adhere strictly to the instructions; the ability of a model to
        # do that is based on the quality and size of the model. We suggest to
        # always do a post-processing step to ensure the model adheres to the
        # instructions.
        messages = [
            {
                "role": "system",
                "content": """
                    Answer the question you are given using only a single letter \
                    (for example, 'A'). \
                    Do not use punctuation. \
                    Do not show your reasoning. \
                    Do not provide any explanation. \
                    Follow the instructions exactly and \
                    always answer using a single uppercase letter.

                    For example, if the question is "What is the capital of France?" and the \
                    choices are "A. Paris", "B. London", "C. Rome", "D. Madrid",
                    - the answer should be "A"
                    - the answer should NOT be "Paris" or "A. Paris" or "A: Paris"

                    Please adhere strictly to the instructions.
                """.strip(),
            },
            {"role": "user", "content": prompt},
        ]
        return messages

    def inference_function(processed_items: list[list], **hyperparameters: Any) -> list[Any]:
        """Run model inference on preprocessed items."""
        outputs = []
        for messages in processed_items:
            output = pipeline(messages)
            outputs.append(output)
        return outputs

    def postprocessor(model_output: Any) -> str:
        """Extract the final answer from model output."""
        return str(model_output[0]["generated_text"][-1]["content"])

    # Create inference pipeline
    inference_pipeline = InferencePipeline(
        model="microsoft/Phi-4-mini-instruct",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )

    # Evaluate Phi-4-mini-instruct using the MMLU-Pro Dataset.
    results = evaluate(inference_pipeline, mmlu_pro, item_limit=10)
    print(results)

    with open(output_dir / "output.json", "w") as output_file:
        json.dump(results, output_file, indent=4)
        print(f"Results saved in {output_dir / 'output.json'}")
