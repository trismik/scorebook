"""Example run."""

import string
from pathlib import Path
from typing import Any

import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy

if __name__ == "__main__":

    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    def inference_function(eval_item: dict) -> Any:
        """Pre-processes dataset items, inferencing and post-processing result."""

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

        output = pipeline(messages)
        output = output[0]["generated_text"][-1]["content"]
        return output

    # Evaluate Phi-4-mimi-instruct using the MMLU-Pro Dataset.
    results = evaluate(inference_function, mmlu_pro, item_limit=10, return_type="object")
    mmlu_pro_eval_results = results["TIGER-Lab/MMLU-Pro"]

    # Save evaluation to a csv file.
    output_path = str(Path(__file__).parent / "results" / "basic_run_results.csv")
    mmlu_pro_eval_results.to_csv(output_path)

    # Save evaluation to a json file
    output_path = str(Path(__file__).parent / "results" / "basic_run_results.json")
    mmlu_pro_eval_results.to_json(output_path)

    # Print evaluation results
    print("\nAResults:")
    print(mmlu_pro_eval_results)

    # Print aggregate scores:
    print("\nAggregate Scores:")
    print(mmlu_pro_eval_results.aggregate_scores)

    # Print item scores:
    print("\nItem Scores:")
    for item_score in mmlu_pro_eval_results.item_scores:
        print(f"\n{item_score}")
