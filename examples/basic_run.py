"""Example run."""

import string

import transformers

from scorebook.datasets import from_huggingface
from scorebook.metrics.precision import Precision

if __name__ == "__main__":
    dataset = from_huggingface("TIGER-Lab/MMLU-Pro")

    metric = Precision()

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    item = dataset[0]

    prompt = f"{item['question']}\nOptions:\n" + "\n".join(
        [f"{letter} : {choice}" for letter, choice in zip(string.ascii_uppercase, item["options"])]
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

    print(f"Prediction: {output}")
    print(f"Reference: {item['answer']}")

    score = metric.evaluate([output], [item["answer"]])

    print(score)
