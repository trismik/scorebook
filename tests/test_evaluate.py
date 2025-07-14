import re
from typing import Dict

import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Precision


def test_evaluate():
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", split="validation"
    )
    assert isinstance(mmlu_pro, EvalDataset)

    pipeline = transformers.pipeline(
        "text-generation", model="microsoft/Phi-4-mini-instruct", trust_remote_code=True
    )

    def inference_function(model_input: Dict):
        options = model_input.get("options", [])
        formatted_options = "\n".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))

        content = (
            model_input.get("question")
            + "Return the correct answer from the given options. "
            + "Wrap your answer in <answer> tags (example <answer>F</answer>). "
            + "Options: "
            + formatted_options
        )
        messages = [{"role": "user", "content": content}]
        outputs = pipeline(messages)
        response = outputs[0]["generated_text"][-1].get("content")
        answer_match = re.search(r"<answer>(.*?)</answer>", response)
        if answer_match:
            return answer_match.group(1)
        else:
            return None

    results = evaluate(inference_function, mmlu_pro, Precision, item_limit=2)
    print("\n=== RESULTS ===")
    for ds, dataset_results in results.items():
        print(f"DATASET: {ds}\n")
        for item in dataset_results.items:
            print(f"          Question: {item.dataset_item.get('question')}")
            print(f"     Output Answer: {item.output}")
            print(f"    Correct Answer: {item.dataset_item.get('answer')}\n")

        print("Metrics:")
        for metric_name, score in dataset_results.metrics.items():
            print(f"    {metric_name}: {score}\n")
