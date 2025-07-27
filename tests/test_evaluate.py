import csv
import re
from pathlib import Path
from typing import Dict

import pytest
import transformers

from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy, Precision


def test_evaluate():
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Precision], split="validation"
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

    results = evaluate(inference_function, mmlu_pro, Precision, item_limit=10)
    print("\n=== RESULTS ===")
    for ds, dataset_results in results.items():
        print(f"DATASET: {ds}\n")
        for item in dataset_results["items"]:
            print(f"          Question: {item['dataset_item'].get('question')}")
            print(f"     Output Answer: {item['output']}")
            print(f"    Correct Answer: {item['dataset_item'].get('answer')}\n")

        print("Metrics:")
        for metric_name, score in dataset_results["metrics"].items():
            print(f"    {metric_name}: {score}\n")


def create_simple_inference_fn(expected_output: str = "1"):
    """Create a simple inference function that always returns the same output."""

    def inference_fn(model_input: Dict) -> str:
        return expected_output

    return inference_fn


def save_results_to_csv(results: Dict, output_path: str):
    """Save evaluation results to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Dataset", "Question", "Prediction", "Ground Truth", "Result"])

        # Write results for each dataset
        for dataset_name, dataset_results in results.items():
            for item in dataset_results["items"]:
                writer.writerow(
                    [
                        dataset_name,
                        item["dataset_item"].get("question", ""),
                        item["output"],
                        item["dataset_item"].get("label", ""),
                        (
                            "correct"
                            if item["output"] == item["dataset_item"].get("label")
                            else "incorrect"
                        ),
                    ]
                )


def test_evaluate_single_dataset():
    """Test evaluation with a single CSV dataset."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    results = evaluate(create_simple_inference_fn("1"), dataset)

    assert isinstance(results, dict)
    assert len(results) == 1
    assert "test_dataset" in results

    dataset_results = results["test_dataset"]
    assert "items" in dataset_results
    assert "metrics" in dataset_results
    assert len(dataset_results["items"]) == 5
    assert "precision" in dataset_results["metrics"]


def test_evaluate_multiple_datasets():
    """Test evaluation with multiple datasets."""
    csv_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    json_path = str(Path(__file__).parent / "data" / "Dataset.json")

    csv_dataset = EvalDataset.from_csv(
        csv_path, label="label", metrics=[Precision], name="csv_dataset"
    )
    json_dataset = EvalDataset.from_json(
        json_path, label="label", metrics=[Precision], name="json_dataset"
    )

    results = evaluate(create_simple_inference_fn("1"), [csv_dataset, json_dataset])

    assert isinstance(results, dict)
    assert len(results) == 2
    assert all(k in results for k in ["csv_dataset", "json_dataset"])


def test_evaluate_with_item_limit():
    """Test evaluation with item limit."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    results = evaluate(create_simple_inference_fn("1"), dataset, item_limit=2)

    assert len(results["test_dataset"]["items"]) == 2


def test_evaluate_with_multiple_metrics():
    """Test evaluation with multiple metrics."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision, Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_fn("1"), dataset)

    metrics = results["test_dataset"]["metrics"]
    assert "precision" in metrics
    assert "accuracy" in metrics


def test_evaluate_with_none_predictions():
    """Test evaluation handling of None predictions."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    results = evaluate(create_simple_inference_fn(None), dataset)

    assert all(item["output"] is None for item in results["test_dataset"]["items"])


def test_evaluate_mmlu_pro():
    """Test evaluation with MMLU-Pro dataset."""
    mmlu_pro = EvalDataset.from_huggingface(
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Precision], split="validation"
    )

    def inference_fn(model_input: Dict) -> str:
        options = model_input.get("options", [])
        return options[0] if options else None

    results = evaluate(inference_fn, mmlu_pro, item_limit=5)

    assert isinstance(results, dict)
    assert "TIGER-Lab/MMLU-Pro" in results
    assert len(results["TIGER-Lab/MMLU-Pro"]["items"]) == 5


def test_evaluate_invalid_inference_fn():
    """Test evaluation with an invalid inference function."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    def bad_inference_fn(model_input: Dict):
        raise ValueError("Inference error")

    with pytest.raises(ValueError):
        evaluate(bad_inference_fn, dataset)


def test_evaluate_return_type():
    """Test different return types."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    # Test dict return type
    dict_results = evaluate(create_simple_inference_fn("1"), dataset, return_type="dict")
    assert isinstance(dict_results, dict)
    assert isinstance(dict_results["test_dataset"], dict)

    # Test default return type
    default_results = evaluate(create_simple_inference_fn("1"), dataset)
    assert isinstance(default_results, dict)


def test_evaluate_score_types():
    """Test evaluation with different score type options."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    # Test aggregate scores
    aggregate_results = evaluate(create_simple_inference_fn("1"), dataset, score_type="aggregate")
    assert isinstance(aggregate_results["test_dataset"]["metrics"]["precision"], float)

    # Test item-level scores
    item_results = evaluate(create_simple_inference_fn("1"), dataset, score_type="item")
    assert isinstance(item_results["test_dataset"]["metrics"]["precision"], list)
    assert all(isinstance(x, str) for x in item_results["test_dataset"]["metrics"]["precision"])

    # Test combined scores
    all_results = evaluate(create_simple_inference_fn("1"), dataset, score_type="all")
    assert isinstance(all_results["test_dataset"]["metrics"]["precision"], dict)
    assert "aggregate" in all_results["test_dataset"]["metrics"]["precision"]
    assert "items" in all_results["test_dataset"]["metrics"]["precision"]


def test_evaluate_with_csv_export():
    """Test evaluation with results export to CSV."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision, Accuracy], name="test_dataset"
    )

    # Run evaluation
    results = evaluate(create_simple_inference_fn("1"), dataset)

    # Save results to CSV
    output_path = str(Path(__file__).parent / "results" / "evaluation_results.csv")
    Path(output_path).parent.mkdir(exist_ok=True)

    save_results_to_csv(results, output_path)

    # Verify CSV was created and contains data
    assert Path(output_path).exists()
    with open(output_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == ["Dataset", "Question", "Prediction", "Ground Truth", "Result"]

        # Verify we have data rows
        data_rows = list(reader)
        assert len(data_rows) > 0


def test_evaluate_invalid_score_type():
    """Test evaluation with invalid score type."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision], name="test_dataset"
    )

    with pytest.raises(ValueError):
        evaluate(create_simple_inference_fn("1"), dataset, score_type="invalid")
