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
        "TIGER-Lab/MMLU-Pro", label="answer", metrics=[Accuracy], split="validation"
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

    results = evaluate(inference_function, mmlu_pro, item_limit=10)
    print("\n=== RESULTS ===")
    for ds, dataset_results in results.items():
        print(f"DATASET: {ds}\n")
        for item in dataset_results["items"]:
            print(f"          Question: {item['item'].get('question')}")
            print(f"     Output Answer: {item['output']}")
            print(f"    Correct Answer: {item['label']}")
            print("    Item Metrics:", ", ".join(f"{k}: {v}" for k, v in item["scores"].items()))
            print()

        print("Aggregate Metrics:")
        for metric_name, score in dataset_results["scores"].items():
            print(f"    {metric_name}: {score:.4f}\n")


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
        writer.writerow(["Dataset", "Question", "Prediction", "Ground Truth", "Metric Scores"])

        # Write results for each dataset
        for dataset_name, dataset_results in results.items():
            for item in dataset_results["items"]:
                # Get question and label from the dataset item
                question = item["item"].get("question", "")
                label = item["label"]
                output = item["output"]

                # Format metric scores as a string
                metric_scores = ", ".join(f"{k}: {v}" for k, v in item["scores"].items())

                writer.writerow([dataset_name, question, output, label, metric_scores])


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
    assert "scores" in dataset_results
    assert len(dataset_results["items"]) == 5
    assert "precision" in dataset_results["scores"]


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

    metric_scores = results["test_dataset"]["scores"]
    assert "precision" in metric_scores
    assert "accuracy" in metric_scores


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
    assert isinstance(aggregate_results["test_dataset"]["scores"], dict)
    assert "precision" in aggregate_results["test_dataset"]["scores"]
    assert isinstance(aggregate_results["test_dataset"]["scores"]["precision"], float)

    # Test item-level scores
    item_results = evaluate(create_simple_inference_fn("1"), dataset, score_type="item")
    assert isinstance(item_results["test_dataset"]["scores"], list)
    assert len(item_results["test_dataset"]["scores"]) > 0
    assert isinstance(item_results["test_dataset"]["scores"][0], dict)
    assert "precision" in item_results["test_dataset"]["scores"][0]
    assert isinstance(item_results["test_dataset"]["scores"][0]["precision"], str)

    # Test combined scores
    all_results = evaluate(create_simple_inference_fn("1"), dataset, score_type="all")
    assert isinstance(all_results["test_dataset"]["scores"], dict)
    assert "aggregate" in all_results["test_dataset"]["scores"]
    assert "items" in all_results["test_dataset"]["scores"]
    assert isinstance(all_results["test_dataset"]["scores"]["aggregate"], dict)
    assert isinstance(all_results["test_dataset"]["scores"]["items"], list)
    assert "precision" in all_results["test_dataset"]["scores"]["aggregate"]
    assert isinstance(all_results["test_dataset"]["scores"]["items"][0], dict)
    assert "precision" in all_results["test_dataset"]["scores"]["items"][0]


def test_evaluate_with_csv_export():
    """Test evaluation with results export to CSV."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Precision, Accuracy], name="test_dataset"
    )

    # Run evaluation
    results = evaluate(create_simple_inference_fn("1"), dataset, return_type="object")

    # Save results to CSV
    output_path = str(Path(__file__).parent / "results" / "evaluation_results.csv")
    eval_result = results["test_dataset"]
    eval_result.to_csv(output_path)

    # Verify CSV was created and contains data
    assert Path(output_path).exists()
    with open(output_path, "r") as f:
        reader = csv.reader(f)

        # Check dataset name
        first_row = next(reader)
        assert first_row[0] == "Dataset Name:"
        assert first_row[1] == "test_dataset"

        # Skip empty row
        next(reader)

        # Check aggregate scores section
        assert next(reader)[0] == "Aggregate Scores:"
        metrics_found = []
        while True:
            row = next(reader)
            if not row:  # Empty row
                break
            metrics_found.append(row[0])
        assert "precision" in metrics_found
        assert "accuracy" in metrics_found

        # Check item results section
        assert next(reader)[0] == "Item Results:"
        headers = next(reader)
        assert "precision" in headers
        assert "accuracy" in headers

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
