import csv
import json
from pathlib import Path
from typing import Dict, List

import pytest

from scorebook.evaluator import evaluate
from scorebook.metrics import Accuracy
from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult
from scorebook.types.inference_pipeline import InferencePipeline


def create_simple_inference_pipeline(expected_output: str = "1"):
    """Create a simple inference pipeline that always returns the same output."""

    def preprocessor(item: Dict, hyperparameters: Dict = None) -> Dict:
        return item

    def inference_function(processed_items: List[Dict], **hyperparameters) -> List[str]:
        return [expected_output for _ in processed_items]

    def postprocessor(output: str, hyperparameters: Dict = None) -> str:
        return output

    return InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )


def test_evaluate_single_dataset():
    """Test evaluation with a single CSV dataset."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="object")

    assert isinstance(results, dict)
    assert "test_dataset" in results
    eval_result = results["test_dataset"]
    assert isinstance(eval_result, EvalResult)

    # Check aggregate metrics
    assert isinstance(eval_result.aggregate_scores, dict)
    assert "accuracy" in eval_result.aggregate_scores

    # Check per-item metrics
    item_scores = eval_result.item_scores
    assert len(item_scores) == len(dataset.items)


def test_evaluate_multiple_datasets():
    """Test evaluation with multiple datasets."""
    csv_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    json_path = str(Path(__file__).parent / "data" / "Dataset.json")

    csv_dataset = EvalDataset.from_csv(
        csv_path, label="label", metrics=[Accuracy], name="csv_dataset"
    )
    json_dataset = EvalDataset.from_json(
        json_path, label="label", metrics=[Accuracy], name="json_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), [csv_dataset, json_dataset], return_type="object"
    )

    assert set(results.keys()) == {"csv_dataset", "json_dataset"}
    assert all(isinstance(r, EvalResult) for r in results.values())


def test_evaluate_with_item_limit():
    """Test evaluation with item limit."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), dataset, item_limit=2, return_type="object"
    )
    eval_result = results["test_dataset"]

    assert len(eval_result.item_scores) == 2


def test_evaluate_with_multiple_metrics():
    """Test evaluation with multiple metrics (Accuracy used twice for now)."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="object")
    eval_result = results["test_dataset"]

    assert "accuracy" in eval_result.aggregate_scores


def test_evaluate_with_none_predictions():
    """Test evaluation handling of None predictions."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_pipeline(None), dataset, return_type="object")
    eval_result = results["test_dataset"]

    assert all(item["accuracy"] is False for item in eval_result.item_scores)


def test_evaluate_invalid_inference_fn():
    """Test evaluation with an invalid inference pipeline."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    def bad_inference_function(processed_items: List[Dict], **hyperparameters):
        raise ValueError("Inference error")

    bad_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x, h=None: x,
        inference_function=bad_inference_function,
        postprocessor=lambda x, h=None: x,
    )

    with pytest.raises(ValueError):
        evaluate(bad_pipeline, dataset)


def test_evaluate_return_type():
    """Test different return types."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    # Test object return type
    obj_results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="object")
    assert isinstance(obj_results["test_dataset"], EvalResult)

    # Test dict return type with different score_types
    # Test aggregate (default)
    dict_results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="dict")
    assert isinstance(dict_results, list)
    assert "accuracy" in dict_results[0]  # Check first result has accuracy score

    # Test all
    dict_results_all = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_type="dict", score_type="all"
    )
    assert "aggregate" in dict_results_all
    assert "per_sample" in dict_results_all
    assert isinstance(dict_results_all["aggregate"], list)
    assert isinstance(dict_results_all["per_sample"], list)
    assert len(dict_results_all["aggregate"]) > 0
    assert len(dict_results_all["per_sample"]) > 0

    # Test item
    dict_results_item = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_type="dict", score_type="item"
    )
    assert isinstance(dict_results_item, list)
    assert len(dict_results_item) > 0


def test_evaluate_with_csv_export(tmp_path):
    """Test evaluation with results export to CSV."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="object")
    eval_result = results["test_dataset"]

    output_path = tmp_path / "evaluation_results.csv"
    eval_result.to_csv(output_path)

    assert output_path.exists()
    with open(output_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert "inference_output" in headers
        assert "accuracy" in headers
        data_rows = list(reader)
        assert len(data_rows) > 0


def test_evaluate_with_json_export(tmp_path):
    """Test evaluation with results export to JSON."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(create_simple_inference_pipeline("1"), dataset, return_type="object")
    eval_result = results["test_dataset"]

    output_path = tmp_path / "evaluation_results.json"
    eval_result.to_json(output_path)

    assert output_path.exists()
    with open(output_path, "r") as f:
        data = json.load(f)
        assert "aggregate" in data
        assert "per_sample" in data


def test_evaluate_invalid_score_type():
    """Test evaluation with invalid score type."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    with pytest.raises(ValueError):
        evaluate(create_simple_inference_pipeline("1"), dataset, score_type="invalid")


def test_evaluate_duplicate_datasets():
    """Test that passing the same dataset multiple times preserves all results."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    # Pass the same dataset twice
    results = evaluate(create_simple_inference_pipeline("1"), [dataset, dataset], score_type="all")

    # Should have results from both dataset runs
    assert len(results["aggregate"]) == 2
    assert len(results["per_sample"]) == 10  # 5 items per dataset * 2 datasets

    # Both should have the same dataset name but be separate entries
    assert results["aggregate"][0]["dataset_name"] == "test_dataset"
    assert results["aggregate"][1]["dataset_name"] == "test_dataset"


def test_evaluate_with_precomputed_hyperparams():
    """Test evaluation with pre-computed hyperparameter grids."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    # Create a list of pre-built hyperparameter configs
    precomputed_configs = [
        {"param1": "value1", "param2": 10},
        {"param1": "value2", "param2": 20},
        {"param1": "value3", "param2": 30},
    ]

    results = evaluate(
        create_simple_inference_pipeline("1"),
        dataset,
        hyperparameters=precomputed_configs,
        return_type="dict",
        score_type="all",
    )

    # Should have results for each hyperparameter config
    assert "aggregate" in results
    assert "per_sample" in results
    assert len(results["aggregate"]) == 3  # 3 hyperparameter configs
    assert len(results["per_sample"]) == 15  # 5 items * 3 hyperparameter configs

    # Check that hyperparameters are correctly passed through
    assert results["aggregate"][0]["param1"] == "value1"
    assert results["aggregate"][1]["param1"] == "value2"
    assert results["aggregate"][2]["param1"] == "value3"
