import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from scorebook.eval_datasets import EvalDataset
from scorebook.evaluate import evaluate
from scorebook.exceptions import ParameterValidationError
from scorebook.inference.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy
from scorebook.types import ClassicEvalRunResult, EvalResult


def create_simple_inference_pipeline(expected_output: str = "1"):
    """Create a simple inference pipeline that always returns the same output."""

    def preprocessor(item: Dict, **hyperparameters) -> Dict:
        return item

    def inference_function(inputs: List, **hyperparameters) -> List[str]:
        return [expected_output for _ in inputs]

    def postprocessor(output: str, **hyperparameters) -> str:
        return output

    return InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=inference_function,
        postprocessor=postprocessor,
    )


def test_evaluate_single_dataset():
    """Test evaluation with a single CSV dataset."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_dict=False, upload_results=False
    )

    # With return_dict=False, we get an EvalResult object directly
    assert isinstance(results, EvalResult)

    # Check that we have one run result
    assert len(results.run_results) == 1
    eval_run_result = results.run_results[0]
    assert isinstance(eval_run_result, ClassicEvalRunResult)

    # Check aggregate metrics
    aggregate_scores = results.aggregate_scores
    assert isinstance(aggregate_scores, list)
    assert len(aggregate_scores) == 1
    assert "accuracy" in aggregate_scores[0]

    # Check per-item metrics
    item_scores = results.item_scores
    assert len(item_scores) == len(dataset.items)


def test_evaluate_multiple_datasets():
    """Test evaluation with multiple datasets."""
    csv_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    json_path = str(Path(__file__).parent.parent / "data" / "Dataset.json")

    csv_dataset = EvalDataset.from_csv(
        csv_path, metrics=[Accuracy], input="input", label="label", name="csv_dataset"
    )
    json_dataset = EvalDataset.from_json(
        json_path, metrics=[Accuracy], input="input", label="label", name="json_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"),
        [csv_dataset, json_dataset],
        return_dict=False,
        upload_results=False,
    )

    # With return_dict=False, we get an EvalResult object directly
    assert isinstance(results, EvalResult)
    # Should have 2 run results (one for each dataset)
    assert len(results.run_results) == 2

    # Check that both run results are ClassicEvalRunResult
    for run_result in results.run_results:
        assert isinstance(run_result, ClassicEvalRunResult)


def test_evaluate_with_item_limit():
    """Test evaluation with item limit."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"),
        dataset,
        sample_size=2,
        return_dict=False,
        upload_results=False,
    )

    # Check that we have the expected number of item scores
    assert len(results.item_scores) == 2


def test_evaluate_with_multiple_metrics():
    """Test evaluation with multiple metrics (Accuracy used twice for now)."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_dict=False, upload_results=False
    )

    # Check aggregate scores
    aggregate_scores = results.aggregate_scores
    assert "accuracy" in aggregate_scores[0]


def test_evaluate_with_none_predictions():
    """Test evaluation handling of None predictions."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline(None), dataset, return_dict=False, upload_results=False
    )

    # Check that all accuracy scores are False (None predictions should be wrong)
    item_scores = results.item_scores
    assert all(item["accuracy"] is False for item in item_scores)


def test_evaluate_invalid_inference_fn():
    """Test evaluation with an invalid inference pipeline."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    def bad_inference_function(inputs: List, **hyperparameters):
        raise ValueError("Inference error")

    bad_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x, h=None: x,
        inference_function=bad_inference_function,
        postprocessor=lambda x, h=None: x,
    )

    result = evaluate(bad_pipeline, dataset, return_dict=False, upload_results=False)

    # Should return a failed run result instead of raising exception
    assert len(result.run_results) == 1
    assert not result.run_results[0].run_completed


def test_evaluate_return_type():
    """Test different return types."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    # Test object return type (EvalResult)
    obj_results = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_dict=False, upload_results=False
    )
    assert isinstance(obj_results, EvalResult)
    assert len(obj_results.run_results) == 1

    # Test dict return type with different score_types
    # Test aggregate (default)
    dict_results = evaluate(
        create_simple_inference_pipeline("1"),
        dataset,
        return_dict=True,
        return_aggregates=True,
        return_items=False,
        upload_results=False,
    )
    assert isinstance(dict_results, list)
    assert "accuracy" in dict_results[0]  # Check first result has accuracy score

    # Test all
    dict_results_all = evaluate(
        create_simple_inference_pipeline("1"),
        dataset,
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        upload_results=False,
    )
    assert "aggregate_results" in dict_results_all
    assert "item_results" in dict_results_all
    assert isinstance(dict_results_all["aggregate_results"], list)
    assert isinstance(dict_results_all["item_results"], list)
    assert len(dict_results_all["aggregate_results"]) > 0
    assert len(dict_results_all["item_results"]) > 0

    # Test item
    dict_results_item = evaluate(
        create_simple_inference_pipeline("1"),
        dataset,
        return_dict=True,
        return_aggregates=False,
        return_items=True,
        upload_results=False,
    )
    assert isinstance(dict_results_item, list)
    assert len(dict_results_item) > 0


def test_evaluate_with_csv_export(tmp_path):
    """Test evaluation with results export to CSV."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_dict=False, upload_results=False
    )

    output_path = tmp_path / "evaluation_results.csv"
    # Export item scores to CSV
    df = pd.DataFrame(results.item_scores)
    df.to_csv(output_path, index=False)

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
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        create_simple_inference_pipeline("1"), dataset, return_dict=False, upload_results=False
    )

    output_path = tmp_path / "evaluation_results.json"
    # Export results to JSON
    result_data = {
        "aggregate_results": results.aggregate_scores,
        "item_results": results.item_scores,
    }
    with open(output_path, "w") as f:
        json.dump(result_data, f)

    assert output_path.exists()
    with open(output_path, "r") as f:
        data = json.load(f)
        assert "aggregate_results" in data
        assert "item_results" in data


def test_evaluate_invalid_param_config():
    """Test evaluation with invalid parameter combination."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    with pytest.raises(ParameterValidationError):
        evaluate(
            create_simple_inference_pipeline("1"),
            dataset,
            return_dict=True,
            return_aggregates=False,
            return_items=False,
            upload_results=False,
        )


def test_evaluate_duplicate_datasets():
    """Test that passing the same dataset multiple times preserves all results."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    # Pass the same dataset twice
    results = evaluate(
        create_simple_inference_pipeline("1"),
        [dataset, dataset],
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        upload_results=False,
    )

    # Should have results from both dataset runs
    assert len(results["aggregate_results"]) == 2
    assert len(results["item_results"]) == 10  # 5 items * 2 datasets

    # Both should have the same dataset name but be separate entries
    assert results["aggregate_results"][0]["dataset"] == "test_dataset"
    assert results["aggregate_results"][1]["dataset"] == "test_dataset"


def test_evaluate_with_precomputed_hyperparams():
    """Test evaluation with pre-computed hyperparameter grids."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
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
        return_dict=True,
        return_aggregates=True,
        return_items=True,
        upload_results=False,
    )

    # Should have results for each hyperparameter config
    assert "aggregate_results" in results
    assert "item_results" in results
    assert len(results["aggregate_results"]) == 3  # 3 hyperparameter configs
    assert len(results["item_results"]) == 15  # 5 items * 3 hyperparameter configs

    # Check that hyperparameters are correctly passed through
    assert results["aggregate_results"][0]["param1"] == "value1"
    assert results["aggregate_results"][1]["param1"] == "value2"
    assert results["aggregate_results"][2]["param1"] == "value3"


def test_evaluate_mixed_success_failure_runs():
    """Test that when some runs fail, others still complete successfully."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    def conditional_failing_inference_function(inputs: List, **hyperparameters) -> List[str]:
        """Inference function that fails only for specific hyperparameter values."""
        fail_on_param = hyperparameters.get("fail_on_param", False)
        if fail_on_param:
            raise ValueError("Intentional failure for testing")
        return ["1" for _ in inputs]

    def preprocessor(item: Dict, **hyperparameters) -> Dict:
        return item

    def postprocessor(output: str, **hyperparameters) -> str:
        return output

    failing_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=conditional_failing_inference_function,
        postprocessor=postprocessor,
    )

    # Test with multiple hyperparameter configs where some should fail
    hyperparams_configs = [
        {"config_name": "success_1", "fail_on_param": False},
        {"config_name": "failure_1", "fail_on_param": True},
        {"config_name": "success_2", "fail_on_param": False},
        {"config_name": "failure_2", "fail_on_param": True},
        {"config_name": "success_3", "fail_on_param": False},
    ]

    results = evaluate(
        failing_pipeline,
        dataset,
        hyperparameters=hyperparams_configs,
        return_dict=False,
        upload_results=False,
    )

    # Should have 5 run results (one for each hyperparameter config)
    assert len(results.run_results) == 5

    # Check that we have both successful and failed runs
    successful_runs = [r for r in results.run_results if r.run_completed]
    failed_runs = [r for r in results.run_results if not r.run_completed]

    assert len(successful_runs) == 3  # success_1, success_2, success_3
    assert len(failed_runs) == 2  # failure_1, failure_2

    # Verify successful runs have proper outputs and metrics
    for run in successful_runs:
        assert run.outputs is not None
        assert len(run.outputs) == 5  # 5 items in dataset
        assert all(output == "1" for output in run.outputs)
        assert run.scores is not None
        assert "accuracy" in run.scores

    # Verify failed runs have None outputs and metrics
    for run in failed_runs:
        # Failed runs should have None outputs and scores
        assert run.outputs is None
        assert run.scores is None

    # Check that hyperparameters are preserved correctly
    success_configs = [r.run_spec.hyperparameter_config["config_name"] for r in successful_runs]
    failure_configs = [r.run_spec.hyperparameter_config["config_name"] for r in failed_runs]

    assert set(success_configs) == {"success_1", "success_2", "success_3"}
    assert set(failure_configs) == {"failure_1", "failure_2"}


def test_evaluate_mixed_success_failure_multiple_datasets():
    """Test that when some runs fail across multiple datasets, others still complete."""
    csv_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    json_path = str(Path(__file__).parent.parent / "data" / "Dataset.json")

    csv_dataset = EvalDataset.from_csv(
        csv_path, metrics=[Accuracy], input="input", label="label", name="csv_dataset"
    )
    json_dataset = EvalDataset.from_json(
        json_path, metrics=[Accuracy], input="input", label="label", name="json_dataset"
    )

    def dataset_selective_failing_inference(inputs: List, **hyperparameters) -> List[str]:
        """Inference function that fails only when fail_this_run is True."""
        fail_this_run = hyperparameters.get("fail_this_run", False)
        if fail_this_run:
            raise ValueError("Intentional failure for testing")
        return ["1" for _ in inputs]

    failing_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x, **h: x,
        inference_function=dataset_selective_failing_inference,
        postprocessor=lambda x, **h: x,
    )

    # Use different hyperparameters for each dataset: one fails, one succeeds
    hyperparams_configs = [
        {"dataset_type": "csv", "fail_this_run": True},  # CSV should fail
        {"dataset_type": "json", "fail_this_run": False},  # JSON should succeed
    ]

    results = evaluate(
        failing_pipeline,
        [csv_dataset, json_dataset],
        hyperparameters=hyperparams_configs,
        return_dict=False,
        upload_results=False,
    )

    # Should have 4 run results (2 datasets × 2 hyperparameter configs)
    assert len(results.run_results) == 4

    # Separate successful and failed runs
    successful_runs = [r for r in results.run_results if r.run_completed]
    failed_runs = [r for r in results.run_results if not r.run_completed]

    # Should have 2 successful runs (both datasets with fail_this_run=False)
    # and 2 failed runs (both datasets with fail_this_run=True)
    assert len(successful_runs) == 2
    assert len(failed_runs) == 2

    # Verify successful runs have proper outputs and scores
    for run in successful_runs:
        assert run.outputs is not None
        assert len(run.outputs) == 5  # 5 items in dataset
        assert all(output == "1" for output in run.outputs)
        assert run.scores is not None
        assert "accuracy" in run.scores
        assert not run.run_spec.hyperparameter_config["fail_this_run"]

    # Verify failed runs have None outputs and scores
    for run in failed_runs:
        assert run.outputs is None
        assert run.scores is None
        assert run.run_spec.hyperparameter_config["fail_this_run"]
