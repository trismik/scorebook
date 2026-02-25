import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from scorebook import EvalDataset, evaluate
from scorebook.exceptions import (
    AllRunsFailedError,
    EvaluationError,
    InferenceError,
    ParameterValidationError,
)
from scorebook.inference.inference_pipeline import InferencePipeline
from scorebook.metrics.accuracy import Accuracy
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
    """Test evaluation raises InferenceError when all predictions are None."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    with pytest.raises(InferenceError, match="returned all None"):
        evaluate(
            create_simple_inference_pipeline(None), dataset, return_dict=False, upload_results=False
        )


def test_evaluate_invalid_inference_fn():
    """Test evaluation with an invalid inference pipeline raises InferenceError."""
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

    with pytest.raises(InferenceError):
        evaluate(bad_pipeline, dataset, return_dict=False, upload_results=False)


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
        assert "output" in headers
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
        assert "accuracy" in run.aggregate_scores
        assert run.error is None

    # Verify failed runs have error and None outputs/scores
    for run in failed_runs:
        assert run.outputs is None
        assert run.scores is None
        assert run.error is not None
        assert isinstance(run.error, InferenceError)

    # Check that hyperparameters are preserved correctly
    success_configs = [r.run_spec.hyperparameter_config["config_name"] for r in successful_runs]
    failure_configs = [r.run_spec.hyperparameter_config["config_name"] for r in failed_runs]

    assert set(success_configs) == {"success_1", "success_2", "success_3"}
    assert set(failure_configs) == {"failure_1", "failure_2"}

    # Verify warnings were issued for failed runs
    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 2


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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
        assert "accuracy" in run.aggregate_scores
        assert not run.run_spec.hyperparameter_config["fail_this_run"]
        assert run.error is None

    # Verify failed runs have error and None outputs/scores
    for run in failed_runs:
        assert run.outputs is None
        assert run.scores is None
        assert run.run_spec.hyperparameter_config["fail_this_run"]
        assert run.error is not None
        assert isinstance(run.error, InferenceError)

    # Verify warnings were issued for failed runs
    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 2


# ---------------------------------------------------------------------------
# Single-run error propagation (3 tests)
# ---------------------------------------------------------------------------


def _create_failing_pipeline():
    """Create a pipeline whose inference function raises ValueError."""

    def bad_inference_function(inputs: List, **hyperparameters):
        raise ValueError("original")

    return InferencePipeline(
        model="test_model",
        preprocessor=lambda x, **h: x,
        inference_function=bad_inference_function,
        postprocessor=lambda x, **h: x,
    )


def _create_sample_dataset():
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    return EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )


def test_evaluate_single_run_no_warning_issued():
    """Single-run failure raises without issuing warnings."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(InferenceError):
            evaluate(bad_pipeline, dataset, return_dict=False, upload_results=False)

    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 0


def test_evaluate_single_run_with_single_hyperparameter_config_raises():
    """A list of one hyperparameter config is still a single run — should raise."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    with pytest.raises(InferenceError):
        evaluate(
            bad_pipeline,
            dataset,
            hyperparameters=[{"temperature": 0.5}],
            return_dict=False,
            upload_results=False,
        )


def test_evaluate_single_run_exception_chain_preserved():
    """The InferenceError wraps the original exception via __cause__."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    with pytest.raises(InferenceError) as exc_info:
        evaluate(bad_pipeline, dataset, return_dict=False, upload_results=False)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "original" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# Multi-run partial failure (3 tests)
# ---------------------------------------------------------------------------


def _create_conditional_pipeline():
    """Create a pipeline that fails when fail_on_param=True."""

    def conditional_inference(inputs: List, **hyperparameters) -> List[str]:
        if hyperparameters.get("fail_on_param", False):
            raise ValueError("conditional failure")
        return ["1" for _ in inputs]

    return InferencePipeline(
        model="test_model",
        preprocessor=lambda x, **h: x,
        inference_function=conditional_inference,
        postprocessor=lambda x, **h: x,
    )


def test_evaluate_multi_run_exactly_two_runs_one_fails():
    """Smallest multi-run scenario: 2 configs, 1 fails, 1 succeeds."""
    dataset = _create_sample_dataset()
    pipeline = _create_conditional_pipeline()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = evaluate(
            pipeline,
            dataset,
            hyperparameters=[{"fail_on_param": False}, {"fail_on_param": True}],
            return_dict=False,
            upload_results=False,
        )

    assert len(results.run_results) == 2
    failed = [r for r in results.run_results if not r.run_completed]
    assert len(failed) == 1
    assert failed[0].error is not None

    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 1


def test_evaluate_multi_run_error_chain_preserved_on_result():
    """Failed run's error preserves the exception chain for debugging."""
    dataset = _create_sample_dataset()
    pipeline = _create_conditional_pipeline()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results = evaluate(
            pipeline,
            dataset,
            hyperparameters=[{"fail_on_param": False}, {"fail_on_param": True}],
            return_dict=False,
            upload_results=False,
        )

    failed_runs = [r for r in results.run_results if not r.run_completed]
    for run in failed_runs:
        assert isinstance(run.error, InferenceError)
        assert run.error.__cause__ is not None
        assert isinstance(run.error.__cause__, ValueError)


def test_evaluate_multi_run_return_dict_with_partial_failure():
    """return_dict=True includes failed runs in aggregates."""
    dataset = _create_sample_dataset()
    pipeline = _create_conditional_pipeline()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results = evaluate(
            pipeline,
            dataset,
            hyperparameters=[
                {"fail_on_param": False},
                {"fail_on_param": True},
                {"fail_on_param": False},
            ],
            return_dict=True,
            return_aggregates=True,
            return_items=True,
            upload_results=False,
        )

    # Aggregates include all runs
    assert len(results["aggregate_results"]) == 3
    failed_aggs = [r for r in results["aggregate_results"] if not r["run_completed"]]
    assert len(failed_aggs) == 1
    # error stays on dataclass, not in aggregate dict
    assert "error" not in failed_aggs[0]

    # Items only from successful runs (2 successful × 5 items)
    assert len(results["item_results"]) == 2 * 5


# ---------------------------------------------------------------------------
# Multi-run all fail (3 tests)
# ---------------------------------------------------------------------------


def test_evaluate_all_runs_fail_raises_all_runs_failed_error():
    """When ALL runs fail in a sweep, AllRunsFailedError is raised."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    with pytest.raises(AllRunsFailedError) as exc_info:
        evaluate(
            bad_pipeline,
            dataset,
            hyperparameters=[{"a": 1}, {"a": 2}],
            return_dict=False,
            upload_results=False,
        )

    assert len(exc_info.value.errors) == 2
    for desc, error in exc_info.value.errors:
        assert isinstance(error, InferenceError)


def test_evaluate_all_runs_fail_catchable_as_evaluation_error():
    """Verify AllRunsFailedError is catchable via except EvaluationError."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    try:
        evaluate(
            bad_pipeline,
            dataset,
            hyperparameters=[{"a": 1}, {"a": 2}],
            return_dict=False,
            upload_results=False,
        )
    except EvaluationError as e:
        assert isinstance(e, AllRunsFailedError)
        assert len(e.errors) == 2
    else:
        pytest.fail("Expected EvaluationError")


def test_evaluate_all_runs_fail_warnings_issued_before_raise():
    """Per-run warnings still fire even when AllRunsFailedError is raised."""
    dataset = _create_sample_dataset()
    bad_pipeline = _create_failing_pipeline()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(AllRunsFailedError):
            evaluate(
                bad_pipeline,
                dataset,
                hyperparameters=[{"a": 1}, {"a": 2}, {"a": 3}],
                return_dict=False,
                upload_results=False,
            )

    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 3


# ---------------------------------------------------------------------------
# Backwards compatibility (2 tests)
# ---------------------------------------------------------------------------


def test_evaluate_successful_single_run_unchanged():
    """Successful single-run produces same results, no warnings, no error."""
    dataset = _create_sample_dataset()
    pipeline = create_simple_inference_pipeline("1")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = evaluate(pipeline, dataset, return_dict=False, upload_results=False)

    assert result.run_results[0].run_completed
    assert result.run_results[0].error is None
    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 0


def test_evaluate_successful_multi_run_unchanged():
    """Successful multi-run produces same results, no warnings, no errors."""
    dataset = _create_sample_dataset()
    pipeline = create_simple_inference_pipeline("1")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = evaluate(
            pipeline,
            dataset,
            hyperparameters=[{"a": 1}, {"a": 2}],
            return_dict=False,
            upload_results=False,
        )

    for run in results.run_results:
        assert run.run_completed
        assert run.error is None
    eval_warnings = [x for x in w if "Evaluation run failed" in str(x.message)]
    assert len(eval_warnings) == 0
