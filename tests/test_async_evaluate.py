import asyncio
import time
from pathlib import Path
from typing import Dict, List

import pytest

from scorebook.evaluator import evaluate
from scorebook.metrics import Accuracy
from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult
from scorebook.types.inference_pipeline import InferencePipeline


def create_async_inference_pipeline(expected_output: str = "1", delay: float = 0.01):
    """Create an async inference pipeline that returns the same output after a delay."""

    def preprocessor(item: Dict) -> Dict:
        return item

    async def async_inference_function(
        processed_items: List[Dict], hyperparameters: Dict
    ) -> List[str]:
        await asyncio.sleep(delay)  # Simulate async work
        return [expected_output for _ in processed_items]

    def postprocessor(output: str) -> str:
        return output

    return InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=async_inference_function,
        postprocessor=postprocessor,
    )


def create_sync_inference_pipeline(expected_output: str = "1"):
    """Create a sync inference pipeline for comparison tests."""

    def preprocessor(item: Dict) -> Dict:
        return item

    def sync_inference_function(processed_items: List[Dict], hyperparameters: Dict) -> List[str]:
        return [expected_output for _ in processed_items]

    def postprocessor(output: str) -> str:
        return output

    return InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=sync_inference_function,
        postprocessor=postprocessor,
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    return EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )


def test_evaluate_with_async_inference_function(sample_dataset):
    """Test evaluation with an async inference function."""
    async_inference_fn = create_async_inference_pipeline("1")

    results = evaluate(async_inference_fn, sample_dataset, return_type="object")

    assert len(results) == 1
    assert "test_dataset" in results
    eval_result = results["test_dataset"]
    assert isinstance(eval_result, EvalResult)
    assert len(eval_result.inference_outputs) > 0
    assert all(pred == "1" for pred in eval_result.inference_outputs)


def test_evaluate_async_vs_sync_same_results(sample_dataset):
    """Test that async and sync inference functions produce identical results."""
    sync_inference_fn = create_sync_inference_pipeline("1")
    async_inference_fn = create_async_inference_pipeline("1")

    sync_results = evaluate(sync_inference_fn, sample_dataset, return_type="object")
    async_results = evaluate(async_inference_fn, sample_dataset, return_type="object")

    # Results should be identical
    assert len(sync_results) == len(async_results)

    sync_eval = sync_results["test_dataset"]
    async_eval = async_results["test_dataset"]

    assert sync_eval.inference_outputs == async_eval.inference_outputs
    assert sync_eval.metric_scores == async_eval.metric_scores


def test_evaluate_async_with_multiple_datasets():
    """Test evaluation with async inference function on multiple datasets."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset1 = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="dataset1"
    )
    dataset2 = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="dataset2"
    )

    async_inference_fn = create_async_inference_pipeline("1")
    results = evaluate(async_inference_fn, [dataset1, dataset2], return_type="object")

    assert len(results) == 2
    assert "dataset1" in results
    assert "dataset2" in results

    for dataset_name, eval_result in results.items():
        assert isinstance(eval_result, EvalResult)
        assert len(eval_result.inference_outputs) > 0
        assert all(pred == "1" for pred in eval_result.inference_outputs)


def test_evaluate_async_with_item_limit(sample_dataset):
    """Test evaluation with async inference function and item limit."""
    async_inference_fn = create_async_inference_pipeline("1")
    item_limit = 3

    results = evaluate(
        async_inference_fn, sample_dataset, item_limit=item_limit, return_type="object"
    )

    eval_result = results["test_dataset"]
    assert len(eval_result.inference_outputs) == item_limit


def test_evaluate_async_different_outputs(sample_dataset):
    """Test async inference pipeline that returns different outputs."""

    async def variable_async_inference_function(
        processed_items: List[Dict], hyperparameters: Dict
    ) -> List[str]:
        await asyncio.sleep(0.001)
        # Return different outputs based on input text content
        results = []
        for processed_item in processed_items:
            if "input" in processed_item and "favorite" in str(processed_item["input"]).lower():
                results.append("1")
            else:
                results.append("0")
        return results

    variable_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x: x,
        inference_function=variable_async_inference_function,
        postprocessor=lambda x: x,
    )

    results = evaluate(variable_pipeline, sample_dataset, return_type="object")
    eval_result = results["test_dataset"]

    # Check that we got some variety in predictions
    predictions = eval_result.inference_outputs
    assert len(set(predictions)) > 1  # Should have different predictions


def test_evaluate_async_with_dict_return_type(sample_dataset):
    """Test async inference function with dict return type."""
    async_inference_fn = create_async_inference_pipeline("1")

    results = evaluate(async_inference_fn, sample_dataset, return_type="dict")

    assert isinstance(results, list)
    assert len(results) > 0

    # Check that results contain expected structure
    first_result = results[0]
    assert "dataset_name" in first_result
    assert "accuracy" in first_result


def test_evaluate_async_performance():
    """Test that async inference functions can handle reasonable load."""
    # Create a larger dataset for performance testing
    data = [{"input": f"test_{i}", "label": str(i % 2)} for i in range(50)]
    dataset = EvalDataset.from_list(name="perf_test", label="label", metrics=[Accuracy], data=data)

    async_inference_fn = create_async_inference_pipeline("1", delay=0.001)  # 1ms delay

    start_time = time.time()
    results = evaluate(async_inference_fn, dataset, return_type="object")
    end_time = time.time()

    # Should complete reasonably quickly (not more than 10 seconds for 50 items)
    assert end_time - start_time < 10.0

    eval_result = results["perf_test"]
    assert len(eval_result.inference_outputs) == 50


@pytest.mark.asyncio
async def test_async_inference_function_directly():
    """Test that we can call async inference pipeline directly."""
    pipeline = create_async_inference_pipeline("test_output")

    result = await pipeline.run([{"input": "test"}], {})
    assert result == ["test_output"]


def test_evaluate_with_failing_async_function(sample_dataset):
    """Test evaluation handles async pipelines that raise exceptions."""

    async def failing_async_inference_function(
        processed_items: List[Dict], hyperparameters: Dict
    ) -> List[str]:
        await asyncio.sleep(0.001)
        raise ValueError("Simulated async function failure")

    failing_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x: x,
        inference_function=failing_async_inference_function,
        postprocessor=lambda x: x,
    )

    with pytest.raises(ValueError, match="Simulated async function failure"):
        evaluate(failing_pipeline, sample_dataset, return_type="object")
