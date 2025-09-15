import asyncio
import time
from pathlib import Path
from typing import Dict, List

import pytest

from scorebook.eval_dataset import EvalDataset
from scorebook.evaluate import evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.metrics import Accuracy
from scorebook.types import ClassicEvalRunResult, EvalResult


def create_async_inference_pipeline(expected_output: str = "1", delay: float = 0.01):
    """Create an async inference pipeline that returns the same output after a delay."""

    def preprocessor(item: Dict, hyperparameters: Dict = None) -> Dict:
        return item

    async def async_inference_function(processed_items: List[Dict], **hyperparameters) -> List[str]:
        await asyncio.sleep(delay)  # Simulate async work
        return [expected_output for _ in processed_items]

    def postprocessor(output: str, hyperparameters: Dict = None) -> str:
        return output

    return InferencePipeline(
        model="test_model",
        preprocessor=preprocessor,
        inference_function=async_inference_function,
        postprocessor=postprocessor,
    )


def create_sync_inference_pipeline(expected_output: str = "1"):
    """Create a sync inference pipeline for comparison tests."""

    def preprocessor(item: Dict, hyperparameters: Dict = None) -> Dict:
        return item

    def sync_inference_function(processed_items: List[Dict], **hyperparameters) -> List[str]:
        return [expected_output for _ in processed_items]

    def postprocessor(output: str, hyperparameters: Dict = None) -> str:
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
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    return EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )


def test_evaluate_with_async_inference_function(sample_dataset):
    """Test evaluation with an async inference function."""
    async_inference_fn = create_async_inference_pipeline("1")

    results = evaluate(async_inference_fn, sample_dataset, return_dict=False, upload_results=False)

    # With return_dict=False, we get an EvalResult object directly
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 1
    eval_run_result = results.run_results[0]
    assert isinstance(eval_run_result, ClassicEvalRunResult)
    assert len(eval_run_result.outputs) > 0
    assert all(pred == "1" for pred in eval_run_result.outputs)


def test_evaluate_async_vs_sync_same_results(sample_dataset):
    """Test that async and sync inference functions produce identical results."""
    sync_inference_fn = create_sync_inference_pipeline("1")
    async_inference_fn = create_async_inference_pipeline("1")

    sync_results = evaluate(
        sync_inference_fn, sample_dataset, return_dict=False, upload_results=False
    )
    async_results = evaluate(
        async_inference_fn, sample_dataset, return_dict=False, upload_results=False
    )

    # Results should be identical
    assert len(sync_results.run_results) == len(async_results.run_results)

    sync_eval = sync_results.run_results[0]
    async_eval = async_results.run_results[0]

    assert sync_eval.outputs == async_eval.outputs
    assert sync_eval.scores == async_eval.scores


def test_evaluate_async_with_multiple_datasets():
    """Test evaluation with async inference function on multiple datasets."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset1 = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="dataset1"
    )
    dataset2 = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="dataset2"
    )

    async_inference_fn = create_async_inference_pipeline("1")
    results = evaluate(
        async_inference_fn, [dataset1, dataset2], return_dict=False, upload_results=False
    )

    # With return_dict=False, we get an EvalResult object directly
    assert isinstance(results, EvalResult)
    assert len(results.run_results) == 2  # Two datasets

    # Check that both run results are ClassicEvalRunResult
    for run_result in results.run_results:
        assert isinstance(run_result, ClassicEvalRunResult)
        assert len(run_result.outputs) > 0
        assert all(pred == "1" for pred in run_result.outputs)


def test_evaluate_async_with_item_limit(sample_dataset):
    """Test evaluation with async inference function and item limit."""
    async_inference_fn = create_async_inference_pipeline("1")
    item_limit = 3

    results = evaluate(
        async_inference_fn,
        sample_dataset,
        sample_size=item_limit,
        return_dict=False,
        upload_results=False,
    )

    # Check the number of item scores matches the limit
    assert len(results.item_scores) == item_limit


def test_evaluate_async_different_outputs(sample_dataset):
    """Test async inference pipeline that returns different outputs."""

    async def variable_async_inference_function(
        processed_items: List[Dict], **hyperparameters
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
        preprocessor=lambda x, h=None: x,
        inference_function=variable_async_inference_function,
        postprocessor=lambda x, h=None: x,
    )

    results = evaluate(variable_pipeline, sample_dataset, return_dict=False, upload_results=False)

    # Get the run result
    eval_run_result = results.run_results[0]

    # Check that we got some variety in predictions
    predictions = eval_run_result.outputs
    assert len(set(predictions)) > 1  # Should have different predictions


def test_evaluate_async_with_dict_return_type(sample_dataset):
    """Test async inference function with dict return type."""
    async_inference_fn = create_async_inference_pipeline("1")

    results = evaluate(
        async_inference_fn,
        sample_dataset,
        return_dict=True,
        return_aggregates=True,
        return_items=False,
        upload_results=False,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    # Check that results contain expected structure
    first_result = results[0]
    assert "dataset" in first_result  # Changed from dataset_name to dataset
    assert "accuracy" in first_result


def test_evaluate_async_performance():
    """Test that async inference functions can handle reasonable load."""
    # Create a larger dataset for performance testing
    data = [{"input": f"test_{i}", "label": str(i % 2)} for i in range(50)]
    dataset = EvalDataset.from_list(name="perf_test", label="label", metrics=[Accuracy], data=data)

    async_inference_fn = create_async_inference_pipeline("1", delay=0.001)  # 1ms delay

    start_time = time.time()
    results = evaluate(async_inference_fn, dataset, return_dict=False, upload_results=False)
    end_time = time.time()

    # Should complete reasonably quickly (not more than 10 seconds for 50 items)
    assert end_time - start_time < 10.0

    # Check that we have 50 item scores
    assert len(results.item_scores) == 50


@pytest.mark.asyncio
async def test_async_inference_function_directly():
    """Test that we can call async inference pipeline directly."""
    pipeline = create_async_inference_pipeline("test_output")

    result = await pipeline.run([{"input": "test"}])
    assert result == ["test_output"]


def test_evaluate_with_failing_async_function(sample_dataset):
    """Test evaluation handles async pipelines that raise exceptions."""

    async def failing_async_inference_function(
        processed_items: List[Dict], **hyperparameters
    ) -> List[str]:
        await asyncio.sleep(0.001)
        raise ValueError("Simulated async function failure")

    failing_pipeline = InferencePipeline(
        model="test_model",
        preprocessor=lambda x, h=None: x,
        inference_function=failing_async_inference_function,
        postprocessor=lambda x, h=None: x,
    )

    result = evaluate(failing_pipeline, sample_dataset, return_dict=False, upload_results=False)

    # Should return a failed run result instead of raising exception
    assert len(result.run_results) == 1
    assert not result.run_results[0].run_completed
