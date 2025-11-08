import asyncio
from pathlib import Path
from typing import List

from scorebook import EvalDataset, evaluate, evaluate_async
from scorebook.metrics.accuracy import Accuracy


def simple_inference_function(inputs: List, **hyperparameters) -> List[str]:
    """Inference function that always returns '1'."""
    return ["1" for _ in inputs]


async def async_inference_function(inputs: List, **hyperparameters) -> List[str]:
    """Async inference function that always returns '1'."""
    return ["1" for _ in inputs]


def test_evaluate_with_sync_inference_function():
    """Test evaluation with a synchronous inference function."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(simple_inference_function, dataset, sample_size=5, upload_results=False)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_async_inference_function():
    """Test evaluation with an asynchronous inference function."""
    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = asyncio.run(
        evaluate_async(async_inference_function, dataset, sample_size=5, upload_results=False)
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_parametric_inference_function():
    """Test evaluation with an inference function that uses hyperparameters."""

    def parametric_inference_function(inputs: List, **hyperparameters) -> List[str]:
        output_value = hyperparameters.get("output", "1")
        return [output_value for _ in inputs]

    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(
        parametric_inference_function,
        dataset,
        hyperparameters={"output": "0"},
        sample_size=5,
        upload_results=False,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_minimal_inference_pipeline():
    """Test evaluation with an InferencePipeline that only has inference function."""
    from scorebook.inference.inference_pipeline import InferencePipeline

    def simple_inference(inputs: List, **hyperparameters) -> List[str]:
        return ["1" for _ in inputs]

    # Create pipeline with only inference function (no preprocessor/postprocessor)
    pipeline = InferencePipeline(model="test_model", inference_function=simple_inference)

    dataset_path = str(Path(__file__).parent.parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, metrics=[Accuracy], input="input", label="label", name="test_dataset"
    )

    results = evaluate(pipeline, dataset, sample_size=5, upload_results=False)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]
