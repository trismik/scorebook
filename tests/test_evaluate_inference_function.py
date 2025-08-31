from pathlib import Path
from typing import Dict, List

from scorebook.evaluator import evaluate
from scorebook.metrics import Accuracy
from scorebook.types.eval_dataset import EvalDataset


def simple_inference_function(items: List[Dict], **hyperparameters) -> List[str]:
    """Inference function that always returns '1'."""
    return ["1" for _ in items]


async def async_inference_function(items: List[Dict], **hyperparameters) -> List[str]:
    """Async inference function that always returns '1'."""
    return ["1" for _ in items]


def test_evaluate_with_sync_inference_function():
    """Test evaluation with a synchronous inference function."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(simple_inference_function, dataset, return_sample_size=5)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_async_inference_function():
    """Test evaluation with an asynchronous inference function."""
    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(async_inference_function, dataset, return_sample_size=5)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_parametric_inference_function():
    """Test evaluation with an inference function that uses hyperparameters."""

    def parametric_inference_function(items: List[Dict], **hyperparameters) -> List[str]:
        output_value = hyperparameters.get("output", "1")
        return [output_value for _ in items]

    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(
        parametric_inference_function,
        dataset,
        hyperparameters={"output": "0"},
        return_sample_size=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]


def test_evaluate_with_minimal_inference_pipeline():
    """Test evaluation with an InferencePipeline that only has inference function."""
    from scorebook.types.inference_pipeline import InferencePipeline

    def simple_inference(items: List[Dict], **hyperparameters) -> List[str]:
        return ["1" for _ in items]

    # Create pipeline with only inference function (no preprocessor/postprocessor)
    pipeline = InferencePipeline(model="test_model", inference_function=simple_inference)

    dataset_path = str(Path(__file__).parent / "data" / "Dataset.csv")
    dataset = EvalDataset.from_csv(
        dataset_path, label="label", metrics=[Accuracy], name="test_dataset"
    )

    results = evaluate(pipeline, dataset, return_sample_size=5)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "accuracy" in results[0]
