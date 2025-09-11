"""Tests for trismik upload_classic_eval_run functionality.

Testing Modes:
1. Mock Mode (default): MOCK_TRISMIK_TESTS=true in .env
   - Uses mocks to test function logic without hitting real backend
   - Fast, safe, and isolated testing
   - Verifies data transformation and API call structure

2. Integration Mode: MOCK_TRISMIK_TESTS=false in .env
   - Tests against real Trismik backend (requires valid credentials)
   - Also set TEST_EXPERIMENT_ID and TEST_PROJECT_ID in .env
   - Slower but tests real API integration

Usage:
- Add to .env file: MOCK_TRISMIK_TESTS=false
- Add to .env file: TEST_EXPERIMENT_ID=your-experiment-id
- Add to .env file: TEST_PROJECT_ID=your-project-id
- Run: pytest tests/test_trismik_services/test_upload_classic_eval_run.py
"""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from scorebook.eval_dataset import EvalDataset
from scorebook.trismik_services.upload_classic_eval_run import upload_classic_eval_run
from scorebook.types import ClassicEvalRunResult, EvalRunSpec

# Load environment variables from .env file
load_dotenv()

# Set to True to use mocks, False to test real backend integration
# Can be set in .env file: MOCK_TRISMIK_TESTS=false
MOCK = os.getenv("MOCK_TRISMIK_TESTS", "true").lower() == "true"
print(f"🧪 Test Configuration: {'MOCK' if MOCK else 'INTEGRATION'} mode")


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = MagicMock(spec=EvalDataset)
    dataset.name = "test_dataset"
    return dataset


@pytest.fixture
def simple_eval_run_spec(mock_dataset):
    """Create a simple EvalRunSpec for testing."""
    return EvalRunSpec(
        dataset=mock_dataset,
        dataset_index=0,
        hyperparameter_config={"temperature": 0.7, "max_tokens": 100},
        hyperparameters_index=0,
        items=[
            {"question": "What is 2+2?", "id": 0},
            {"question": "What is the capital of France?", "id": 1},
            {"question": "What color is the sky?", "id": 2},
        ],
        labels=["4", "Paris", "blue"],
    )


@pytest.fixture
def simple_eval_run_result(simple_eval_run_spec):
    """Create a simple ClassicEvalRunResult for testing."""
    return ClassicEvalRunResult(
        run_spec=simple_eval_run_spec,
        outputs=["4", "Paris", "blue"],
        scores={"accuracy": 1.0, "f1_score": 0.95},
    )


@pytest.fixture
def structured_eval_run_result(simple_eval_run_spec):
    """Create a ClassicEvalRunResult with structured metric data."""
    return ClassicEvalRunResult(
        run_spec=simple_eval_run_spec,
        outputs=["4", "Paris", "blue"],
        scores={
            "accuracy": {
                "aggregate_scores": {"accuracy": 1.0, "total_correct": 3, "total_items": 3},
                "item_scores": [1.0, 1.0, 1.0],
            },
            "f1_score": {
                "aggregate_scores": {"f1_score": 0.95, "precision": 0.96, "recall": 0.94},
                "item_scores": [0.95, 0.94, 0.96],
            },
        },
    )


@pytest.fixture
def mock_trismik_response():
    """Create a mock Trismik response."""
    response = MagicMock()
    response.id = "test-run-123"
    return response


@pytest.fixture
def mock_adaptive_test(mock_trismik_response):
    """Create a mock AdaptiveTest."""
    mock_test = MagicMock()
    mock_test.submit_classic_eval_async = AsyncMock(return_value=mock_trismik_response)
    return mock_test


class TestUploadClassicEvalRun:
    """Test upload_classic_eval_run functionality."""

    @pytest.mark.asyncio
    async def test_simple_metrics(
        self, simple_eval_run_result, mock_adaptive_test, mock_trismik_response
    ):
        """Test upload with simple metric data."""
        print("\n=== Testing Simple Metrics Upload ===")
        if MOCK:
            print("Running in MOCK mode - using mocked backend")
            # Use mocks for testing
            with patch(
                "scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest",
                return_value=mock_adaptive_test,
            ):
                print("Uploading classic eval run with simple metrics...")
                response = await upload_classic_eval_run(
                    run=simple_eval_run_result,
                    experiment_id="exp-123",
                    project_id="proj-456",
                    model="gpt-4",
                    metadata={"test": "value"},
                )
                print(f"Upload completed successfully, response: {response}")

                # Verify response
                assert response == mock_trismik_response

                # Verify AdaptiveTest was called with correct arguments
                mock_adaptive_test.submit_classic_eval_async.assert_called_once()
                call_args = mock_adaptive_test.submit_classic_eval_async.call_args[0][0]

                # Verify request structure
                assert call_args.projectId == "proj-456"
                assert call_args.experimentName == "exp-123"
                assert call_args.datasetId == "test_dataset"
                assert call_args.modelName == "gpt-4"
                assert call_args.hyperparameters == {"temperature": 0.7, "max_tokens": 100}

                # Verify items
                assert len(call_args.items) == 3
                for i, item in enumerate(call_args.items):
                    assert item.datasetItemId == str(i)
                    assert item.modelInput == str(simple_eval_run_result.run_spec.items[i])
                    assert item.modelOutput == simple_eval_run_result.outputs[i]
                    assert item.goldOutput == simple_eval_run_result.run_spec.labels[i]
                    assert item.metrics["accuracy"] == 1.0
                    assert item.metrics["f1_score"] == 0.95

                # Verify metrics
                assert len(call_args.metrics) == 2
                metric_names = [m.metricId for m in call_args.metrics]
                assert "accuracy" in metric_names
                assert "f1_score" in metric_names

                for metric in call_args.metrics:
                    if metric.metricId == "accuracy":
                        assert metric.value == 1.0
                    elif metric.metricId == "f1_score":
                        assert metric.value == 0.95
        else:
            print("Running in INTEGRATION mode - using real backend")
            # Real backend integration test
            print("Uploading classic eval run to real Trismik backend...")
            response = await upload_classic_eval_run(
                run=simple_eval_run_result,
                experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
                project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
                model="gpt-4",
                metadata={"test": "integration"},
            )

            # Verify real response
            assert response is not None
            assert hasattr(response, "id")
            assert response.id is not None
            print(f"Integration test successful! Run ID: {response.id}")
        print("=== Simple Metrics Upload Test Complete ===\n")

    # @pytest.mark.asyncio
    # async def test_structured_metrics(self, structured_eval_run_result, mock_adaptive_test, mock_trismik_response):
    #     """Test upload with structured metric data."""
    #     print("\n=== Testing Structured Metrics Upload ===")
    #     if MOCK:
    #         print("Running in MOCK mode - using mocked backend")
    #         print("Testing structured metrics with aggregate_scores and item_scores...")
    #         with patch('scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest', return_value=mock_adaptive_test):
    #             response = await upload_classic_eval_run(
    #                 run=structured_eval_run_result,
    #                 experiment_id="exp-123",
    #                 project_id="proj-456",
    #                 model="gpt-4",
    #                 metadata={"test": "value"}
    #             )
    #
    #             print("Upload completed, verifying structured metrics processing...")
    #             # Verify response
    #             assert response == mock_trismik_response
    #
    #             call_args = mock_adaptive_test.submit_classic_eval_async.call_args[0][0]
    #
    #             # Verify items have correct item-level metrics
    #             assert len(call_args.items) == 3
    #             for i, item in enumerate(call_args.items):
    #                 assert item.datasetItemId == str(i)
    #                 # Check item-level scores from structured data
    #                 assert item.metrics["accuracy"] == 1.0  # from item_scores[i]
    #                 assert item.metrics["f1_score"] in [0.95, 0.94, 0.96]  # from item_scores[i]
    #
    #             # Verify aggregate metrics
    #             assert len(call_args.metrics) == 6  # 3 for accuracy + 3 for f1_score
    #             metric_ids = [m.metricId for m in call_args.metrics]
    #
    #             # Check accuracy metrics
    #             assert "accuracy" in metric_ids
    #             assert "accuracy_total_correct" in metric_ids
    #             assert "accuracy_total_items" in metric_ids
    #
    #             # Check f1_score metrics
    #             assert "f1_score" in metric_ids
    #             assert "f1_score_precision" in metric_ids
    #             assert "f1_score_recall" in metric_ids
    #
    #             # Verify specific values
    #             for metric in call_args.metrics:
    #                 if metric.metricId == "accuracy":
    #                     assert metric.value == 1.0
    #                 elif metric.metricId == "accuracy_total_correct":
    #                     assert metric.value == 3
    #                 elif metric.metricId == "accuracy_total_items":
    #                     assert metric.value == 3
    #                 elif metric.metricId == "f1_score":
    #                     assert metric.value == 0.95
    #                 elif metric.metricId == "f1_score_precision":
    #                     assert metric.value == 0.96
    #                 elif metric.metricId == "f1_score_recall":
    #                     assert metric.value == 0.94
    #     else:
    #         print("Running in INTEGRATION mode - using real backend")
    #         print("Uploading structured metrics to real Trismik backend...")
    #         response = await upload_classic_eval_run(
    #             run=structured_eval_run_result,
    #             experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
    #             project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
    #             model="gpt-4",
    #             metadata={"test": "integration"}
    #         )
    #
    #         # Verify real response
    #         assert response is not None
    #         assert hasattr(response, 'id')
    #         assert response.id is not None
    #         print(f"Integration test successful! Run ID: {response.id}")
    #     print("=== Structured Metrics Upload Test Complete ===\n")

    @pytest.mark.asyncio
    async def test_empty_labels(self, mock_dataset, mock_adaptive_test, mock_trismik_response):
        """Test upload when labels list is shorter than items."""
        print("\n=== Testing Empty Labels Handling ===")
        print("Testing case where labels list is shorter than items list...")
        run_spec = EvalRunSpec(
            dataset=mock_dataset,
            dataset_index=0,
            hyperparameter_config={"temperature": 0.7},
            hyperparameters_index=0,
            items=[{"question": "What is 2+2?"}, {"question": "What is the capital of France?"}],
            labels=["4"],  # Only one label for two items
        )

        eval_run_result = ClassicEvalRunResult(
            run_spec=run_spec, outputs=["4", "Paris"], scores={"accuracy": 0.5}
        )

        if MOCK:
            print("Running in MOCK mode - using mocked backend")
            with patch(
                "scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest",
                return_value=mock_adaptive_test,
            ):
                await upload_classic_eval_run(
                    run=eval_run_result,
                    experiment_id="exp-123",
                    project_id="proj-456",
                    model="gpt-4",
                    metadata={},
                )

                call_args = mock_adaptive_test.submit_classic_eval_async.call_args[0][0]

                print("Verifying that missing labels are handled correctly...")
                # Verify items
                assert len(call_args.items) == 2
                assert call_args.items[0].goldOutput == "4"
                assert call_args.items[1].goldOutput == ""  # Empty string for missing label
                print("Empty labels handled correctly - missing label converted to empty string")
        else:
            print("Running in INTEGRATION mode - using real backend")
            print("Uploading empty labels test to real Trismik backend...")
            response = await upload_classic_eval_run(
                run=eval_run_result,
                experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
                project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
                model="gpt-4",
                metadata={"test": "integration"},
            )

            # Verify real response
            assert response is not None
            assert hasattr(response, "id")
            assert response.id is not None
            print(f"Integration test successful! Run ID: {response.id}")
        print("=== Empty Labels Test Complete ===\n")

    # @pytest.mark.asyncio
    # async def test_mixed_metric_types(self, mock_dataset, mock_adaptive_test, mock_trismik_response):
    #     """Test upload with mixed simple and structured metrics."""
    #     print("\n=== Testing Mixed Metric Types ===")
    #     print("Testing combination of simple metrics (float) and structured metrics (dict)...")
    #     run_spec = EvalRunSpec(
    #         dataset=mock_dataset,
    #         dataset_index=0,
    #         hyperparameter_config={"temperature": 0.8},
    #         hyperparameters_index=0,
    #         items=[{"question": "Test question"}],
    #         labels=["Test answer"]
    #     )
    #
    #     eval_run_result = ClassicEvalRunResult(
    #         run_spec=run_spec,
    #         outputs=["Test response"],
    #         scores={
    #             "simple_metric": 0.85,  # Simple value
    #             "complex_metric": {     # Structured data
    #                 "aggregate_scores": {"score": 0.9},
    #                 "item_scores": [0.9]
    #             }
    #         }
    #     )
    #
    #     if MOCK:
    #         print("Running in MOCK mode - using mocked backend")
    #         with patch('scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest', return_value=mock_adaptive_test):
    #             await upload_classic_eval_run(
    #                 run=eval_run_result,
    #                 experiment_id="exp-123",
    #                 project_id="proj-456",
    #                 model="gpt-4",
    #                 metadata={}
    #             )
    #
    #             call_args = mock_adaptive_test.submit_classic_eval_async.call_args[0][0]
    #
    #             # Verify item has both metric types
    #             item = call_args.items[0]
    #             assert item.metrics["simple_metric"] == 0.85
    #             assert item.metrics["complex_metric"] == 0.9
    #
    #             # Verify aggregate metrics
    #             metric_ids = [m.metricId for m in call_args.metrics]
    #             assert "simple_metric" in metric_ids
    #             assert "complex_metric_score" in metric_ids
    #
    #             for metric in call_args.metrics:
    #                 if metric.metricId == "simple_metric":
    #                     assert metric.value == 0.85
    #                 elif metric.metricId == "complex_metric_score":
    #                     assert metric.value == 0.9
    #             print("Mixed metric types processed correctly")
    #     else:
    #         print("Running in INTEGRATION mode - using real backend")
    #         print("Uploading mixed metric types to real Trismik backend...")
    #         response = await upload_classic_eval_run(
    #             run=eval_run_result,
    #             experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
    #             project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
    #             model="gpt-4",
    #             metadata={"test": "integration"}
    #         )
    #
    #         # Verify real response
    #         assert response is not None
    #         assert hasattr(response, 'id')
    #         assert response.id is not None
    #         print(f"Integration test successful! Run ID: {response.id}")
    #     print("=== Mixed Metric Types Test Complete ===\n")

    @pytest.mark.asyncio
    async def test_string_conversion(self, mock_dataset, mock_adaptive_test, mock_trismik_response):
        """Test that items, outputs, and labels are properly converted to strings."""
        print("\n=== Testing String Conversion ===")
        print("Testing that complex objects, None, and primitives are converted to strings...")
        run_spec = EvalRunSpec(
            dataset=mock_dataset,
            dataset_index=0,
            hyperparameter_config={},
            hyperparameters_index=0,
            items=[{"complex": {"nested": "data"}}, 123, None],
            labels=[True, 456, {"answer": "complex"}],
        )

        eval_run_result = ClassicEvalRunResult(
            run_spec=run_spec, outputs=[{"response": "json"}, 789, False], scores={"accuracy": 0.33}
        )

        if MOCK:
            print("Running in MOCK mode - using mocked backend")
            with patch(
                "scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest",
                return_value=mock_adaptive_test,
            ):
                await upload_classic_eval_run(
                    run=eval_run_result,
                    experiment_id="exp-123",
                    project_id="proj-456",
                    model="gpt-4",
                    metadata={},
                )

                call_args = mock_adaptive_test.submit_classic_eval_async.call_args[0][0]

                # Verify all fields are converted to strings
                for item in call_args.items:
                    assert isinstance(item.datasetItemId, str)
                    assert isinstance(item.modelInput, str)
                    assert isinstance(item.modelOutput, str)
                    assert isinstance(item.goldOutput, str)

                # Check specific conversions
                assert call_args.items[0].modelInput == "{'complex': {'nested': 'data'}}"
                assert call_args.items[1].modelInput == "123"
                assert call_args.items[2].modelInput == "None"

                assert call_args.items[0].goldOutput == "True"
                assert call_args.items[1].goldOutput == "456"
                assert call_args.items[2].goldOutput == "{'answer': 'complex'}"
                print("All data types correctly converted to strings")
        else:
            print("Running in INTEGRATION mode - using real backend")
            print("Uploading string conversion test to real Trismik backend...")
            response = await upload_classic_eval_run(
                run=eval_run_result,
                experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
                project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
                model="gpt-4",
                metadata={"test": "integration"},
            )

            # Verify real response
            assert response is not None
            assert hasattr(response, "id")
            assert response.id is not None
            print(f"Integration test successful! Run ID: {response.id}")
        print("=== String Conversion Test Complete ===\n")

    @pytest.mark.asyncio
    async def test_logging_output(
        self, simple_eval_run_result, mock_adaptive_test, mock_trismik_response, caplog
    ):
        """Test that logging works correctly."""
        print("\n=== Testing Logging Output ===")

        if MOCK:
            print("Running in MOCK mode - using mocked backend")
            print("Testing that success messages are logged correctly...")
            # Set log level to capture INFO messages
            caplog.set_level(logging.INFO)

            with patch(
                "scorebook.trismik_services.upload_classic_eval_run.AdaptiveTest",
                return_value=mock_adaptive_test,
            ):
                await upload_classic_eval_run(
                    run=simple_eval_run_result,
                    experiment_id="exp-123",
                    project_id="proj-456",
                    model="gpt-4",
                    metadata={},
                )

                print("Verifying log messages...")
                # Check that success message was logged
                assert (
                    "Classic eval run uploaded successfully with run_id: test-run-123"
                    in caplog.text
                )
                print("Success message logged correctly")
        else:
            print("Running in INTEGRATION mode - using real backend")
            print("Testing logging with real backend integration...")
            # Set log level to capture INFO messages
            caplog.set_level(logging.INFO)

            response = await upload_classic_eval_run(
                run=simple_eval_run_result,
                experiment_id=os.getenv("TEST_EXPERIMENT_ID", "test-exp"),
                project_id=os.getenv("TEST_PROJECT_ID", "test-proj"),
                model="gpt-4",
                metadata={"test": "integration"},
            )

            # Verify real response
            assert response is not None
            assert hasattr(response, "id")
            assert response.id is not None

            # Check that success message was logged with real run ID
            assert (
                f"Classic eval run uploaded successfully with run_id: {response.id}" in caplog.text
            )
            print(f"Integration test successful! Run ID: {response.id}, logging verified")
        print("=== Logging Output Test Complete ===\n")
