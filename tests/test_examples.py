"""Tests for the example scripts to ensure they work correctly."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_example_1_simple_eval():
    """Test that example_1_simple_eval.py runs successfully."""
    # Add examples directory to Python path
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))

    try:
        # Import the example module
        import example_1_simple_eval

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock sys.argv to provide the --output-dir argument
            with patch("sys.argv", ["example_1_simple_eval.py", "--output-dir", temp_dir]):
                # Run the main function
                example_1_simple_eval.main()

                # Verify output file was created
                output_file = Path(temp_dir) / "example_1_output.json"
                assert output_file.exists(), "Output file was not created"

                # Verify output file contains valid JSON
                with open(output_file) as f:
                    results = json.load(f)

                # Basic validation of results structure
                assert isinstance(results, list), "Results should be a list"
                assert len(results) > 0, "Results should not be empty"
                assert isinstance(results[0], dict), "Each result should be a dictionary"
                assert "accuracy" in results[0], "Results should contain accuracy metric"

    finally:
        # Clean up sys.path
        if str(examples_dir) in sys.path:
            sys.path.remove(str(examples_dir))

        # Remove the module from cache to avoid conflicts
        if "example_1_simple_eval" in sys.modules:
            del sys.modules["example_1_simple_eval"]


def test_example_2_inference_pipelines():
    """Test that example_2_inference_pipelines.py runs successfully."""
    # Add examples directory to Python path
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))

    try:
        # Import the example module
        import example_2_inference_pipelines

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock sys.argv to provide the --output-dir argument
            with patch("sys.argv", ["example_2_inference_pipelines.py", "--output-dir", temp_dir]):
                # Run the main function
                example_2_inference_pipelines.main()

                # Verify output file was created
                output_file = Path(temp_dir) / "example_2_output.json"
                assert output_file.exists(), "Output file was not created"

                # Verify output file contains valid JSON
                with open(output_file) as f:
                    results = json.load(f)

                # Basic validation of results structure
                assert isinstance(results, list), "Results should be a list"
                assert len(results) > 0, "Results should not be empty"
                assert isinstance(results[0], dict), "Each result should be a dictionary"
                assert "accuracy" in results[0], "Results should contain accuracy metric"

    finally:
        # Clean up sys.path
        if str(examples_dir) in sys.path:
            sys.path.remove(str(examples_dir))

        # Remove the module from cache to avoid conflicts
        if "example_2_inference_pipelines" in sys.modules:
            del sys.modules["example_2_inference_pipelines"]


def test_example_4_hyperparameter_sweeps():
    """Test that example_4_hyperparameter_sweeps.py runs successfully."""
    # Add examples directory to Python path
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))

    try:
        # Import the example module
        import example_4_hyperparameter_sweeps

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock sys.argv to provide the --output-dir argument
            with patch(
                "sys.argv", ["example_4_hyperparameter_sweeps.py", "--output-dir", temp_dir]
            ):
                # Run the main function
                example_4_hyperparameter_sweeps.main()

                # Verify output file was created
                output_file = Path(temp_dir) / "example_4_output.json"
                assert output_file.exists(), "Output file was not created"

                # Verify output file contains valid JSON
                with open(output_file) as f:
                    results = json.load(f)

                # Basic validation of results structure for hyperparameter sweep
                assert isinstance(results, dict), (
                    "Hyperparameter sweep results should be a dict with "
                    "'aggregate' and 'per_sample' keys"
                )
                assert "aggregate" in results, "Results should have 'aggregate' key"
                assert "per_sample" in results, "Results should have 'per_sample' key"

                # Verify aggregate results structure
                aggregate_results = results["aggregate"]
                assert isinstance(aggregate_results, list), "Aggregate results should be a list"
                assert len(aggregate_results) == 4, "Should have 4 hyperparameter combinations"

                # Verify each aggregate result has the expected structure
                for result in aggregate_results:
                    assert isinstance(result, dict), "Each result should be a dictionary"
                    assert (
                        "max_new_tokens" in result
                    ), "Each result should have max_new_tokens hyperparameter"
                    assert (
                        "temperature" in result
                    ), "Each result should have temperature hyperparameter"
                    assert "accuracy" in result, "Each result should have accuracy metric"

    finally:
        # Clean up sys.path
        if str(examples_dir) in sys.path:
            sys.path.remove(str(examples_dir))

        # Remove the module from cache to avoid conflicts
        if "example_4_hyperparameter_sweeps" in sys.modules:
            del sys.modules["example_4_hyperparameter_sweeps"]
