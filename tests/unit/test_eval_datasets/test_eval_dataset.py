import os
from pathlib import Path

import pytest

from scorebook import EvalDataset
from scorebook.exceptions import DatasetConfigurationError, DatasetParseError
from scorebook.metrics.exactmatch import ExactMatch


def test_load_flat_dataset():
    json_dataset_path = Path(__file__).parent.parent / "data" / "Dataset.json"
    data_flat = EvalDataset.from_json(
        str(json_dataset_path), metrics=ExactMatch, input="input", label="label"
    )
    assert isinstance(data_flat, EvalDataset)
    assert len(data_flat) == 5
    assert "input" in data_flat.column_names
    assert "label" in data_flat.column_names


def test_load_split_dataset():
    json_dataset_dict_path = Path(__file__).parent.parent / "data" / "DatasetDict.json"
    data_split = EvalDataset.from_json(
        str(json_dataset_dict_path), metrics=ExactMatch, input="input", label="label", split="train"
    )
    assert isinstance(data_split, EvalDataset)
    assert len(data_split) == 5
    assert "input" in data_split.column_names
    assert "label" in data_split.column_names


def test_load_csv_dataset():
    csv_dataset_path = Path(__file__).parent.parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(
        str(csv_dataset_path), metrics=ExactMatch, input="input", label="label"
    )
    assert isinstance(data_csv, EvalDataset)
    assert len(data_csv) == 5
    assert "input" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_huggingface_dataset():
    """Test loading HuggingFace dataset (uses mock data, no network calls).

    Note: This test uses mocked HuggingFace data defined in tests/fixtures/mock_hf_datasets.py
    to avoid network dependencies and ensure fast, reliable tests.
    """
    data_hf = EvalDataset.from_huggingface(
        "imdb", metrics=ExactMatch, input="text", label="label", split="test"
    )
    assert isinstance(data_hf, EvalDataset)
    assert len(data_hf) > 0
    # Original columns are preserved when no templates are used
    assert "text" in data_hf.column_names
    assert "label" in data_hf.column_names
    # Verify the dataset tracks which columns to use
    assert data_hf.input == "text"
    assert data_hf.label == "label"


def test_nonexistent_files():
    # Test nonexistent CSV file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_csv("nonexistent.csv", metrics=ExactMatch, input="input", label="label")

    # Test nonexistent JSON file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_json("nonexistent.json", metrics=ExactMatch, input="input", label="label")

    # Test nonexistent YAML file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_yaml("nonexistent.yaml")


def test_invalid_split():
    json_dataset_path = Path(__file__).parent.parent / "data" / "DatasetDict.json"
    with pytest.raises(DatasetConfigurationError):
        EvalDataset.from_json(
            str(json_dataset_path),
            metrics=ExactMatch,
            input="input",
            label="label",
            split="testing",
        )


def test_metric_types():
    dataset_path = Path(__file__).parent.parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(
        str(dataset_path), metrics=[ExactMatch, "Accuracy"], input="input", label="label"
    )
    assert isinstance(data_csv, EvalDataset)
    assert len(data_csv) == 5
    assert "input" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_yaml_dataset():
    """Test loading dataset from YAML config (uses mock HuggingFace data).

    Note: The YAML config specifies a HuggingFace dataset path, but this test
    uses mocked data to avoid network dependencies. See conftest.py for details.
    """
    yaml_path = Path(__file__).parent.parent / "data" / "dataset_template.yaml"

    # Load dataset from YAML
    data_yaml = EvalDataset.from_yaml(str(yaml_path))

    # Verify the dataset was loaded correctly
    assert isinstance(data_yaml, EvalDataset)
    assert len(data_yaml) > 0
    # When templates are used, computed columns are added with "*" prefix
    assert "*input" in data_yaml.column_names
    assert "*label" in data_yaml.column_names
    # Verify the dataset tracks which columns to use
    assert data_yaml.input == "*input"
    assert data_yaml.label == "*label"
    # Original columns are preserved (question, options, answer, etc.)
    assert "question" in data_yaml.column_names
    assert "options" in data_yaml.column_names
    assert "answer" in data_yaml.column_names


def test_yaml_missing_required_fields(tmp_path):
    # Create a YAML file missing required fields
    yaml_content = """
name: "imdb"
split: "test"
"""
    yaml_path = tmp_path / "invalid_config.yaml"
    yaml_path.write_text(yaml_content)

    # Test that loading raises DatasetConfigurationError
    with pytest.raises(DatasetConfigurationError):
        EvalDataset.from_yaml(str(yaml_path))


def test_invalid_yaml_syntax(tmp_path):
    # Create a YAML file with invalid syntax
    yaml_content = """
name: "imdb"
label: "label"
metrics: [
  "accuracy"
  invalid:
"""
    yaml_path = tmp_path / "invalid_syntax.yaml"
    yaml_path.write_text(yaml_content)

    # Test that loading raises DatasetParseError
    with pytest.raises(DatasetParseError):
        EvalDataset.from_yaml(str(yaml_path))


# === Integration Tests (Optional - Skipped in CI) ===


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skip integration tests in CI to avoid network dependencies"
)
def test_load_real_huggingface_dataset():
    """Integration test that actually loads from HuggingFace Hub.

    This test is marked with @pytest.mark.integration and will NOT use mocks.
    It will be skipped in CI environments to avoid network dependencies.

    To run this test locally:
        pytest tests/test_eval_dataset.py::test_load_real_huggingface_dataset -v
    """
    # Load a small subset to keep it fast
    data_hf = EvalDataset.from_huggingface(
        "imdb", metrics=ExactMatch, input="text", label="label", split="test[:10]"
    )
    assert isinstance(data_hf, EvalDataset)
    assert len(data_hf) == 10  # We only loaded 10 samples
    assert "text" in data_hf.column_names
    assert "label" in data_hf.column_names
