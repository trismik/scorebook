from pathlib import Path

import pytest

from scorebook import EvalDataset
from scorebook.metrics import Precision


def test_load_flat_dataset():
    json_dataset_path = Path(__file__).parent / "data" / "Dataset.json"
    data_flat = EvalDataset.from_json(
        str(json_dataset_path), metrics=Precision, input="input", label="label"
    )
    assert isinstance(data_flat, EvalDataset)
    assert len(data_flat) == 5
    assert "input" in data_flat.column_names
    assert "label" in data_flat.column_names


def test_load_split_dataset():
    json_dataset_dict_path = Path(__file__).parent / "data" / "DatasetDict.json"
    data_split = EvalDataset.from_json(
        str(json_dataset_dict_path), metrics=Precision, input="input", label="label", split="train"
    )
    assert isinstance(data_split, EvalDataset)
    assert len(data_split) == 5
    assert "input" in data_split.column_names
    assert "label" in data_split.column_names


def test_load_csv_dataset():
    csv_dataset_path = Path(__file__).parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(
        str(csv_dataset_path), metrics=Precision, input="input", label="label"
    )
    assert isinstance(data_csv, EvalDataset)
    assert len(data_csv) == 5
    assert "input" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_huggingface_dataset():
    data_hf = EvalDataset.from_huggingface(
        "imdb", metrics=Precision, input="text", label="label", split="test"
    )
    assert isinstance(data_hf, EvalDataset)
    assert len(data_hf) > 0
    assert "input" in data_hf.column_names
    assert "label" in data_hf.column_names


def test_nonexistent_files():
    # Test nonexistent CSV file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_csv("nonexistent.csv", metrics=Precision, input="input", label="label")

    # Test nonexistent JSON file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_json("nonexistent.json", metrics=Precision, input="input", label="label")

    # Test nonexistent YAML file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_yaml("nonexistent.yaml")


def test_invalid_split():
    json_dataset_path = Path(__file__).parent / "data" / "DatasetDict.json"
    with pytest.raises(ValueError):
        EvalDataset.from_json(
            str(json_dataset_path), metrics=Precision, input="input", label="label", split="testing"
        )


def test_metric_types():
    dataset_path = Path(__file__).parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(
        str(dataset_path), metrics=[Precision, "Accuracy"], input="input", label="label"
    )
    assert isinstance(data_csv, EvalDataset)
    assert len(data_csv) == 5
    assert "input" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_yaml_dataset():
    yaml_path = Path(__file__).parent / "data" / "dataset_template.yaml"

    # Load dataset from YAML
    data_yaml = EvalDataset.from_yaml(str(yaml_path))

    # Verify the dataset was loaded correctly
    assert isinstance(data_yaml, EvalDataset)
    assert len(data_yaml) > 0
    # After transformation, dataset has only "input" and "label" columns
    assert "input" in data_yaml.column_names
    assert "label" in data_yaml.column_names
    # Verify it has exactly 2 columns
    assert len(data_yaml.column_names) == 2


def test_yaml_missing_required_fields(tmp_path):
    # Create a YAML file missing required fields
    yaml_content = """
name: "imdb"
split: "test"
"""
    yaml_path = tmp_path / "invalid_config.yaml"
    yaml_path.write_text(yaml_content)

    # Test that loading raises ValueError
    with pytest.raises(ValueError):
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

    # Test that loading raises ValueError
    with pytest.raises(ValueError):
        EvalDataset.from_yaml(str(yaml_path))
