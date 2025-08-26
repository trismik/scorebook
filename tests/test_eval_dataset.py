from pathlib import Path

import pytest

from scorebook import EvalDataset
from scorebook.metrics import Precision


def test_load_flat_dataset():
    json_dataset_path = Path(__file__).parent / "data" / "Dataset.json"
    data_flat = EvalDataset.from_json(str(json_dataset_path), label="label", metrics=Precision)
    assert isinstance(data_flat, EvalDataset)
    assert len(data_flat) == 5
    assert "input" in data_flat.column_names
    assert "label" in data_flat.column_names


def test_load_split_dataset():
    json_dataset_dict_path = Path(__file__).parent / "data" / "DatasetDict.json"
    data_split = EvalDataset.from_json(
        str(json_dataset_dict_path), label="label", split="train", metrics=Precision
    )
    assert isinstance(data_split, EvalDataset)
    assert len(data_split) == 5
    assert "input" in data_split.column_names
    assert "label" in data_split.column_names


def test_load_csv_dataset():
    csv_dataset_path = Path(__file__).parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(str(csv_dataset_path), label="label", metrics=Precision)
    assert isinstance(data_csv, EvalDataset)
    assert len(data_csv) == 5
    assert "input" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_huggingface_dataset():
    data_hf = EvalDataset.from_huggingface("imdb", label="label", split="test", metrics=Precision)
    assert isinstance(data_hf, EvalDataset)
    assert len(data_hf) > 0
    assert "text" in data_hf.column_names
    assert "label" in data_hf.column_names


def test_nonexistent_files():
    # Test nonexistent CSV file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_csv("nonexistent.csv", label="label", metrics=Precision)

    # Test nonexistent JSON file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_json("nonexistent.json", label="label", metrics=Precision)

    # Test nonexistent YAML file
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_yaml("nonexistent.yaml")


def test_invalid_split():
    json_dataset_path = Path(__file__).parent / "data" / "DatasetDict.json"
    with pytest.raises(ValueError):
        EvalDataset.from_json(
            str(json_dataset_path), label="label", split="testing", metrics=Precision
        )


def test_metric_types():
    dataset_path = Path(__file__).parent / "data" / "Dataset.csv"
    data_csv = EvalDataset.from_csv(
        str(dataset_path), label="label", metrics=[Precision, "Accuracy"]
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
    assert "question" in data_yaml.column_names
    assert "answer" in data_yaml.column_names
    assert "options" in data_yaml.column_names
    assert data_yaml.prompt_template is not None
    assert "{{ question }}" in data_yaml.prompt_template


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
