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
    with pytest.raises(FileNotFoundError, match="File not found"):
        EvalDataset.from_csv("nonexistent.csv", label="label", metrics=Precision)

    # Test nonexistent JSON file
    with pytest.raises(FileNotFoundError, match="File not found"):
        EvalDataset.from_json("nonexistent.json", label="label", metrics=Precision)


def test_invalid_split():
    json_dataset_path = Path(__file__).parent / "data" / "DatasetDict.json"
    with pytest.raises(ValueError, match="Split 'testing' not found in JSON file"):
        EvalDataset.from_json(
            str(json_dataset_path), label="label", split="testing", metrics=Precision
        )
