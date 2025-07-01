from pathlib import Path

from datasets import Dataset

from scorebook.loaders import from_csv, from_huggingface, from_json


def test_load_flat_dataset():
    json_dataset_path = Path(__file__).parent / "data" / "Dataset.json"
    data_flat = from_json(str(json_dataset_path))
    assert isinstance(data_flat, Dataset)
    assert len(data_flat) == 5
    assert "text" in data_flat.column_names
    assert "label" in data_flat.column_names


def test_load_split_dataset():
    json_dataset_dict_path = Path(__file__).parent / "data" / "DatasetDict.json"
    data_split = from_json(str(json_dataset_dict_path), split="train")
    assert isinstance(data_split, Dataset)
    assert len(data_split) == 5
    assert "text" in data_split.column_names
    assert "label" in data_split.column_names


def test_load_csv_dataset():
    csv_dataset_path = Path(__file__).parent / "data" / "Dataset.csv"
    data_csv = from_csv(str(csv_dataset_path))
    assert isinstance(data_csv, Dataset)
    assert len(data_csv) == 5
    assert "text" in data_csv.column_names
    assert "label" in data_csv.column_names


def test_load_huggingface_dataset():
    data_hf = from_huggingface("imdb", split="test")
    assert isinstance(data_hf, Dataset)
    assert len(data_hf) > 0
    assert "text" in data_hf.column_names
    assert "label" in data_hf.column_names
