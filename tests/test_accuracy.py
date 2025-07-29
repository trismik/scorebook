import pytest

from scorebook.metrics.accuracy import Accuracy


def test_accuracy_perfect_score():
    """Test Accuracy with all correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == 1.0
    assert all(items)
    assert len(items) == len(outputs)


def test_accuracy_zero_score():
    """Test Accuracy with all incorrect predictions."""
    outputs = ["A", "B", "C"]
    labels = ["X", "Y", "Z"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == 0.0
    assert all(not x for x in items)
    assert len(items) == len(outputs)


def test_accuracy_partial_score():
    """Test Accuracy with some correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "X", "C"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == 2 / 3
    assert items == [True, False, True]


def test_accuracy_empty_lists():
    """Test Accuracy with empty outputs and labels."""
    agg, items = Accuracy.score([], [])
    assert agg == 0.0
    assert items == []


def test_accuracy_mismatched_lengths():
    """Test Accuracy raises error when outputs and labels have different lengths."""
    outputs = ["A", "B"]
    labels = ["A"]

    with pytest.raises(ValueError):
        Accuracy.score(outputs, labels)
