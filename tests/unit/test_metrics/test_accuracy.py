from scorebook.metrics.accuracy import Accuracy


def test_accuracy_perfect_score():
    """Test Accuracy with all correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == {"accuracy": 1.0}
    assert all(item["accuracy"] for item in items)
    assert len(items) == len(outputs)


def test_accuracy_zero_score():
    """Test Accuracy with all incorrect predictions."""
    outputs = ["A", "B", "C"]
    labels = ["X", "Y", "Z"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == {"accuracy": 0.0}
    assert all(not item["accuracy"] for item in items)
    assert len(items) == len(outputs)


def test_accuracy_partial_score():
    """Test Accuracy with some correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "X"]

    agg, items = Accuracy.score(outputs, labels)

    assert agg == {"accuracy": 2 / 3}
    assert items == [{"accuracy": True}, {"accuracy": True}, {"accuracy": False}]


def test_accuracy_empty_lists():
    """Test Accuracy with empty inputs."""
    outputs = []
    labels = []

    agg, items = Accuracy.score(outputs, labels)

    assert agg == {"accuracy": 0.0}
    assert items == []
